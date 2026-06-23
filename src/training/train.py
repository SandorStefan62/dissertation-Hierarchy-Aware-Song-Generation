"""
Training script: MuQ hierarchical conditioning for MusicGen.

Usage:
    python src/training/train.py --model facebook/musicgen-small --lr 1e-4 --epochs 5 --batch-size 8 --grad-accum 4 --scheduler exponential --num-workers 8
"""

import math
import mlflow
import multiprocessing
import sys
import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "audiocraft"))

mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)

multiprocessing.set_start_method("fork", force=True)

from audiocraft.models.musicgen import MusicGen
from src.model.hierarchical_conditioner1 import HierarchicalConditioner
from src.model.lora import apply_lora, lora_params, lora_state_dict, load_lora_state_dict
from src.data_loaders.songformdb_dataset import SongWindowDataset, load_split

PROCESSED_DATASETS_PATH = Path(r"<switch-this-to-your-path>")       # note: make sure to point the script to where the datasets are.

def patch_fuser(lm, cond_name: str = "muq") -> None:
    """Register a new cross-attention conditioning key in the ConditionFuser.

    MusicGen's fuser asserts that every key in condition_tensors is registered.
    We add "muq" at runtime so we can pass condition_tensors={"muq": (...)}
    without touching any audiocraft source file.
    """
    lm.fuser.fuse2cond.setdefault("cross", []).append(cond_name)
    lm.fuser.cond2fuse[cond_name] = "cross"


def compute_loss(lm_out, codes: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over valid, non-padded codebook positions.

    lm_out.logits: [B, K, T, card]
    lm_out.mask:   [B, K, T]  - True where predictions are valid (LM causal mask)
    codes:         [B, K, T]

    Zero-padding added by the dataset (F.pad fills with 0) is excluded by
    building a padding mask: frames where ALL K codebooks are 0 are padding.
    This prevents the model from learning to predict code-0 for silence.
    """
    # [B, T] - True where frame is real audio (at least one codebook != 0)
    real_frames = (codes != 0).any(dim=1, keepdim=True).expand_as(lm_out.mask)
    valid = lm_out.mask & real_frames                   # [B, K, T]
    logits  = lm_out.logits[valid]                      # [N, card]
    targets = codes[valid]                              # [N]
    return F.cross_entropy(logits, targets)


def compute_per_codebook_metrics(lm_out, codes: torch.Tensor) -> tuple[list[float], list[float]]:
    """Loss and top-1 accuracy per codebook, excluding padding frames.

    Returns:
        losses: list of length K, cross-entropy per codebook
        accs:   list of length K, top-1 accuracy per codebook
    """
    real_frames = (codes != 0).any(dim=1, keepdim=True).expand_as(lm_out.mask)
    losses, accs = [], []
    for k in range(codes.shape[1]):
        valid_k   = lm_out.mask[:, k, :] & real_frames[:, k, :]     # [B, T]
        logits_k  = lm_out.logits[:, k, :, :][valid_k]              # [N, card]
        targets_k = codes[:, k, :][valid_k]                         # [N]
        if valid_k.any():
            losses.append(F.cross_entropy(logits_k, targets_k).item())
            accs.append((logits_k.argmax(dim=-1) == targets_k).float().mean().item())
        else:
            losses.append(0.0)
            accs.append(0.0)
    return losses, accs


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, mode: str = "cosine",
                   cosine_decay_steps: int | None = None):
    """LR schedule: linear warmup then cosine decay (or flat hold).

    Uses SequentialLR over two standard PyTorch schedulers:
      - LinearLR:           ramp from 0 to base_lr over warmup_steps optimizer steps
      - CosineAnnealingLR:  decay base_lr to 0 over cosine_decay_steps steps,
                            then holds at eta_min=0 for the remainder
      - ConstantLR:         hold base_lr throughout (mode="flat")

    Args:
        warmup_steps:       optimizer steps for the linear warmup phase
        total_steps:        total optimizer steps across all epochs
        mode:               "cosine" | "flat"
        cosine_decay_steps: steps over which cosine decay runs; defaults to
                            total_steps - warmup_steps (original behaviour).
                            Set to e.g. 2 * steps_per_epoch to decay fast and
                            hold near-zero for the rest of training.
    """
    if warmup_steps == 0:
        if mode == "cosine":
            if cosine_decay_steps is None:
                cosine_decay_steps = max(1, total_steps)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=max(1, cosine_decay_steps), 
                eta_min=0.0
            )
        elif mode == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=0.99, 
                last_epoch=total_steps
            )
        else:
            return torch.optim.lr_scheduler.ConstantLR(
                optimizer, 
                factor=1.0, 
                total_iters=total_steps
            )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    if mode == "cosine":
        if cosine_decay_steps is None:
            cosine_decay_steps = max(1, total_steps - warmup_steps)
        post_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cosine_decay_steps),
            eta_min=0.0,
        )
    elif mode == "exponential":
        post_warmup = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.99,
            last_epoch=total_steps,
        )
    else:
        post_warmup = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=total_steps,
        )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, post_warmup],
        milestones=[warmup_steps],
    )


def build_datasets(
        batch_size: int, 
        num_workers: int, 
        model_name: str,
        datasets: tuple[str, ...] = None,
        train_txt: Path | None = None, 
        val_txt: Path | None = None
):
    model_slug = model_name.replace("/", "-")   # e.g. "facebook-musicgen-small"

    datasets_lower = {d.lower() for d in datasets}
    found_datasets = [dataset for dataset in PROCESSED_DATASETS_PATH.iterdir() if dataset.stem.lower() in datasets_lower]
    print(f"Looking for datasets {datasets_lower} in {PROCESSED_DATASETS_PATH}")
    print(f"Found: {[d.stem for d in found_datasets]}")

    loaded_train_datasets = []
    loaded_val_datasets   = []

    for found_dataset in found_datasets:
        found_dataset_root      = PROCESSED_DATASETS_PATH / found_dataset
        found_dataset_feat      = found_dataset_root / "muq_features"
        found_dataset_audio     = found_dataset_root / "audio"
        found_dataset_codes     = found_dataset_root / "codes" / model_slug
        found_dataset_train_ids = load_split(train_txt if train_txt else found_dataset_root / "train.txt")
        found_dataset_val_ids   = load_split(val_txt   if val_txt   else found_dataset_root / "val.txt")

        loaded_train_datasets.append(SongWindowDataset(found_dataset_feat, found_dataset_audio, codes_dir=found_dataset_codes, song_ids=found_dataset_train_ids))
        loaded_val_datasets.append(SongWindowDataset(found_dataset_feat, found_dataset_audio, codes_dir=found_dataset_codes, song_ids=found_dataset_val_ids))

    train_ds = ConcatDataset(loaded_train_datasets)
    val_ds   = ConcatDataset(loaded_val_datasets)

    # hx_feat   = songformdb_hx_processed_root / "muq_features"
    # hx_audio  = songformdb_hx_processed_root / "audio"
    # hx_codes  = songformdb_hx_codes_root / model_slug
    # hx_train_ids = load_split(train_txt if train_txt else songformdb_hx_processed_root / "train.txt")
    # hx_val_ids   = load_split(val_txt   if val_txt   else songformdb_hx_processed_root / "val.txt")

    # hook_feat  = songformdb_hook_processed_root / "muq_features"
    # hook_audio = songformdb_hook_processed_root / "audio"
    # hook_codes = songformdb_hook_codes_root / model_slug

    # print("Building train index (loads .pt files once to read duration_s) ...")
    # train_ds = ConcatDataset([
    #     SongWindowDataset(hx_feat, hx_audio, codes_dir=hx_codes, song_ids=hx_train_ids),
    #     # SongWindowDataset(hook_feat, hook_audio, codes_dir=hook_codes),
    # ])
    # print("Building val index ...")
    # val_ds = SongWindowDataset(hx_feat, hx_audio, codes_dir=hx_codes, song_ids=hx_val_ids)

    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader


def run_epoch(lm, conditioner, loader, optimizer, scheduler, device, amp_dtype,
              grad_accum_steps: int, train: bool, epoch: int, epochs: int,
              global_step: int = 0, log_interval: int = 50,
              trainable_params: list | None = None,
              null_cond: bool = False, cfg_dropout: float = 0.0,
              max_grad_norm: float = 1.0):
    conditioner.train(train)
    lm.train(train)

    total_loss      = 0.0
    rolling_loss    = 0.0
    last_grad_norm  = 0.0
    cb_losses_total: list[float] | None = None
    cb_accs_total:   list[float] | None = None
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, total=len(loader), disable=not train, desc=f"Epoch {epoch}/{epochs}")):
        codes          = batch["codes"].to(device)              # [B,  K, T_codes]
        local          = batch["local"].to(device)              # [B, 36, n_layers, 1024]
        contextual     = batch["contextual"].to(device)         # [B, 12, n_layers, 1024]
        global_emb     = batch["global"].to(device)             # [B,  1, n_layers, 1024]
        duration_s     = batch["duration_s"].to(device)         # [B]
        window_start_s = batch["window_start_s"].to(device)     # [B]

        grad_ctx = torch.no_grad() if not train else torch.enable_grad()
        with grad_ctx, torch.autocast(device_type=device.type, dtype=amp_dtype):
            if null_cond:
                B = codes.shape[0]
                cond_tokens = torch.zeros(B, conditioner.N_TOKENS, conditioner.output_dim, device=device)
                cond_mask   = torch.zeros(B, conditioner.N_TOKENS, dtype=torch.bool, device=device)
            else:
                cond_tokens, cond_mask = conditioner(
                    local, contextual, global_emb, duration_s, window_start_s
                )  # [B, 49, dim], [B, 49]
                if train and cfg_dropout > 0.0 and torch.rand(1).item() < cfg_dropout:
                    cond_tokens = torch.zeros_like(cond_tokens)
                    cond_mask   = torch.zeros(cond_mask.shape, dtype=torch.bool, device=device)

            lm_out = lm.compute_predictions(
                codes,
                conditions=[],
                condition_tensors={"muq": (cond_tokens, cond_mask)},
            )

            loss = compute_loss(lm_out, codes)

        if train:
            # loss has no grad_fn when cfg_dropout zeros cond_tokens (torch.zeros_like
            # breaks the graph). Without LoRA there is no other gradient path, so
            # null-cond steps are skipped so they shouldn't update the conditioner anyway.
            if loss.requires_grad:
                (loss / grad_accum_steps).backward()
            total_loss   += loss.item()
            rolling_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0:
                clip_params = trainable_params if trainable_params else conditioner.parameters()
                last_grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm).item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % log_interval == 0:
                mlflow.log_metrics({
                    "train_loss_step": rolling_loss / log_interval,
                    "learning_rate":   optimizer.param_groups[0]["lr"],
                    "grad_norm":       last_grad_norm,
                }, step=global_step + step)
                rolling_loss = 0.0
        else:
            total_loss += loss.item()
            cb_losses, cb_accs = compute_per_codebook_metrics(lm_out, codes)
            if cb_losses_total is None:
                cb_losses_total = [0.0] * len(cb_losses)
                cb_accs_total   = [0.0] * len(cb_accs)
            for k in range(len(cb_losses)):
                cb_losses_total[k] += cb_losses[k]
                cb_accs_total[k]   += cb_accs[k]

    # flush any remaining accumulated gradients at epoch end
    if train and (len(loader) % grad_accum_steps) != 0:
        clip_params = trainable_params if trainable_params else conditioner.parameters()
        last_grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm).item()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    n = len(loader)
    avg_cb_losses = [v / n for v in cb_losses_total] if cb_losses_total else None
    avg_cb_accs   = [v / n for v in cb_accs_total]   if cb_accs_total   else None
    return total_loss / n, avg_cb_losses, avg_cb_accs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model",              type=str,   default="facebook/musicgen-small")
    parser.add_argument("--n-layers",           type=int,   default=8)
    parser.add_argument("--hidden-dim",         type=int,   default=1024)
    parser.add_argument("--n-heads",            type=int,   default=8)
    parser.add_argument("--n-encoder-layers",   type=int,   default=2)
    parser.add_argument("--max-grad-norm",      type=float, default=1.0,
                        help="gradient clipping max norm (default 1.0)")
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--epochs",             type=int,   default=50)
    parser.add_argument("--batch-size",         type=int,   default=4)
    parser.add_argument("--grad-accum",         type=int,   default=4,
                        help="gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--warmup-steps",       type=int,   default=500)
    parser.add_argument("--cosine-decay-steps", type=int,   default=None,
                        help="Optimizer steps over which cosine decays to 0. "
                             "Defaults to total_steps-warmup (slow). "
                             "Set to e.g. 2*steps_per_epoch for fast decay then hold.")
    parser.add_argument("--scheduler",          type=str,   choices=["cosine", "flat", "exponential"], default="cosine",
                        help="LR schedule after warmup: cosine decay or flat")
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--save-dir",           type=Path,  default=Path("checkpoints"))
    parser.add_argument("--resume",             type=Path,  default=None,
                        help="path to a checkpoint .pt to resume from")
    parser.add_argument("--fresh-optimizer",    action="store_true", default=False,
                        help="discard saved optimizer/scheduler state on resume (use when resuming from a cosine-decayed checkpoint with a higher LR)")
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--experiment-name",    type=str,   default="muq-conditioner")
    parser.add_argument("--log-interval",       type=int,   default=50)
    parser.add_argument("--lora-rank",          type=int,   default=0,
                        help="LoRA rank for cross-attention; 0 = disabled")
    parser.add_argument("--lora-alpha",         type=float, default=8.0,
                        help="LoRA alpha scaling (effective scale = alpha / rank)")
    parser.add_argument("--lora-targets",       type=str,   default="k,v",
                        help="Comma-separated QKV targets for LoRA, e.g. 'k,v' or 'q,k,v'")
    parser.add_argument("--datasets",           type=str,   default="HX", 
                        help="Comma-separated datasets to load before training starts.")
    parser.add_argument("--train-txt",          type=Path,  default=None,
                        help="Override train split .txt (default: HX train.txt + Hook)")
    parser.add_argument("--val-txt",            type=Path,  default=None,
                        help="Override val split .txt (default: HX val.txt)")
    parser.add_argument("--offset-emb",         action="store_true", default=False,
                        help="Add window-offset positional embedding to conditioner")
    parser.add_argument("--cfg-dropout",         type=float, default=0.0,
                        help="Probability of zeroing out MuQ conditioning per batch (CFG training). "
                             "Recommended: 0.1 to 0.2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f"Loading {args.model} ...")
    musicgen = MusicGen.get_pretrained(args.model, device=device)
    lm = musicgen.lm

    patch_fuser(lm)

    for p in lm.parameters():
        p.requires_grad_(False)

    if args.lora_rank > 0:
        targets = tuple(args.lora_targets.split(","))
        apply_lora(lm, rank=args.lora_rank, alpha=args.lora_alpha, targets=targets)

    # Use bfloat16 autocast for the forward pass - ~2x faster.
    # autocast handles the mixed-precision LM correctly (LayerNorm stays fp32
    # per PyTorch's op whitelist), so we no longer need lm.float().
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"AMP dtype: {amp_dtype}")

    print(f"LM dim: {lm.dim}  |  n_q: {lm.n_q}  |  card: {lm.card}")

    conditioner = HierarchicalConditioner(
        n_layers=args.n_layers,
        muq_dim=1024,
        hidden_dim=args.hidden_dim,
        output_dim=lm.dim,
        # n_heads=args.n_heads,
        # n_encoder_layers=args.n_encoder_layers,
        use_offset_emb=args.offset_emb,
    ).to(device)

    n_conditioner_params = sum(p.numel() for p in conditioner.parameters())
    n_lora_params        = sum(p.numel() for p in lora_params(lm))
    n_params             = n_conditioner_params + n_lora_params
    print(f"Conditioner params: {n_conditioner_params:,}  |  LoRA params: {n_lora_params:,}  |  Total trainable: {n_params:,}")

    trainable_params = list(conditioner.parameters()) + lora_params(lm)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)

    datasets = tuple(args.datasets.split(","))

    # Build datasets first so we know steps_per_epoch for the scheduler.
    train_loader, val_loader = build_datasets(args.batch_size, args.num_workers, args.model,
                                                              datasets=datasets,
                                                              train_txt=args.train_txt, val_txt=args.val_txt)
    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / args.grad_accum)
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs

    scheduler = make_scheduler(optimizer, args.warmup_steps, total_optimizer_steps, args.scheduler, args.cosine_decay_steps)
    decay_steps_msg = args.cosine_decay_steps if args.cosine_decay_steps else f"{total_optimizer_steps - args.warmup_steps} (default)"
    print(f"Scheduler: {args.scheduler}  warmup={args.warmup_steps}  cosine_decay={decay_steps_msg}  total={total_optimizer_steps} optimizer steps")

    start_epoch = 1
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        conditioner.load_state_dict(ckpt["conditioner"])
        if "lora" in ckpt and args.lora_rank > 0:
            load_lora_state_dict(lm, ckpt["lora"])
        if "optimizer" in ckpt and not args.fresh_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
            # override LR in case resuming with a different value than the saved run
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
            opt_msg = f"optimizer restored (LR overridden to {args.lr})"
        else:
            opt_msg = "optimizer fresh"
        if "scheduler" in ckpt and not args.fresh_optimizer:
            scheduler.load_state_dict(ckpt["scheduler"])
            sched_msg = "scheduler restored"
        else:
            sched_msg = "scheduler fresh"
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']} | {opt_msg} | {sched_msg}")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent.parent
    mlflow_db = repo_root / "experiments" / "mlflow" / "mlflow.db"
    mlflow_db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment(f"{args.experiment_name}-{args.model}")

    with mlflow.start_run(run_name=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", log_system_metrics=True):
        mlflow.log_params({
            **vars(args),
            "effective_batch_size":  args.batch_size * args.grad_accum,
            "n_params":              n_params,
            "lm_dim":                lm.dim,
            "amp_dtype":             str(amp_dtype),
            "device":                str(device),
            "total_optimizer_steps": total_optimizer_steps,
        })

        for epoch in range(start_epoch, args.epochs + 1):
            global_step = (epoch - 1) * steps_per_epoch
            train_loss, _, _ = run_epoch(
                lm, conditioner, train_loader,
                optimizer, scheduler, device, amp_dtype, args.grad_accum, train=True,
                epoch=epoch, epochs=args.epochs,
                global_step=global_step, log_interval=args.log_interval,
                trainable_params=trainable_params,
                cfg_dropout=args.cfg_dropout,
                max_grad_norm=args.max_grad_norm,
            )
            with torch.no_grad():
                val_loss, val_cb_losses, val_cb_accs = run_epoch(
                    lm, conditioner, val_loader,
                    optimizer, scheduler, device, amp_dtype, args.grad_accum, train=False,
                    epoch=epoch, epochs=args.epochs,
                )
                val_loss_null, null_cb_losses, _ = run_epoch(
                    lm, conditioner, val_loader,
                    optimizer, scheduler, device, amp_dtype, args.grad_accum, train=False,
                    epoch=epoch, epochs=args.epochs,
                    null_cond=True,
                )

            val_gain = val_loss_null - val_loss
            cb_gains = [n - c for n, c in zip(null_cb_losses, val_cb_losses)]

            print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | val_null {val_loss_null:.4f} | gain {val_gain:+.4f} | cb_gains {[f'{g:+.4f}' for g in cb_gains]}")

            metrics = {
                "train_loss_epoch":      train_loss,
                "val_loss_epoch":        val_loss,
                "val_loss_null":         val_loss_null,
                "val_conditioning_gain": val_gain,
            }
            for k, (cb_loss, cb_acc, cb_gain) in enumerate(zip(val_cb_losses, val_cb_accs, cb_gains)):
                metrics[f"val_loss_cb{k}"]              = cb_loss
                metrics[f"val_top1_acc_cb{k}"]          = cb_acc
                metrics[f"val_conditioning_gain_cb{k}"] = cb_gain
            mlflow.log_metrics(metrics, step=epoch)

            ckpt_path = args.save_dir / f"epoch_{epoch:03d}.pt"
            ckpt = {
                "epoch":        epoch,
                "conditioner":  conditioner.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "args":         vars(args),
            }
            if args.lora_rank > 0:
                ckpt["lora"] = lora_state_dict(lm)
            torch.save(ckpt, ckpt_path)
            mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")

