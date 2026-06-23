# Hierarchical MuQ Conditioning for MusicGen

This repository contains the code for a dissertation project investigating whether
musical structure can improve long-form music generation. The core idea: extract a
hierarchical structural representation of an input song using a self-supervised music
foundation model (MuQ), then use it to condition MusicGen's language model via
cross-attention, guiding generation to preserve the song's structural identity.

The motivation is the "boss theme problem" -- MusicGen generates coherent music at
30 seconds but loses musical identity over longer durations. A structural conditioning
signal derived from the full song gives the model a continuous reference to anchor
generation against.

---

## Repository layout

```
src/
  mfms/              MuQ feature extraction
  model/             Conditioner architecture and LoRA
  data_loaders/      Dataset and EnCodec pre-computation
  training/          Training loop
  analysis/          Evaluation and visualisation
  scripts/           End-to-end pipelines
src/audiocraft/      AudioCraft (Meta) -- vendored, unmodified
```

---

## 1. MuQ hierarchical feature extraction

MuQ (from `muq` on PyPI) is a self-supervised music foundation model trained on a
masked-prediction objective over spectral features. Its internal hidden states carry
rich harmonic, timbral, and structural information at multiple levels of abstraction.

For each song, three scales of features are extracted by `src/mfms/get_embeddings.py`:

| Scale | Tokens | Window | Content |
|---|---|---|---|
| Global | 1 | Full song (mean pool) | Overall tonal identity |
| Contextual | 12 | 30 s each | Section-level structure |
| Local | 36 | 10 s each | Phrase-level detail |

All windows are drawn from the first 360 s of the song; shorter songs produce fewer
valid tokens, tracked via a duration mask. The last `n_layers` (default 8) hidden
states are saved, giving each token shape `[n_layers, 1024]`.

Features are saved as `.pt` files under `datasets/processed/{dataset}/muq_features/`:

```
{
  "global":      Tensor[1,  n_layers, 1024],
  "contextual":  Tensor[12, n_layers, 1024],
  "local":       Tensor[36, n_layers, 1024],
  "duration_s":  float,
}
```

**Usage:**

```bash
python src/mfms/get_embeddings.py --dataset HX
```

`src/analysis/visualize_muq_features.py` produces four diagnostic plots for a single
song's feature file: pairwise cosine similarity between the 49 tokens (reveals section
boundaries), token L2 norms (reveals silence/padding), a PCA of all tokens, and --
if a checkpoint is provided -- the conditioner's learned layer-importance weights.

```bash
python src/analysis/visualize_muq_features.py \
    --feature-pt datasets/processed/HX/muq_features/HX_0082_dragosteadintei.pt \
    --checkpoint checkpoints/epoch_025.pt
```

---

## 2. Conditioner architecture

`src/model/hierarchical_conditioner.py` maps the three feature scales into a sequence
of 49 conditioning tokens for MusicGen's cross-attention.

**Per token:**
1. Learned softmax-weighted sum over the `n_layers` axis -- the model learns which MuQ
   layers carry the most useful information.
2. LayerNorm.
3. Add a learned scale-type embedding (global / contextual / local), giving the LM
   positional context about what each token represents.
4. Shared two-layer MLP: `muq_dim -> hidden_dim -> output_dim`.

The output is `[B, 49, output_dim]` plus a boolean mask `[B, 49]` that marks tokens
whose window start exceeds the song's actual duration. The mask is forwarded to
MusicGen's cross-attention so padded windows are ignored.

`output_dim` must match MusicGen's internal dimension: 1024 for `musicgen-small`,
1536 for `musicgen-medium`, 2048 for `musicgen-large`.

---

## 3. LoRA

`src/model/lora.py` adapts MusicGen's cross-attention layers to the conditioning task.
AudioCraft uses a fused `in_proj_weight [3*dim, dim]` parameter (not a standard
`nn.Linear`), so standard PEFT tooling does not find it. The implementation works
around this by registering a `forward_pre_hook` on each cross-attention module: the
hook intercepts the key/value projections before the fused operation and adds the
low-rank update (`delta = (x @ A.T) @ B.T`, scaled by `alpha / rank`) on the fly.

Targets can be configured as any subset of `q`, `k`, `v`, `o`. Best results were
obtained with `k,v`.

LoRA is optional -- passing `--lora-rank 0` (the default) trains only the conditioner.

---

## 4. Pre-computing EnCodec tokens

Training requires EnCodec-compressed token sequences for each song. Pre-compute and
cache them once before training to avoid running the encoder on every step:

```bash
python src/data_loaders/precompute_codes.py \
    --audio-dir  datasets/processed/HX/audio \
    --output-dir datasets/processed/HX/codes/facebook-musicgen-small
```

Each `.pt` file stores `{"codes": Tensor[K, T], "frame_rate": float}`.

---

## 5. Training

`src/training/train.py` trains the conditioner (and optionally LoRA) against
MusicGen's next-token prediction loss on the pre-computed EnCodec codes.

**Loop per batch:**
1. Load MuQ features and pass through the conditioner to get 49 conditioning tokens.
2. Optionally zero out the conditioning signal with probability `--cfg-dropout` to
   train the model to also function unconditioned (classifier-free guidance).
3. Pass codes and conditioning tokens to `lm.compute_predictions`, which runs
   MusicGen's transformer with the custom cross-attention signal injected.
4. Compute cross-entropy loss over valid (non-padding) token positions.
5. Backpropagate through the conditioner and optional LoRA layers; MusicGen's base
   weights remain frozen throughout.

The key validation metric is `val_conditioning_gain = val_loss_null - val_loss`.
A positive value confirms the conditioning signal carries information the model can
use. Per-codebook gain (cb0 = coarse structure through cb3 = fine acoustic detail)
is also logged. Training metrics are tracked with MLflow.

**Example run:**

```bash
python src/training/train.py \
    --lr 2e-4 \
    --epochs 50 \
    --scheduler flat \
    --cfg-dropout 0.2 \
    --lora-rank 8 \
    --lora-targets k,v \
    --experiment-name muq-conditioner
```

Full argument reference: `python src/training/train.py --help`

---

## 6. Dataset

The pipeline works with any collection of full-length audio files (.wav). The expected
directory layout per dataset is:

```
datasets/processed/{DatasetName}/
  audio/                        full-length .wav files
  muq_features/                 .pt feature files (one per song)
  codes/
    facebook-musicgen-small/    EnCodec .pt files (one per song)
  train.txt                     song IDs for training (one per line, no extension)
  val.txt                       song IDs for validation
```

**SongFormDB-HX** was used for this project. It provides full-length pop songs with
structural annotations (section boundaries and labels). Dataset available on
Hugging Face:
[https://huggingface.co/datasets/ASLP-lab/SongFormDB](https://huggingface.co/datasets/ASLP-lab/SongFormDB)

Note: structural annotations are only required to validate MuQ feature quality -- for
example, checking whether cosine similarity drops at section boundaries. Training and
generation do not use annotations.

`src/analysis/linear_probe.py` uses structural annotations to directly test whether
MuQ features are linearly separable by section label (verse, chorus, bridge, etc.).
A logistic regression is trained on the pooled token embeddings and evaluated with
cross-validation, giving a quantitative measure of how much structural information
the features encode before any conditioner training.

---

## 7. Inference and analysis

### Generating audio

`src/analysis/generate.py` loads a checkpoint and generates audio for a single song,
producing three variants: conditioned (the song's own MuQ features), unconditioned
(null conditioning), and noise-conditioned (random tokens). This gives a direct
within-song comparison of what the conditioning signal contributes.

```bash
python src/analysis/generate.py \
    --checkpoint checkpoints/epoch_025.pt \
    --feature-pt datasets/processed/HX/muq_features/HX_0082_dragosteadintei.pt \
    --out-dir experiments/generated/audio/my_run
```

`src/scripts/generate_songs.py` wraps this into a batch pipeline over a list of songs.

### MuQ-space evaluation

`src/scripts/analyze_generations.py` compares the MuQ features extracted from
generated audio against those of the original. It operates entirely in MuQ embedding
space and reports:

- **Cosine similarity** at global, contextual, and local granularity
- **Euclidean distance** as a complementary metric
- **Neighbour overlap** -- whether the nearest-neighbour structure is preserved
- **Borda-count pool ranking** -- how reliably the correct original can be identified
  from a pool of all originals using only the generated features

```bash
python src/scripts/analyze_generations.py \
    --muq-dir  experiments/generated/muq_features/my_run \
    --orig-dir datasets/processed/HX/muq_features \
    --summary-plot
```

### Raw audio evaluation

`src/analysis/analyze_audio_metrics.py` is an independent evaluation that uses only
raw audio features via librosa, with no reference to the MuQ embeddings used during
training. This avoids circularity with the training objective. Metrics per song
against the corresponding original:

- **MFCC cosine similarity** -- timbral identity (higher = better)
- **Spectral centroid difference** -- brightness match in Hz (lower = better)
- **Spectral rolloff difference** -- high-frequency energy match in Hz (lower = better)

```bash
python src/analysis/analyze_audio_metrics.py \
    --audio-dir experiments/generated/audio/my_run \
    --orig-dir  datasets/processed/HX/audio
```

Both scripts print per-song tables and an aggregate summary, and save a plot to the
output directory.
