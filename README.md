<div align="center">

# EqC: Equilibrium Energy Control for Introspective Lifelong Robot Learning

**Haodong Yang**<sup>1,\*</sup> &nbsp;
[**Runhan Huang**](https://hrh6666.github.io/)<sup>1,\*,†</sup> &nbsp;
**Bowen Zhang**<sup>1,2,\*</sup> &nbsp;
**Iofai Lei**<sup>1,\*</sup> &nbsp;
[**Zunzhe Zhang**](https://scholar.google.com/citations?user=2W320-8AAAAJ&hl=en)<sup>1</sup>
<br>
[**Xiangcheng Zhang**](https://scholar.google.com/citations?user=pBnAahcAAAAJ&hl=en)<sup>1</sup> &nbsp;
[**Shaoting Zhu**](https://zst1406217.github.io/)<sup>1,2</sup> &nbsp;
[**Linzhan Mou**](https://linzhanm.github.io/)<sup>3</sup> &nbsp;
[**Yicheng Liu**](https://mrmoore98.github.io/liuyicheng/)<sup>1,2</sup> &nbsp;
[**Hang Zhao**](https://hangzhaomit.github.io/)<sup>1,2,‡</sup>

<sup>1</sup>Tsinghua University &nbsp;&nbsp;
<sup>2</sup>Galaxea AI &nbsp;&nbsp;
<sup>3</sup>Princeton University

<sup>\*</sup>Equal contribution, random order &nbsp;&nbsp;
<sup>†</sup>Project lead &nbsp;&nbsp;
<sup>‡</sup>Corresponding author

</div>

<p align="center">
  <img src="assets/teaser.png" alt="Overview of Equilibrium Energy Control (EqC)" width="95%">
</p>

**Overview of EqC.** *Top:* EqC generates actions through energy-based refinement, while the geometry of the
refinement path reveals when autonomous execution is weakly supported. Energy-Gated DAgger uses this signal to trigger
expert takeover and subsequent policy improvement. *Bottom:* In real-world deployment, persistent U<sub>ΔE</sub>
alarms prompt timely assistance before failure, progressively converting assisted interventions into fully autonomous
execution through lifelong learning.

## Abstract

Long-term robot autonomy depends on more than effective action generation: a policy must recognize when its behavior is
no longer sufficiently supported and use assistance to improve. We introduce **Equilibrium Energy Control (EqC)**, a
framework for introspective lifelong robot learning built around an explicit conditional energy landscape. EqC casts
generative action refinement as equilibrium-seeking optimization, while the geometry of the refinement process reveals
whether the generated behavior remains consistent with the learned landscape. This competence signal enables
Energy-Gated DAgger, where persistent energy-consistency violations trigger expert takeover and the resulting recovery
trajectories are aggregated for subsequent policy updates. EqC integrates naturally into both Diffusion Transformer
policies and flow-matching-based vision-language-action (VLA) policies. Across simulation and real-world bimanual
manipulation, EqC matches or improves control performance and robustness while enabling timely intervention near the
policy's competence boundary.

This repository contains the official implementation of the EqC **Diffusion Transformer (DiT) policy on LIBERO**:
the equilibrium energy objective, energy-gradient action refinement, and the LIBERO training and evaluation pipeline.

## Installation

The code is tested with Python 3.10 and CUDA-enabled PyTorch.

```bash
conda create -n eqc python=3.10 && conda activate eqc
pip install -r requirements.txt

# CleanDiffuser (lightning branch)
git clone -b lightning https://github.com/CleanDiffuserTeam/CleanDiffuser.git
pip install -e CleanDiffuser
```

Then install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) following the CleanDiffuser documentation.
The pretrained T5 and DINOv2 encoders are downloaded from Hugging Face on the first run.

### Data

Convert the LIBERO datasets to the CleanDiffuser `.zarr` format and place them under a single directory:

```text
DATASET_ROOT/
├── libero_goal.zarr
├── libero_spatial.zarr
├── libero_object.zarr
└── libero_10.zarr
```

Supported task suites: `libero_goal`, `libero_spatial`, `libero_object`, and `libero_10`.

## Training

```bash
bash scripts/train_libero.sh DATASET_ROOT RUNS_DIR TASK_SUITE GPU_IDS [HYDRA_OVERRIDES...]

# example
bash scripts/train_libero.sh /path/to/libero_zarr ./runs libero_goal 0 training.steps=40000
```

The script is a thin wrapper around the Hydra entry point; the equivalent direct call is

```bash
python eqc.py mode=training task_suite=libero_goal \
  paths.dataset_root=/path/to/libero_zarr paths.save_root=./runs
```

Common overrides:

| Override | Description |
|---|---|
| `training.steps=40000` | Number of training steps |
| `training.dataloader.batch_size=128` | Global batch size (default 256) |
| `training.resume_ckpt=/path/to/ckpt` | Resume from a checkpoint |
| `seed=1` | Random seed |
| `devices='[0,1]'` | Multi-GPU training (also set `GPU_IDS=0,1`) |
| `wandb.enabled=true wandb.mode=online` | Log to Weights & Biases (disabled by default) |

## Evaluation

```bash
bash scripts/eval_libero.sh DATASET_ROOT CHECKPOINT TASK_SUITE GPU_IDS [HYDRA_OVERRIDES...]

# example
bash scripts/eval_libero.sh /path/to/libero_zarr ./runs/libero_goal/checkpoints/step=40000.ckpt libero_goal 0
```

The script reports the success of each trial, per-task success counts, and the overall success rate. For a quick
sanity check, evaluate a single task and initial state:

```bash
bash scripts/eval_libero.sh /path/to/libero_zarr /path/to/step=40000.ckpt libero_goal 0 \
  'inference.task_ids=[0]' inference.max_init_states=1
```

## Method Configuration

At inference, EqC starts from noise and refines the whole action chunk by gradient ascent on the learned energy,
`x ← x + η_k ∇_x E(x | o)`. The relevant settings in `conf/config.yaml` are:

- `sampler_config`: `gd`, energy-gradient refinement;
- `scheduler_config`: `custom`, a fixed 32-step step-size schedule `η_k` (decaying from 0.039 to 0.010);
- `stopper_config`: `null`, always uses the full refinement budget.

## Code Structure

```text
eqc.py                       Hydra entry point
conf/config.yaml             Training and inference configuration
eqc/model.py                 Energy model, EqC objective, and action refinement
eqc/pipeline.py              LIBERO data loading, training, and evaluation
policy/eqc/                  Transport and interpolation utilities
policy/eqc_samplers/         Energy-gradient refinement sampler
policy/eqc_step_schedulers/  Step-size schedule
policy/eqc_stoppers/         Stopping criterion
scripts/                     Training and evaluation wrappers
```

## Citation

BibTeX coming soon.

## License

This project is released under the [MIT License](LICENSE).

## Contact

For questions about the paper or code, please open an issue or contact Runhan Huang
(runhanhuang2004@gmail.com).

## Acknowledgements

This implementation builds on [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) and
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
