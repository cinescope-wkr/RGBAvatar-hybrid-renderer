# RGBAvatar Technical Specification

Forked and modified codebase based on the CVPR 2025 paper:
`RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars`.

Paper: [RGBAvatar (CVPR 2025)](https://arxiv.org/pdf/2503.12886)  
Project page: [RGBAvatar Website](https://gapszju.github.io/RGBAvatar/)
GitHub repository: [gapszju/RGBAvatar](https://github.com/gapszju/RGBAvatar)

## Fork Notice

> [!NOTE]
> This repository is a fork of the RGBAvatar project.
> It is modified in part for Gaussian-mesh hybrid rendering research workflows while keeping the RGBAvatar training/rendering pipeline usable.
>
> **Fork maintainer**: Jinwoo Lee (cinescope@kaist.ac.kr)

Changes in this fork:
- `render.py` modernization:
  - Refactored into clearer functions (`parse_args`, `render_frames`, `render_orbit`, `build_dataset_and_model`, `main`).
  - CLI improved for practical rendering control (`batch_size`, `io_workers`, orbit controls, PLY range export).
  - Deformed PLY export simplified to a single consistent policy (direct render-time deformed geometry export).
- `model/reconstruction.py` cleanup:
  - Broken/garbled comments removed and rewritten in clear English.
  - Unused imports/legacy noise removed while preserving training behavior.
  - Loss blocks and scheduling logic documented and organized for maintainability.
- Configuration and training updates for dense fixed-topology runs:
  - Added/used surface bind regularization (`lambda_surface`), LPIPS warmup behavior, and tuned high-density defaults in `config/offline.yaml`.
- Documentation overhaul:
  - README rewritten as technical spec with explicit input/output contracts, dataset layout, command references, and external preprocessing links.

### External References

- [FLAME Model Download](https://flame.is.tue.mpg.de/download.php)
- [Metrical Photometric Tracker](https://github.com/Zielon/metrical-tracker)
- [Dataset Generation (INSTA)](https://github.com/Zielon/INSTA?tab=readme-ov-file#dataset-and-training)
- [Pretrained Avatar Models (OneDrive)](https://1drv.ms/u/c/c605a9d7c777e7ad/EX9KEcOnCgpOp_TWX0yCjO8BZlWfLv_Wbj3HDw6cPXwpIg?e=KJas7Z)

### Functional Capabilities

- UV raster-space binding of Gaussians to mesh faces via barycentric coordinates.
- Per-frame mesh deformation and Gaussian re-binding (`gaussian_deform_batch`).
- Blend-weight projection module (linear or MLP) for reduced Gaussian blendshape control.
- Fast-forward color initialization for uninitialized Gaussians.
- Training losses including Charbonnier/SSIM/LPIPS/alpha/sparsity/orth/normal/scale/surface-bind.
- Export and reload model as `.ply` with basis parameters.
- Utility conversion from 3DGS-style `.ply` to `.pt` tensor format (useful for LibTorch/C++ or custom runtime loaders that consume tensor checkpoints instead of PLY parsing).

## TL;DR (Quick Start)

```bash
# 1) Install
conda create -n rgbavatar python=3.10 -y && conda activate rgbavatar
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/NVlabs/nvdiffrast && pip install -r requirements.txt && pip install submodules/diff-gaussian-rasterization

# 2) Put data at: ./INSTA/<SUBJECT_NAME>/{images,checkpoint} and FLAME at: ./data/FLAME2020/generic_model.pkl
# 3) Train
python train_offline.py --subject <SUBJECT_NAME> --work_name <RUN_NAME> --config config/offline.yaml --split train --preload
# 4) Render
python render.py --subject <SUBJECT_NAME> --output_dir output --work_name <RUN_NAME>
```

## Quick Navigation

- [Fork Notice](#fork-notice)
- [1. Build and Dependency Requirements](#1-build-and-dependency-requirements)
- [2. Data and Asset Requirements](#2-data-and-asset-requirements)
- [3. Runtime Entry Points](#3-runtime-entry-points)
- [4. Input/Output Specification](#4-inputoutput-specification)
- [5. System Architecture and Frame/Data Flow](#5-system-architecture-and-framedata-flow)
- [6. Core Contracts](#6-core-contracts)
- [7. Configurations](#7-configurations)
- [8. Output and Artifact Layout](#8-output-and-artifact-layout)
- [9. Utilities](#9-utilities)
- [10. Resource Constraints and Limits](#10-resource-constraints-and-limits)
- [11. Error Diagnostics](#11-error-diagnostics)
- [12. File Responsibilities](#12-file-responsibilities)
- [13. Citation](#13-citation)

<details open>
<summary><strong>1. Build and Dependency Requirements</strong></summary>

## 1. Build and Dependency Requirements

### 1.1 Supported Host

- OS: Linux or Windows with CUDA-capable GPU (mainly tested on Windows 11).
- Python: 3.10 recommended.
- CUDA runtime: environment must match installed PyTorch CUDA build.

### 1.2 Required Dependencies

- PyTorch / TorchVision / Torchaudio (CUDA build recommended).
- `nvdiffrast`.
- Packages in `requirements.txt`.
- Local CUDA extension:
  - `submodules/diff-gaussian-rasterization` (installed as Python package).

### 1.3 Installation

```bash
git clone https://github.com/gapszju/RGBAvatar.git
cd RGBAvatar

conda create -n rgbavatar python=3.10
conda activate rgbavatar

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/NVlabs/nvdiffrast
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
```

</details>

<details open>
<summary><strong>2. Data and Asset Requirements</strong></summary>

## 2. Data and Asset Requirements

### 2.1 FLAME Model

- Download FLAME 2020 model from [FLAME website](https://flame.is.tue.mpg.de/download.php).
- Place `generic_model.pkl` under:
  - `data/FLAME2020/generic_model.pkl`

### 2.2 Offline Dataset Layout (INSTA-style)

```text
<DATA_DIR>/
  <SUBJECT_NAME>/
    checkpoint/   # per-frame tracked parameters
    images/       # RGBA or masked images
```

- Set `data_dir` in config files (`config/offline.yaml`, etc.) to `<DATA_DIR>`.

### 2.3 Subject Dataset Contract (What Must Exist)

For each `subject`, training/evaluation scripts resolve:

- `data_path = <data_dir>/<subject>`

With the current default in `config/offline.yaml`:

- `data_dir: ./INSTA`
- expected subject root:
  - `<repo_root>/INSTA/<SUBJECT_NAME>/`

Mandatory structure for `FLAMEDataset`:

```text
<repo_root>/
  INSTA/
    <SUBJECT_NAME>/
      checkpoint/
        00000.frame
        00001.frame
        ...
      images/
        00000.png
        00001.png
        ...
```

Required conditions:

- `checkpoint/00000.frame` must exist (camera intrinsics/extrinsics are loaded from frame 0).
- `images/` and `checkpoint/` must be frame-aligned by sorted filename order.
- Image count should match tracked frame count for clean training behavior.
- Alpha/mask channels should already be reflected in `images/` content produced by preprocessing.

### 2.4 How to Build Subject Data (Metrical Tracker -> INSTA)

This repo expects the final subject folder shown above. A practical pipeline is:

1. Run Metrical Tracker on a raw subject video.
1. Export per-frame FLAME tracking results.
1. Run INSTA preprocessing (`generate.sh` or equivalent) to produce masked head images.
1. Assemble outputs into this repo layout.

Role split by tool:

- Metrical Tracker:
  - Produces per-frame FLAME parameter files (`*.frame`) used as `checkpoint/`.
- INSTA:
  - Produces processed subject images used as `images/`.

Placement in this repo:

- Create `<repo_root>/INSTA/<SUBJECT_NAME>/`.
- Copy tracker outputs to:
  - `<repo_root>/INSTA/<SUBJECT_NAME>/checkpoint/`
- Copy INSTA image outputs to:
  - `<repo_root>/INSTA/<SUBJECT_NAME>/images/`

Then train with:

```bash
python train_offline.py --subject <SUBJECT_NAME> --config config/offline.yaml
```

Reference workflow from INSTA README:

```bash
# 1) Run Metrical Photometric Tracker for a selected actor
python tracker.py --cfg ./configs/actors/duda.yml

# 2) Generate dataset from absolute input/output paths
./generate.sh /metrical-tracker/output/duda INSTA/data/duda 100
#            {input}                        {output}       {# of test frames from the end}
```

Notes:
- INSTA recommends at least 1000 frames for training.
- You can also use [pretrained avatar models](https://1drv.ms/u/c/c605a9d7c777e7ad/EX9KEcOnCgpOp_TWX0yCjO8BZlWfLv_Wbj3HDw6cPXwpIg?e=KJas7Z).
- Equivalent subject data can be acquired by running Metrical Tracker + INSTA and arranging into this repository layout.

### 2.5 NeRSemble Notes

- Use preprocessed data compatible with `dataset/nersemble_ga_dataset.py`.
- Set FLAME 2023 path for NeRSemble flow as expected by script.

</details>

<details open>
<summary><strong>3. Runtime Entry Points</strong></summary>

## 3. Runtime Entry Points

### 3.1 Offline Training

```bash
python train_offline.py --subject SUBJECT_NAME --work_name WORK_NAME --config config/offline.yaml --split train --preload
```

Arguments:
- `--subject`: subject directory name.
- `--work_name`: experiment name (defaults to timestamp when omitted).
- `--config`: YAML path.
- `--split`: `train|test|all`.
- `--preload`: preload images to CPU memory.

### 3.2 Online Training

```bash
python train_online.py --subject SUBJECT_NAME --work_name WORK_NAME --config config/online.yaml --video_fps 25
```

### 3.3 Rendering (Single-view sequence + orbit)

```bash
python render.py --subject SUBJECT_NAME --output_dir output --work_name WORK_NAME
```

Optional:
- `--white_bg`
- `--alpha`
- `--batch_size`
- `--io_workers`
- `--save_ply_start`
- `--save_ply_end`
- `--save_ply_every`
- `--save_ply_interval`
- `--disable_orbit`
- `--orbit_anchor_idx`
- `--orbit_fps`
- `--orbit_duration_sec`
- `--orbit_yaw_deg`
- `--disable_orbit_gif`

Example (save deformed PLY only for frames 100 to 200):

```bash
python render.py --subject SUBJECT_NAME --output_dir output --work_name WORK_NAME \
  --save_ply_start 100 --save_ply_end 200 --save_ply_every 1
```

### 3.4 Evaluation

```bash
python calculate_metrics.py --subject SUBJECT_NAME --output_dir output --work_name WORK_NAME --split -350
```

### 3.5 NeRSemble

```bash
python train_offline_nersemble.py --subject SUBJECT_NAME --work_name WORK_NAME --config config/nersemble.yaml
python render_nersemble.py --subject SUBJECT_NAME --output_dir output --work_name WORK_NAME
```

</details>

<details open>
<summary><strong>4. Input/Output Specification</strong></summary>

## 4. Input/Output Specification

### A. Common Inputs 

| Input | Location / Form | Role |
|---|---|---|
| Subject frames | `<data_dir>/<subject>/images/` | GT supervision images for train/render/metrics. |
| Tracked params | `<data_dir>/<subject>/checkpoint/` | Per-frame FLAME/FuHead parameters to reconstruct/deform mesh. |
| Template model | `data/FLAME2020/generic_model.pkl` (or FuHead assets) | Base mesh topology and parametric deformation source. |
| Run config | `config/*.yaml` | Controls model density, optimizer, losses, data behavior, output path. |
| Optional pretrained model | `<output>/<subject>/<work_name>/model.ply` | Loads trained Gaussian state for render/eval. |

### B. Script-Level Inputs and Outputs

#### `train_offline.py`

Inputs:
- `--subject`, `--config`, `--split`, `--preload`
- Dataset frames/track params + template model
- Config sections:
  - `dataset`: loading behavior
  - `model`: Gaussian topology/initialization (`tex_size`, `init_scaling`, ...)
  - `train.recon`: optimizer/loss/schedule

Outputs:
- `<output>/<subject>/<work_name>/model.ply`
  - Trained Gaussian parameters and blend bases (primary deployable artifact).
- `<output>/<subject>/<work_name>/config.yaml`
  - Frozen run config snapshot for reproducibility.
- `<output>/<subject>/<work_name>/speed.txt`
  - Training throughput summary.
- TensorBoard logs (when enabled).

#### `train_online.py`

Inputs:
- Video-time frame order from dataset + sampler policy (`train.sampler`)
- `--video_fps` for stream pacing

Outputs:
- Same core outputs as offline (`model.ply`, `config.yaml`, `speed.txt`)
- `sample.txt`, `step2frame.txt`
  - Sampling history / mapping of optimization steps to source frame timeline.

#### `render.py`

Inputs:
- `model.ply` from a trained run
- Corresponding `config.yaml` in same run directory
- Original subject data path (resolved from saved config)

Outputs:
- `render_image/*.png`
  - Final per-frame rendered RGB (or RGBA with `--alpha`).
- `deformed_ply/deformed_*.ply` (optional interval/range based)
  - Per-frame deformed Gaussian geometry dump from the render pipeline.
- `render_orbit/*.png` + `orbit_animation.gif`
  - Novel-view orbit visualization.

#### `calculate_metrics.py`

Inputs:
- Trained `model.ply`
- Full dataset frames and masks

Outputs:
- `metrics.npz`
  - Framewise arrays: `l1_error`, `l2_error`, `psnr`, `ssim`, `lpips`.
- `metrics.txt`
  - Aggregated train/test split summary.

### C. Tensor-Level Input/Output Roles in Core Renderer

`GaussianAttributes` runtime contract:
- `xyz`: Gaussian center positions in world space after mesh binding.
- `opacity`: alpha contribution strength per Gaussian.
- `scaling`: anisotropic Gaussian size along local axes.
- `rotation`: orientation quaternion for anisotropic splats.
- `sh`: color representation (DC/SH channels used by rasterizer path).

Raster output (`render_gs_batch` / batch rasterizer):
- `color`: rendered RGB image tensor.
- `alpha`: rendered alpha/mask tensor.
- `est_color`: accumulated color estimator for fast-forward init.
- `est_weight`: visibility/accumulation weight used for init and loss weighting.
- `radii`: projected Gaussian size info (visibility/debug utility).

</details>

<details>
<summary><strong>5. System Architecture and Frame/Data Flow</strong></summary>

## 5. System Architecture and Frame/Data Flow

One training step in `model/reconstruction.py` executes in this order:

1. Batch fetch [CPU->GPU]
1. Background compositing
1. Blend gating by `blend_start_iter`
1. Local Gaussian attribute fetch (`get_batch_attributes`)
1. Mesh-bound Gaussian deformation (`gaussian_deform_batch`)
1. Rasterization (`BatchGaussianRenderer` / `render_gs_batch`)
1. Loss aggregation
1. Backprop + optimizer step
1. Optional fast-forward update

One rendering batch in `render.py` executes:

1. Load mesh and blend weights
1. Deform Gaussians for each frame in batch
1. Optional deformed `.ply` export path
1. Actual image rasterization (`render_gs_batch`)
1. Write images to `render_image/`

</details>

<details>
<summary><strong>6. Core Contracts</strong></summary>

## 6. Core Contracts

### 6.1 Binding Contract

- Binding is generated from UV rasterization (`compute_rast_info`).
- Each valid UV texel corresponds to one Gaussian.
- Each Gaussian stores:
  - `binding_face_id`
  - `binding_face_bary`
- Runtime deformation composes:
  - triangle-space offsets
  - face TBN rotation
  - local Gaussian transform

### 6.2 Gaussian Attribute Contract

`GaussianAttributes` fields:
- `xyz`: `[B,N,3]` or `[N,3]`
- `opacity`: `[B,N,1]` or `[N,1]`
- `scaling`: `[B,N,3]` or `[N,3]`
- `rotation`: `[B,N,4]` or `[N,4]`
- `sh`: `[B,N,1,3]`-compatible SH/DC layout in this repo flow

All runtime tensors are expected as `float32` on CUDA during render/train.

### 6.3 Loss Contract (Current Reconstruction Path)

Configured in `train.recon`:
- `lambda_charbonnier`
- `lambda_ssim`
- `lambda_lpips` (with warmup)
- `lambda_alpha`
- `lambda_sparsity`
- `lambda_orth`
- `lambda_normal`
- `lambda_scale_l2`
- `lambda_surface`

Additional behavior:
- LPIPS warmup schedule in `perceptual_loss`.
- Surface bind loss penalizes local `z` offset before TBN binding.
- Normal loss uses smooth vertex normals (`scatter_add_` accumulation).

</details>

<details>
<summary><strong>7. Configurations</strong></summary>

## 7. Configurations

### 7.1 `config/offline.yaml` (current high-density tuning snapshot)

- `model.tex_size`: `256` (update as needed for density target)
- `model.init_scaling`: `0.0005`
- `train.batch_size`: `4`
- `train.recon.position_lr`: `0.0004`
- `train.recon.position_lr_max_steps`: `45_000`
- `train.recon.lambda_lpips`: `0.05`
- `train.recon.lambda_scale_l2`: `0.15`
- `train.recon.lambda_surface`: `0.5`

### 7.2 `config/online.yaml`

- Online sampler and replay parameters are defined under `train.sampler`.
- Uses streaming training thread with frame ingestion.

### 7.3 `config/nersemble.yaml`

- Multi-view training config with extended iteration horizon.
- FLAME 2023 path expectation is hardcoded in script.

</details>

<details>
<summary><strong>8. Output and Artifact Layout</strong></summary>

## 8. Output and Artifact Layout

For a run:
`<output_dir>/<subject>/<work_name>/`

Typical artifacts:
- `config.yaml` (copied run config)
- `model.ply`
- `speed.txt`
- `metrics.npz` / `metrics.txt` (evaluation)
- `render_image/*.png` (or `.jpg` in NeRSemble renderer)
- `deformed_ply/*.ply` (optional, render-time export)
- `render_orbit/*.png` and `orbit_animation.gif`

</details>

<details>
<summary><strong>9. Utilities</strong></summary>

## 9. Utilities

### 9.1 Convert `.ply` to `.pt`

Use this when downstream runtime/tooling expects tensor files (`.pt`) and you want to avoid implementing a PLY parser in that path.

```bash
python tools/ply_to_pt.py --in path/to/point_cloud.ply --out point_cloud.pt --sh-degree auto
```

Useful options:
- `--frest-layout`
- `--scale-mode`
- `--opacity-mode`

</details>

<details>
<summary><strong>10. Resource Constraints and Limits</strong></summary>

## 10. Resource Constraints and Limits

- GPU memory grows with `tex_size`, batch size, and image resolution.
- `render.py` defaults to batch rendering and threaded image I/O.
- Mixed OpenMP runtimes on Windows may require:
  - `KMP_DUPLICATE_LIB_OK=TRUE`
- `nvdiffrast` GL context creation can emit deprecation warning for `RasterizeGLContext`.

</details>

<details open>
<summary><strong>11. Error Diagnostics</strong></summary>

## 11. Error Diagnostics

### 11.1 Common Runtime Failures

- `ModuleNotFoundError: yaml`
  - Install PyYAML in the same interpreter used to run scripts.
- `OMP: Error #15 ... libiomp5md.dll already initialized`
  - Set `KMP_DUPLICATE_LIB_OK=TRUE` (workaround).
- CUDA OOM
  - Reduce `train.batch_size`, reduce `tex_size`, or lower input resolution.
- Missing FLAME model file
  - Verify `data/FLAME2020/generic_model.pkl` path.

### 11.2 Config/Code Mismatch Risks

- Ensure selected training path matches expected config keys.
- Keep `train.recon` fields aligned with loss terms used by target reconstruction class.
- Confirm `subject`, `output_dir`, and `work_name` correspond to existing trained run when rendering/evaluating.

</details>

<details open>
<summary><strong>12. File Responsibilities</strong></summary>

## 12. File Responsibilities

- `train_offline.py`: single-view offline training entry.
- `train_online.py`: online training loop and frame sampler thread.
- `train_offline_nersemble.py`: multi-view NeRSemble training.
- `render.py`: sequence rendering + optional deformed `.ply` export + orbit rendering.
- `render_nersemble.py`: multi-camera validation rendering.
- `calculate_metrics.py`: L1/L2/PSNR/SSIM/LPIPS evaluation.
- `model/gaussian.py`: Gaussian parameter containers, blend projection, IO.
- `model/binding.py`: UV-face binding and mesh-aware deformation.
- `model/reconstruction.py`: main single-view reconstruction/training logic.
- `model/mv_reconstruction.py`: multi-view reconstruction logic.
- `diff_renderer/*`: rasterization wrappers.

</details>

<details open>
<summary><strong>13. Citation</strong></summary>

## 13. Citation

```bibtex
@misc{lee2026_perception_aware_gaussian_mesh,
  author       = {Jinwoo Lee},
  title        = {Perception-aware Gaussian-Mesh Hybrid Rendering for Autostereoscopic Telepresence},
  year         = {2026},
  howpublished = {GitHub repository},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  url          = {https://github.com/cinescope-wkr/Perception-aware-Gaussian-Mesh-Hybrid-Rendering-for-Autostereoscopic-Telepresence}
}

@InProceedings{Li_2025_CVPR,
    author    = {Li, Linzhou and Li, Yumeng and Weng, Yanlin and Zheng, Youyi and Zhou, Kun},
    title     = {RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {10747-10757}
}
```

</details>
