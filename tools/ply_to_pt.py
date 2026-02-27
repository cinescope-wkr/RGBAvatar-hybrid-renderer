import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
from plyfile import PlyData


@dataclass(frozen=True)
class PlyToPtOptions:
    sh_degree: str
    frest_layout: str
    opacity_mode: str
    scale_mode: str


def _sorted_prop_names(props, prefix: str) -> list[str]:
    names = [p.name for p in props if p.name.startswith(prefix)]
    if not names:
        return []
    return sorted(names, key=lambda n: int(n.split("_")[-1]))


def _require_fields(el, names: list[str]) -> None:
    available = {p.name for p in el.properties}
    missing = [n for n in names if n not in available]
    if missing:
        raise KeyError(f"Missing PLY properties: {missing}. Available includes: {sorted(list(available))[:30]}...")


def _as_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_quat_wxyz(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # expected order: (r, x, y, z)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.maximum(norm, eps)


def _infer_degree_from_rest(rest_count: int) -> int | None:
    if rest_count == 0:
        return 0
    if rest_count % 3 != 0:
        return None
    sh_size = 1 + (rest_count // 3)
    root = int(round(math.sqrt(sh_size)))
    if root * root != sh_size:
        return None
    return root - 1


def _build_shs(
    dc: np.ndarray,  # (P, 3)
    rest: np.ndarray,  # (P, R) where R is multiple of 3
    frest_layout: str,
) -> np.ndarray:
    if rest.size == 0:
        return dc[:, None, :]  # (P, 1, 3)

    if rest.shape[1] % 3 != 0:
        raise ValueError(f"f_rest_* count must be multiple of 3, got {rest.shape[1]}")

    if frest_layout == "coeff_major":
        # f_rest_0.. correspond to [coeff1.R, coeff1.G, coeff1.B, coeff2.R, ...]
        rest_coeff = rest.reshape(rest.shape[0], -1, 3)  # (P, sh_size-1, 3)
    elif frest_layout == "channel_major":
        # f_rest_0.. correspond to [R(coeff1..), G(coeff1..), B(coeff1..)]
        rest_coeff = rest.reshape(rest.shape[0], 3, -1).transpose(0, 2, 1)  # (P, sh_size-1, 3)
    else:
        raise ValueError(f"Unknown frest_layout={frest_layout}")

    return np.concatenate([dc[:, None, :], rest_coeff], axis=1)  # (P, sh_size, 3)


def _pad_or_trim_shs(shs: np.ndarray, target_degree: int) -> np.ndarray:
    target_size = (target_degree + 1) ** 2
    curr_size = shs.shape[1]
    if curr_size == target_size:
        return shs
    if curr_size > target_size:
        return shs[:, :target_size, :]
    pad = np.zeros((shs.shape[0], target_size - curr_size, 3), dtype=np.float32)
    return np.concatenate([shs, pad], axis=1)


def ply_to_tensor_dict(ply_path: str, opts: PlyToPtOptions) -> dict[str, torch.Tensor]:
    ply = PlyData.read(ply_path)
    el = next((e for e in ply.elements if e.name == "vertex"), ply.elements[0])

    _require_fields(el, ["x", "y", "z"])
    xyz = np.stack([_as_float32(el["x"]), _as_float32(el["y"]), _as_float32(el["z"])], axis=1)
    num_points = xyz.shape[0]

    # Opacity
    opacity_name = None
    for candidate in ("opacity", "alpha"):
        if any(p.name == candidate for p in el.properties):
            opacity_name = candidate
            break
    if opacity_name is None:
        raise KeyError("Missing opacity field (expected 'opacity' or 'alpha').")
    opacities = _as_float32(el[opacity_name]).reshape(num_points, 1)
    if opts.opacity_mode == "sigmoid":
        opacities = _sigmoid(opacities)
    elif opts.opacity_mode == "auto":
        if (opacities.min() < -0.01) or (opacities.max() > 1.01):
            opacities = _sigmoid(opacities)
    elif opts.opacity_mode == "raw":
        pass
    else:
        raise ValueError(f"Unknown opacity_mode={opts.opacity_mode}")

    # Scales
    scale_names = _sorted_prop_names(el.properties, "scale_")
    if len(scale_names) < 3:
        raise KeyError(f"Missing scale_* fields (need at least 3). Found: {scale_names}")
    scales = np.stack([_as_float32(el[n]) for n in scale_names[:3]], axis=1)
    if opts.scale_mode == "exp":
        scales = np.exp(scales)
    elif opts.scale_mode == "auto":
        if scales.min() <= 0.0:
            scales = np.exp(scales)
    elif opts.scale_mode == "raw":
        pass
    else:
        raise ValueError(f"Unknown scale_mode={opts.scale_mode}")

    # Rotations
    rot_names = [f"rot_{i}" for i in range(4)]
    alt_rot_names = ["qw", "qx", "qy", "qz"]
    if all(any(p.name == n for p in el.properties) for n in rot_names):
        rots = np.stack([_as_float32(el[n]) for n in rot_names], axis=1)
    elif all(any(p.name == n for p in el.properties) for n in alt_rot_names):
        rots = np.stack([_as_float32(el[n]) for n in alt_rot_names], axis=1)
    else:
        raise KeyError("Missing rotation fields (expected rot_0..rot_3 or qw/qx/qy/qz).")
    rots = _normalize_quat_wxyz(rots)

    # SHs
    _require_fields(el, ["f_dc_0", "f_dc_1", "f_dc_2"])
    dc = np.stack([_as_float32(el["f_dc_0"]), _as_float32(el["f_dc_1"]), _as_float32(el["f_dc_2"])], axis=1)
    rest_names = _sorted_prop_names(el.properties, "f_rest_")
    rest = np.zeros((num_points, 0), dtype=np.float32)
    if rest_names:
        rest = np.stack([_as_float32(el[n]) for n in rest_names], axis=1)
    shs = _build_shs(dc, rest, opts.frest_layout)

    inferred_degree = _infer_degree_from_rest(rest.shape[1])
    if opts.sh_degree == "auto":
        if inferred_degree is None:
            raise ValueError(
                f"Cannot infer SH degree from f_rest_* count={rest.shape[1]} (must be 3*(square-1)). "
                "Pass --sh-degree explicitly."
            )
        shs = _pad_or_trim_shs(shs, inferred_degree)
    else:
        target_degree = int(opts.sh_degree)
        shs = _pad_or_trim_shs(shs, target_degree)

    return {
        "means3D": torch.from_numpy(xyz).to(dtype=torch.float32),
        "opacities": torch.from_numpy(opacities).to(dtype=torch.float32),
        "scales": torch.from_numpy(scales).to(dtype=torch.float32),
        "rotations": torch.from_numpy(rots).to(dtype=torch.float32),
        "shs": torch.from_numpy(shs).to(dtype=torch.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 3DGS-style .ply to a torch .pt tensor dict.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input .ply path")
    parser.add_argument("--out", dest="out_path", required=True, help="Output .pt path")
    parser.add_argument(
        "--sh-degree",
        default="auto",
        help="SH degree to write: 0/1/2/3 or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--frest-layout",
        default="coeff_major",
        choices=["coeff_major", "channel_major"],
        help="How f_rest_* is laid out in the PLY (default: coeff_major; matches most 3DGS exports).",
    )
    parser.add_argument(
        "--opacity-mode",
        default="auto",
        choices=["auto", "sigmoid", "raw"],
        help="How to interpret opacity values (default: auto).",
    )
    parser.add_argument(
        "--scale-mode",
        default="auto",
        choices=["auto", "exp", "raw"],
        help="How to interpret scale_* values (default: auto).",
    )
    args = parser.parse_args()

    opts = PlyToPtOptions(
        sh_degree=args.sh_degree,
        frest_layout=args.frest_layout,
        opacity_mode=args.opacity_mode,
        scale_mode=args.scale_mode,
    )
    tensor_dict = ply_to_tensor_dict(args.in_path, opts)
    torch.save(tensor_dict, args.out_path)

    sh_size = tensor_dict["shs"].shape[1]
    sh_degree = int(round(math.sqrt(sh_size))) - 1
    print(
        f"Saved {args.out_path}\n"
        f"- means3D:   {tuple(tensor_dict['means3D'].shape)}\n"
        f"- opacities: {tuple(tensor_dict['opacities'].shape)}\n"
        f"- scales:    {tuple(tensor_dict['scales'].shape)}\n"
        f"- rotations: {tuple(tensor_dict['rotations'].shape)}\n"
        f"- shs:       {tuple(tensor_dict['shs'].shape)} (degree={sh_degree})"
    )


if __name__ == "__main__":
    main()
