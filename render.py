import os
# Workaround for duplicate OpenMP runtime loading on some Windows setups.
# Keep this before importing torch/numpy-backed libraries.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import yaml
import torch
import numpy as np
import struct
import imageio

from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import nvdiffrast.torch as dr
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from camera import IntrinsicsCamera
from dataset import FLAMEDataset, FuHeadDataset
from diff_renderer import render_gs_batch
from model import FLAMEBindingModel, FuHeadBindingModel
from submodules.flame import FLAME, FlameConfig
from submodules.fuhead import FuHead
from utils import Struct


def parse_args(argv) -> Namespace:
    parser = ArgumentParser(description="Render sequence frames and optional deformed PLY/orbit views.")
    parser.add_argument("--subject", type=str, default="bala", help="Subject name under data_dir.")
    parser.add_argument("--output_dir", type=str, default="output", help="Root output directory.")
    parser.add_argument("--work_name", type=str, required=True, help="Experiment folder name.")
    parser.add_argument("--white_bg", action="store_true", help="Use white background instead of black.")
    parser.add_argument("--alpha", action="store_true", help="Save RGBA render output.")

    parser.add_argument("--batch_size", type=int, default=10, help="Render batch size.")
    parser.add_argument("--io_workers", type=int, default=8, help="Thread workers for image write.")

    parser.add_argument(
        "--save_ply_start",
        type=int,
        default=None,
        help="Start frame index (inclusive) for saving deformed .ply.",
    )
    parser.add_argument(
        "--save_ply_end",
        type=int,
        default=None,
        help="End frame index (inclusive) for saving deformed .ply.",
    )
    parser.add_argument(
        "--save_ply_every",
        type=int,
        default=1,
        help="Save every N frames within [start, end].",
    )
    parser.add_argument(
        "--save_ply_interval",
        type=int,
        default=86,
        help="Fallback interval when start/end are not provided.",
    )

    parser.add_argument("--disable_orbit", action="store_true", help="Skip orbit rendering.")
    parser.add_argument("--orbit_anchor_idx", type=int, default=86, help="Dataset frame index used as orbit anchor.")
    parser.add_argument("--orbit_fps", type=int, default=30, help="Orbit output FPS.")
    parser.add_argument("--orbit_duration_sec", type=float, default=4.0, help="Orbit duration in seconds.")
    parser.add_argument("--orbit_yaw_deg", type=float, default=45.0, help="Max yaw amplitude in degrees.")
    parser.add_argument("--disable_orbit_gif", action="store_true", help="Do not export orbit GIF.")
    return parser.parse_args(argv)


def save_image(image_data: np.ndarray, output_img_path: str, index: int) -> None:
    Image.fromarray(image_data).save(os.path.join(output_img_path, f"{index:05d}.png"))


def _should_save_deformed_ply(
    frame_idx: int,
    save_interval: int,
    save_ply_start: Optional[int],
    save_ply_end: Optional[int],
    save_ply_every: int,
) -> bool:
    if save_ply_start is not None or save_ply_end is not None:
        start = 0 if save_ply_start is None else int(save_ply_start)
        end = frame_idx if save_ply_end is None else int(save_ply_end)
        every = max(int(save_ply_every), 1)
        return (start <= frame_idx <= end) and ((frame_idx - start) % every == 0)
    return frame_idx == 0 or (save_interval > 0 and frame_idx % save_interval == 0)


def save_deformed_ply(gaussian, path: str) -> None:
    """
    Save one-frame deformed Gaussian as-is (no heuristic cleanup).
    """
    xyz = gaussian.xyz[0].detach().cpu().numpy()
    num_pts = xyz.shape[0]

    sh_orig = gaussian.sh[0].detach().cpu().numpy()
    opacity = gaussian.opacity[0].detach().cpu().numpy()
    scaling = np.log(gaussian.scaling[0].detach().cpu().numpy() + 1e-8)
    rotation = gaussian.rotation[0].detach().cpu().numpy()
    sh_final = sh_orig.reshape(num_pts, -1)

    if sh_final.shape[1] < 48:
        sh_final = np.concatenate([sh_final, np.zeros((num_pts, 48 - sh_final.shape[1]))], axis=1)

    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {num_pts}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property float nx\nproperty float ny\nproperty float nz\n"
    header += "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
    for i in range(45):
        header += f"property float f_rest_{i}\n"
    header += "property float opacity\nproperty float scale_0\nproperty float scale_1\n"
    header += "property float scale_2\nproperty float rot_0\nproperty float rot_1\n"
    header += "property float rot_2\nproperty float rot_3\nend_header\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(num_pts):
            data = [
                xyz[i, 0], xyz[i, 1], xyz[i, 2],
                0.0, 0.0, 0.0,
            ] + sh_final[i].tolist() + [
                opacity[i, 0], scaling[i, 0], scaling[i, 1], scaling[i, 2],
                rotation[i, 0], rotation[i, 1], rotation[i, 2], rotation[i, 3],
            ]
            f.write(struct.pack("<62f", *data))
    print(f"PLY saved: {path} ({num_pts} pts)")


def render_frames(
    gaussian_model,
    dataset,
    camera,
    bg_color: torch.Tensor,
    alpha: bool,
    output_path: str,
    batch_size: int,
    io_workers: int,
    save_interval: int,
    save_ply_start: Optional[int],
    save_ply_end: Optional[int],
    save_ply_every: int,
) -> None:
    output_img_path = os.path.join(output_path, "render_image")
    os.makedirs(output_img_path, exist_ok=True)

    ply_out_path = os.path.join(output_path, "deformed_ply")
    os.makedirs(ply_out_path, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    global_frame_idx = 0
    progress_bar = tqdm(range(len(dataset)), desc="Rendering")

    with ThreadPoolExecutor(max_workers=max(io_workers, 1)) as executor:
        for data in dataloader:
            mesh = data["mesh"].cuda()
            blend_weight = data["blend_weight"].cuda()
            bs = mesh.shape[0]
            gaussian = gaussian_model.gaussian_deform_batch(mesh, blend_weight)

            for batch_idx in range(bs):
                frame_idx = global_frame_idx + batch_idx
                single_gaussian = Struct(
                    xyz=gaussian.xyz[batch_idx:batch_idx + 1],
                    sh=gaussian.sh[batch_idx:batch_idx + 1],
                    opacity=gaussian.opacity[batch_idx:batch_idx + 1],
                    scaling=gaussian.scaling[batch_idx:batch_idx + 1],
                    rotation=gaussian.rotation[batch_idx:batch_idx + 1],
                )

                if _should_save_deformed_ply(
                    frame_idx=frame_idx,
                    save_interval=save_interval,
                    save_ply_start=save_ply_start,
                    save_ply_end=save_ply_end,
                    save_ply_every=save_ply_every,
                ):
                    ply_name = f"deformed_{frame_idx:05d}.ply"
                    save_deformed_ply(
                        single_gaussian,
                        os.path.join(ply_out_path, ply_name),
                    )

            render_pkg = render_gs_batch(camera, bg_color, gaussian)
            image = torch.cat([render_pkg["color"], render_pkg["alpha"]], dim=1) if alpha else render_pkg["color"]
            image = (image.permute(0, 2, 3, 1) * 255.0).to(dtype=torch.uint8, device="cpu").numpy()

            for j in range(bs):
                executor.submit(save_image, image[j], output_img_path, global_frame_idx)
                global_frame_idx += 1
            progress_bar.update(bs)
    progress_bar.close()


def render_orbit(
    gaussian_model,
    dataset,
    output_path: str,
    bg_color: torch.Tensor,
    anchor_idx: int,
    fps: int,
    duration_sec: float,
    yaw_deg: float,
    make_gif: bool = True,
) -> None:
    print("\n[Start] Rendering Orbit (Novel Views)...")
    orbit_dir = os.path.join(output_path, "render_orbit")
    os.makedirs(orbit_dir, exist_ok=True)

    anchor_idx = int(np.clip(anchor_idx, 0, len(dataset) - 1))
    base_R = torch.tensor(dataset.camera_extri[:3, :3], device="cuda", dtype=torch.float32)
    base_T = torch.tensor(dataset.camera_extri[:3, 3], device="cuda", dtype=torch.float32)
    K = dataset.camera_intri

    data = dataset[anchor_idx]
    mesh = data["mesh"].cuda().unsqueeze(0)
    blend_weight = data["blend_weight"].cuda().unsqueeze(0)
    with torch.no_grad():
        gaussian = gaussian_model.gaussian_deform_batch(mesh, blend_weight)

    total_frames = max(int(fps * duration_sec), 1)
    yaw_rad = np.deg2rad(yaw_deg)
    frames_for_gif = []

    for i in tqdm(range(total_frames), desc="Orbit Rendering"):
        angle = np.sin(2 * np.pi * i / total_frames) * yaw_rad
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R_y = torch.tensor([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], device="cuda", dtype=torch.float32)
        curr_R = base_R @ R_y.T

        cam = IntrinsicsCamera(
            K=K,
            R=curr_R.cpu().numpy(),
            T=base_T.cpu().numpy(),
            width=dataset.image_width,
            height=dataset.image_height,
        ).cuda()

        with torch.no_grad():
            render_pkg = render_gs_batch(cam, bg_color, gaussian)
            img_np = (render_pkg["color"].permute(0, 2, 3, 1) * 255.0).squeeze().cpu().numpy().astype(np.uint8)
            Image.fromarray(img_np).save(os.path.join(orbit_dir, f"{i:04d}.png"))
            if make_gif:
                frames_for_gif.append(img_np)

    if make_gif and len(frames_for_gif) > 0:
        gif_path = os.path.join(output_path, "orbit_animation.gif")
        print(f"Saving GIF to {gif_path}...")
        imageio.mimsave(gif_path, frames_for_gif, fps=fps)
    print(f"Done! Check {orbit_dir}")


def build_dataset_and_model(config, subject: str, glctx):
    data_path = os.path.join(config["data_dir"], subject)
    if config["template_type"] == "Flame":
        flame_model = FLAME(FlameConfig()).cuda()
        dataset = FLAMEDataset(flame_model, data_path, split="all", **config["dataset"])
        gaussian_model = FLAMEBindingModel(Struct(**config["model"]), flame_model, glctx)
    elif config["template_type"] == "FuHead":
        fuhead_model = FuHead().cuda()
        dataset = FuHeadDataset(fuhead_model, data_path, split="all", **config["dataset"])
        gaussian_model = FuHeadBindingModel(Struct(**config["model"]), fuhead_model, glctx)
    else:
        raise NotImplementedError(f"Unsupported template_type: {config['template_type']}")
    return dataset, gaussian_model


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    glctx = dr.RasterizeGLContext()
    output_path = os.path.join(args.output_dir, args.subject, args.work_name)
    with open(os.path.join(output_path, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset, gaussian_model = build_dataset_and_model(config, args.subject, glctx)
    gaussian_model.load_ply(os.path.join(output_path, "model.ply"))

    camera = IntrinsicsCamera(
        K=dataset.camera_intri,
        R=dataset.camera_extri[:3, :3],
        T=dataset.camera_extri[:3, 3],
        width=dataset.image_width,
        height=dataset.image_height,
    ).cuda()
    bg_color = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_bg else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    render_frames(
        gaussian_model=gaussian_model,
        dataset=dataset,
        camera=camera,
        bg_color=bg_color,
        alpha=args.alpha,
        output_path=output_path,
        batch_size=args.batch_size,
        io_workers=args.io_workers,
        save_interval=args.save_ply_interval,
        save_ply_start=args.save_ply_start,
        save_ply_end=args.save_ply_end,
        save_ply_every=args.save_ply_every,
    )

    if not args.disable_orbit:
        render_orbit(
            gaussian_model=gaussian_model,
            dataset=dataset,
            output_path=output_path,
            bg_color=bg_color,
            anchor_idx=args.orbit_anchor_idx,
            fps=args.orbit_fps,
            duration_sec=args.orbit_duration_sec,
            yaw_deg=args.orbit_yaw_deg,
            make_gif=not args.disable_orbit_gif,
        )


if __name__ == "__main__":
    main()
