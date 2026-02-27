import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import lpips

from diff_renderer import BatchGaussianRenderer
from camera import Camera
from utils import Struct, l1_loss, get_expon_lr_func, create_window, _ssim
from .binding import BindingModel


class Reconstruction:
    def __init__(
        self,
        camera: Camera,
        gaussian_model: BindingModel,
        batch_size: int,
        recon_config: Struct,
        tb_writer: SummaryWriter = None,
    ):
        self.batch_size = batch_size
        self.recon_config = recon_config
        self.tb_writer = tb_writer
        self.camera = camera
        self.gaussian_model = gaussian_model

        self.bg_color = torch.tensor(self.recon_config.bg_color, dtype=torch.float32, device="cuda")
        self.batch_gaussian_renderer = BatchGaussianRenderer(
            bg_color=self.bg_color,
            static_camera=camera,
            num_gaussians=gaussian_model.num_gaussian,
            batch_size=batch_size,
        )

        gs_params, bs_params, adapter_params = gaussian_model.training_params(self.recon_config)
        self.optimizer = Adam(params=gs_params + bs_params + adapter_params, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=self.recon_config.position_lr * self.recon_config.scene_extent,
            lr_final=self.recon_config.position_lr_final * self.recon_config.scene_extent,
            lr_delay_mult=self.recon_config.position_lr_delay_mult,
            max_steps=self.recon_config.position_lr_max_steps,
        )

        self.iteration = 0
        self.perceptual_model = None
        self.ssim_window = None
        self.global_lr_scale = 1.0

    def perceptual_loss(self, image: torch.Tensor, gt_image: torch.Tensor):
        # LPIPS warmup to avoid unstable gradients early in training.
        start_iter = 15000
        warmup_steps = 5000
        progress = min(max((self.iteration - start_iter) / warmup_steps, 0.0), 1.0)
        if progress <= 0.0:
            return 0.0

        if self.perceptual_model is None:
            self.perceptual_model = lpips.LPIPS(net="vgg").cuda()

        image = image * 2.0 - 1.0
        gt_image = gt_image * 2.0 - 1.0
        return self.perceptual_model(image, gt_image).mean() * progress

    def ssim_loss(self, image: torch.Tensor, gt_image: torch.Tensor):
        if self.ssim_window is None:
            self.ssim_window = create_window(11, 3).to(device=image.device, dtype=image.dtype)
        return 1.0 - _ssim(image, gt_image, self.ssim_window, 11, 3, size_average=True)

    def charbonnier_loss(self, pred, gt, eps=1e-3):
        return torch.sqrt((pred - gt) ** 2 + eps**2).mean()

    def get_shortest_axis_dynamic(self, q, s):
        """
        Compute the world-space shortest Gaussian axis from quaternion + scales.
        q: [..., 4] as (r, x, y, z)
        s: [..., 3]
        """
        r, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        r_x = torch.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y + r * z),
                2 * (x * z - r * y),
            ],
            dim=-1,
        )
        r_y = torch.stack(
            [
                2 * (x * y - r * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z + r * x),
            ],
            dim=-1,
        )
        r_z = torch.stack(
            [
                2 * (x * z + r * y),
                2 * (y * z - r * x),
                1 - 2 * (x * x + y * y),
            ],
            dim=-1,
        )
        rotation_cols = torch.stack([r_x, r_y, r_z], dim=-2)  # [..., 3, 3]

        min_idx = torch.argmin(s, dim=-1)  # [...]
        idx_expanded = min_idx.unsqueeze(-1).unsqueeze(-1).expand(*min_idx.shape, 1, 3)
        shortest_axis = torch.gather(rotation_cols, -2, idx_expanded).squeeze(-2)
        return torch.nn.functional.normalize(shortest_axis, dim=-1)

    def compute_vertex_normals(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth vertex normals by area-weighted face normal accumulation.
        vertices: [B, V, 3]
        faces: [F, 3]
        """
        bsz, _, _ = vertices.shape
        faces = faces.long()
        num_faces = faces.shape[0]

        v0 = vertices[:, faces[:, 0]]
        v1 = vertices[:, faces[:, 1]]
        v2 = vertices[:, faces[:, 2]]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # [B, F, 3]

        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            idx = faces[:, i].unsqueeze(0).unsqueeze(-1).expand(bsz, num_faces, 3)
            vertex_normals.scatter_add_(1, idx, face_normals)

        return torch.nn.functional.normalize(vertex_normals, p=2, dim=-1)

    def compute_face_normals(self, vertices, faces):
        """Compute unit face normals. Kept for debugging/analysis utilities."""
        v0 = vertices[:, faces[:, 0]]
        v1 = vertices[:, faces[:, 1]]
        v2 = vertices[:, faces[:, 2]]
        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        return torch.nn.functional.normalize(normals, dim=-1)

    def as_tensor(self, x, device="cuda"):
        if torch.is_tensor(x):
            return x
        return torch.tensor(x, device=device, dtype=torch.float32)

    def update_learning_rate(self):
        xyz_lr = self.xyz_scheduler_args(self.iteration) * self.recon_config.scene_extent
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = xyz_lr * self.global_lr_scale
            elif param_group["name"] == "opacity":
                param_group["lr"] = self.recon_config.opacity_lr * self.global_lr_scale
            elif param_group["name"] == "scaling":
                param_group["lr"] = self.recon_config.scaling_lr * self.global_lr_scale
            elif param_group["name"] == "rotation":
                param_group["lr"] = self.recon_config.rotation_lr * self.global_lr_scale
            elif param_group["name"] == "f_dc":
                param_group["lr"] = self.recon_config.feature_lr * self.global_lr_scale
            elif param_group["name"] == "xyz_b":
                param_group["lr"] = xyz_lr * self.recon_config.position_b_lr_scale * self.global_lr_scale
            elif param_group["name"] == "rotation_b":
                param_group["lr"] = (
                    self.recon_config.rotaion_b_lr_scale
                    * self.recon_config.rotation_lr
                    * self.global_lr_scale
                )
            elif param_group["name"] == "f_dc_b":
                param_group["lr"] = (
                    self.recon_config.feature_b_lr_scale
                    * self.recon_config.feature_lr
                    * self.global_lr_scale
                )
            elif param_group["name"] == "weight_module":
                param_group["lr"] = self.recon_config.weight_module_lr * self.global_lr_scale

    def step(
        self,
        gt_image: torch.Tensor,
        template_mesh: torch.Tensor,
        blend_weight: torch.Tensor,
    ):
        batch_size = gt_image.shape[0]
        self.update_learning_rate()

        gt_rgb, gt_mask = gt_image[:, :3], gt_image[:, 3:4]

        # Composite GT over background (randomized or fixed).
        if self.recon_config.random_bg_color:
            bg_color = torch.rand([batch_size, 3, 1, 1], dtype=torch.float32, device="cuda")
        else:
            bg_color = self.bg_color.reshape(1, 3, 1, 1).repeat(batch_size, 1, 1, 1)
        gt_rgb = gt_rgb * gt_mask + bg_color * (1.0 - gt_mask)

        # Blend activation schedule.
        blend_weight = None if self.iteration < self.recon_config.blend_start_iter else blend_weight

        # local_gs: local Gaussian offsets before TBN mesh binding (used for surface regularization).
        local_gs = self.gaussian_model.get_batch_attributes(template_mesh.shape[0], blend_weight)
        gaussian = self.gaussian_model.gaussian_deform_batch(template_mesh, blend_weight)

        self.optimizer.zero_grad(set_to_none=True)
        render_pkg = self.batch_gaussian_renderer.render(bg_color, gaussian, gt_image)
        image, alpha = render_pkg["color"], render_pkg["alpha"]

        # ---------------------------------------------------------------------
        # Normal-Scaling Alignment Loss (visibility weighted)
        # ---------------------------------------------------------------------
        normal_loss_val = torch.tensor(0.0, device=image.device)
        effective_lambda_normal = 0.0
        target_lambda_normal = getattr(self.recon_config, "lambda_normal", 0.0)

        if target_lambda_normal > 0.0 and blend_weight is not None:
            faces = self.gaussian_model.template_faces.long()  # [F, 3]
            vertex_normals = self.compute_vertex_normals(template_mesh, faces)  # [B, V, 3]

            binding_ids = self.gaussian_model.binding_face_id.long()  # [N]
            binding_faces = faces[binding_ids]  # [N, 3]
            binding_bary = self.gaussian_model.binding_face_bary  # [N, 3]

            target_normals = (
                vertex_normals[:, binding_faces, :] * binding_bary.unsqueeze(0).unsqueeze(-1)
            ).sum(dim=2)  # [B, N, 3]
            target_normals = torch.nn.functional.normalize(target_normals, dim=-1)

            shortest_axis = self.get_shortest_axis_dynamic(gaussian.rotation, gaussian.scaling)  # [B, N, 3]
            dot_prod = torch.sum(target_normals * shortest_axis, dim=-1)  # [B, N]
            per_g = 1.0 - torch.abs(dot_prod)  # [B, N]

            vis_w = render_pkg.get("est_weight", None)
            if vis_w is None:
                vis_w = torch.sigmoid(gaussian.opacity).squeeze(-1)  # [B, N]
            else:
                if vis_w.dim() == 3 and vis_w.shape[-1] == 1:
                    vis_w = vis_w.squeeze(-1)
                vis_w = vis_w.clamp(min=0.0)
                vis_w = torch.log1p(vis_w)
                vis_w = vis_w / (vis_w.mean(dim=1, keepdim=True) + 1e-8)
            vis_w = vis_w.detach()

            # Encourage this alignment mainly on anisotropic Gaussians.
            s = gaussian.scaling.detach()
            s_sorted, _ = torch.sort(s, dim=-1)
            anis_ratio = s_sorted[..., 1] / (s_sorted[..., 0] + 1e-12)
            anis_mask = (anis_ratio > 1.2).float()

            w = vis_w * anis_mask
            if w.sum().item() < 1e-12:
                w = vis_w

            normal_loss_val = (per_g * w).sum() / (w.sum() + 1e-8)

            # Warm up normal regularization after blend activation.
            progress = min(
                max((self.iteration - self.recon_config.blend_start_iter), 0) / 5000,
                1.0,
            )
            effective_lambda_normal = target_lambda_normal * progress

        # ---------------------------------------------------------------------
        # Scale L2 Loss (visibility weighted)
        # ---------------------------------------------------------------------
        loss_scale_l2 = 0.0
        if self.recon_config.lambda_scale_l2 > 0.0:
            vis_w = render_pkg.get("est_weight", None)
            if vis_w is None:
                vis_w = torch.sigmoid(gaussian.opacity).squeeze(-1)
            else:
                if vis_w.dim() == 3 and vis_w.shape[-1] == 1:
                    vis_w = vis_w.squeeze(-1)
                vis_w = vis_w.clamp(min=0.0)
                vis_w = torch.log1p(vis_w)
                vis_w = vis_w / (vis_w.mean(dim=1, keepdim=True) + 1e-8)
            vis_w = vis_w.detach()

            s2 = (gaussian.scaling**2).mean(dim=-1)
            loss_scale_l2 = (s2 * vis_w).sum() / (vis_w.sum() + 1e-8)

        # Surface bind regularization in local tangent space.
        loss_surface_bind = 0.0
        if getattr(self.recon_config, "lambda_surface", 0.0) > 0.0:
            loss_surface_bind = torch.mean(torch.square(local_gs.xyz[..., 2]))

        # ---------------------------------------------------------------------
        # Total loss
        # ---------------------------------------------------------------------
        charbonnier_loss_val = (
            self.charbonnier_loss(image, gt_rgb) if self.recon_config.lambda_charbonnier > 0.0 else 0.0
        )
        ssim_loss_val = self.ssim_loss(image, gt_rgb) if self.recon_config.lambda_ssim > 0.0 else 0.0
        lpips_loss_val = self.perceptual_loss(image, gt_rgb) if self.recon_config.lambda_lpips > 0.0 else 0.0
        alpha_loss_val = l1_loss(alpha, gt_mask) if self.recon_config.lambda_alpha > 0.0 else 0.0
        sparsity_loss_val = (
            self.gaussian_model.sparsity_loss(blend_weight) if self.recon_config.lambda_sparsity > 0.0 else 0.0
        )
        sparsity_loss_val = self.as_tensor(sparsity_loss_val, device=image.device)
        orth_loss_val = (
            self.gaussian_model.orth_loss()
            if self.recon_config.lambda_orth > 0.0 and blend_weight is not None
            else 0.0
        )
        loss_scale_l2 = loss_scale_l2 if self.recon_config.lambda_scale_l2 > 0.0 else 0.0

        total_loss = (
            self.recon_config.lambda_charbonnier * charbonnier_loss_val
            + self.recon_config.lambda_ssim * ssim_loss_val
            + self.recon_config.lambda_lpips * lpips_loss_val
            + self.recon_config.lambda_alpha * alpha_loss_val
            + self.recon_config.lambda_sparsity * sparsity_loss_val
            + self.recon_config.lambda_orth * orth_loss_val
            + self.recon_config.lambda_scale_l2 * loss_scale_l2
            + getattr(self.recon_config, "lambda_surface", 0.0) * loss_surface_bind
            + effective_lambda_normal * normal_loss_val
        )

        total_loss.backward()
        self.optimizer.step()

        if self.recon_config.use_fast_forward:
            self.gaussian_model.fast_forward(render_pkg["est_color"], render_pkg["est_weight"])

        if self.tb_writer is not None:
            if self.recon_config.lambda_charbonnier > 0.0:
                self.tb_writer.add_scalar("train_loss/charbonnier_loss", charbonnier_loss_val.item(), self.iteration)
            if self.recon_config.lambda_ssim > 0.0:
                self.tb_writer.add_scalar("train_loss/ssim_loss", ssim_loss_val.item(), self.iteration)
            if self.recon_config.lambda_lpips > 0.0:
                self.tb_writer.add_scalar("train_loss/lpips_loss", lpips_loss_val.item(), self.iteration)
            if self.recon_config.lambda_alpha > 0.0:
                self.tb_writer.add_scalar("train_loss/alpha_loss", alpha_loss_val.item(), self.iteration)
            if self.recon_config.lambda_orth > 0.0:
                self.tb_writer.add_scalar("train_loss/orth_loss", orth_loss_val.item(), self.iteration)
            if self.recon_config.lambda_sparsity > 0.0:
                self.tb_writer.add_scalar("train_loss/sparsity_loss", sparsity_loss_val.item(), self.iteration)
            if self.recon_config.lambda_normal > 0.0:
                self.tb_writer.add_scalar("train_loss/normal_loss", normal_loss_val.item(), self.iteration)
            if getattr(self.recon_config, "lambda_surface", 0.0) > 0.0:
                self.tb_writer.add_scalar("train_loss/surface_bind_loss", loss_surface_bind.item(), self.iteration)
            self.tb_writer.add_scalar("train_loss/total_loss", total_loss.item(), self.iteration)

        self.iteration += batch_size
