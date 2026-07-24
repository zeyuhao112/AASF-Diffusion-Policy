import torch
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.policy.diffusion_unet_lowdim_policy import (
    DiffusionUnetLowdimPolicy,
)


class SmoothDiffusionUnetLowdimPolicy(DiffusionUnetLowdimPolicy):
    """
    Diffusion Policy with trajectory finite-difference regularization.

    D1: action variation
    D2: acceleration proxy
    D3: jerk proxy
    """

    def __init__(
        self,
        *args,
        lambda_smooth=0.0,
        lambda_acc=0.0,
        lambda_jerk=0.0,
        timestep_weighted=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lambda_smooth = float(lambda_smooth)
        self.lambda_acc = float(lambda_acc)
        self.lambda_jerk = float(lambda_jerk)
        self.timestep_weighted = bool(timestep_weighted)

        # 用于后续记录各个损失分量
        self.last_loss_dict = {}

    def _finite_difference_losses(self, action_pred, sample_weight):
        """
        action_pred:
            normalized predicted clean actions, shape (B, T, Da)

        sample_weight:
            timestep-dependent weight, shape (B,)
        """
        batch_size = action_pred.shape[0]
        zero = action_pred.new_zeros(batch_size)

        # D1: a_t - a_{t-1}
        if action_pred.shape[1] >= 2:
            d1 = action_pred[:, 1:] - action_pred[:, :-1]
            d1_loss = d1.square().mean(dim=(1, 2))
        else:
            d1 = None
            d1_loss = zero

        # D2: D1_t - D1_{t-1}
        if d1 is not None and d1.shape[1] >= 2:
            d2 = d1[:, 1:] - d1[:, :-1]
            d2_loss = d2.square().mean(dim=(1, 2))
        else:
            d2 = None
            d2_loss = zero

        # D3: D2_t - D2_{t-1}
        if d2 is not None and d2.shape[1] >= 2:
            d3 = d2[:, 1:] - d2[:, :-1]
            d3_loss = d3.square().mean(dim=(1, 2))
        else:
            d3_loss = zero

        smooth_loss = (d1_loss * sample_weight).mean()
        acc_loss = (d2_loss * sample_weight).mean()
        jerk_loss = (d3_loss * sample_weight).mean()

        return smooth_loss, acc_loss, jerk_loss

    def compute_loss(self, batch):
        # =========================
        # 1. Normalize input
        # =========================
        assert "valid_mask" not in batch

        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]

        # Ensure normalized tensors are on the same device as the policy/model
        _device = next(self.parameters()).device
        obs = obs.to(_device)
        action = action.to(_device)

        # =========================
        # 2. Build conditioning
        # =========================
        local_cond = None
        global_cond = None
        trajectory = action

        if self.obs_as_local_cond:
            # clone 避免直接修改原始 normalized observation
            local_cond = obs.clone()
            local_cond[:, self.n_obs_steps:, :] = 0

        elif self.obs_as_global_cond:
            global_cond = obs[:, :self.n_obs_steps, :].reshape(
                obs.shape[0], -1
            )

            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To

                if self.oa_step_convention:
                    start = To - 1

                end = start + self.n_action_steps
                trajectory = action[:, start:end]

        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # =========================
        # 3. Inpainting mask
        # =========================
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(
                trajectory,
                dtype=torch.bool
            )
        else:
            condition_mask = self.mask_generator(trajectory.shape).to(trajectory.device)

        # =========================
        # 4. Forward diffusion
        # =========================
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory,
            noise,
            timesteps,
        )

        loss_mask = ~condition_mask

        # Apply known conditioning values
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # =========================
        # 5. Model prediction
        # =========================
        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type

        # alpha_bar_t，用于从 epsilon 恢复预测的 clean trajectory
        alpha_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=trajectory.device,
            dtype=trajectory.dtype,
        )

        alpha_bar_1d = alpha_cumprod[timesteps]

        expand_shape = (
            bsz,
        ) + (1,) * (trajectory.ndim - 1)

        alpha_bar = alpha_bar_1d.view(expand_shape)

        if pred_type == "epsilon":
            target = noise

            # x_t = sqrt(alpha_bar) * x_0
            #       + sqrt(1-alpha_bar) * epsilon
            #
            # x_0_hat = (x_t - sqrt(1-alpha_bar)*epsilon_hat)
            #           / sqrt(alpha_bar)
            pred_x0 = (
                noisy_trajectory
                - torch.sqrt(
                    torch.clamp(1.0 - alpha_bar, min=0.0)
                ) * pred
            ) / torch.sqrt(
                torch.clamp(alpha_bar, min=1e-8)
            )

        elif pred_type == "sample":
            target = trajectory
            pred_x0 = pred

        else:
            raise ValueError(
                f"Unsupported prediction type {pred_type}"
            )

        # =========================
        # 6. Original diffusion loss
        # =========================
        diffusion_loss = F.mse_loss(
            pred,
            target,
            reduction="none",
        )

        diffusion_loss = (
            diffusion_loss
            * loss_mask.type(diffusion_loss.dtype)
        )

        diffusion_loss = reduce(
            diffusion_loss,
            "b ... -> b (...)",
            "mean",
        )

        diffusion_loss = diffusion_loss.mean()

        # =========================
        # 7. Smoothness regularization
        # =========================

        # 只对 action 维度计算平滑损失，
        # 不能把 observation 维度包含进来
        pred_action = pred_x0[..., :self.action_dim]

        if self.timestep_weighted:
            # 高噪声 timestep 下 x0 恢复误差更大，
            # 因此使用 alpha_bar 降低其正则化权重
            sample_weight = alpha_bar_1d.detach()
        else:
            sample_weight = torch.ones_like(alpha_bar_1d)

        smooth_loss, acc_loss, jerk_loss = (
            self._finite_difference_losses(
                pred_action,
                sample_weight,
            )
        )

        # =========================
        # 8. Total loss
        # =========================
        total_loss = (
            diffusion_loss
            + self.lambda_smooth * smooth_loss
            + self.lambda_acc * acc_loss
            + self.lambda_jerk * jerk_loss
        )

        # 后续修改 Workspace 时用于日志记录
        self.last_loss_dict = {
            "diffusion_loss": diffusion_loss.detach(),
            "smooth_loss": smooth_loss.detach(),
            "acc_loss": acc_loss.detach(),
            "jerk_loss": jerk_loss.detach(),
            "total_loss": total_loss.detach(),
        }

        return total_loss
