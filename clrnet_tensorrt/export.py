"""ONNX export helpers for official CLRNet models."""

import math
import types

import torch


def _repeat_to_offsets(tensor: torch.Tensor, n_offsets: int) -> torch.Tensor:
    return tensor.unsqueeze(2).repeat(1, 1, n_offsets)


def _replace_prediction_fields(
    priors: torch.Tensor,
    cls_logits: torch.Tensor,
    reg: torch.Tensor,
    prior_ys: torch.Tensor,
    img_w: int,
    img_h: int,
    n_offsets: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build prediction tensors with cat instead of in-place slice assignment.

    TensorRT/ONNX conversion is sensitive to PyTorch slice assignment because it
    exports Scatter-style ONNX ops. Building fresh tensors with torch.cat keeps
    the graph closer to Gather/Concat arithmetic that TensorRT handles.
    """
    start_y = priors[..., 2] + reg[..., 0]
    start_x = priors[..., 3] + reg[..., 1]
    theta = priors[..., 4] + reg[..., 2]
    length = reg[..., 3]

    repeated_x = _repeat_to_offsets(start_x, n_offsets)
    repeated_y = _repeat_to_offsets(start_y, n_offsets)
    repeated_theta = _repeat_to_offsets(theta, n_offsets)
    repeated_prior_ys = prior_ys.reshape(1, 1, n_offsets)

    line_xs = (
        repeated_x * (img_w - 1)
        + ((1 - repeated_prior_ys - repeated_y) * img_h)
        / torch.tan(repeated_theta * math.pi + 1e-5)
    ) / (img_w - 1)

    prediction_lines = torch.cat(
        [
            cls_logits,
            start_y.unsqueeze(-1),
            start_x.unsqueeze(-1),
            theta.unsqueeze(-1),
            length.unsqueeze(-1),
            line_xs,
        ],
        dim=-1,
    )
    predictions = torch.cat(
        [
            prediction_lines[..., :6],
            prediction_lines[..., 6:] + reg[..., 4:],
        ],
        dim=-1,
    )
    return predictions, prediction_lines


def export_safe_clr_head_forward(self, x, **kwargs):
    """Forward replacement for CLRHead that avoids Scatter-like ONNX ops."""
    batch_features = list(x[len(x) - self.refine_layers :])
    batch_features.reverse()
    batch_size = batch_features[-1].shape[0]

    priors = self.priors.repeat(batch_size, 1, 1)
    priors_on_featmap = self.priors_on_featmap.repeat(batch_size, 1, 1)
    predictions_lists = []
    prior_features_stages = []

    for stage in range(self.refine_layers):
        num_priors = priors_on_featmap.shape[1]
        prior_xs = torch.flip(priors_on_featmap, dims=[2])

        batch_prior_features = self.pool_prior_features(
            batch_features[stage],
            num_priors,
            prior_xs,
        )
        prior_features_stages.append(batch_prior_features)

        fc_features = self.roi_gather(prior_features_stages, batch_features[stage], stage)
        fc_features = fc_features.view(num_priors, batch_size, -1).reshape(
            batch_size * num_priors,
            self.fc_hidden_dim,
        )

        cls_features = fc_features.clone()
        reg_features = fc_features.clone()
        for cls_layer in self.cls_modules:
            cls_features = cls_layer(cls_features)
        for reg_layer in self.reg_modules:
            reg_features = reg_layer(reg_features)

        cls_logits = self.cls_layers(cls_features).reshape(batch_size, -1, 2)
        reg = self.reg_layers(reg_features).reshape(batch_size, -1, self.n_offsets + 4)

        predictions, prediction_lines = _replace_prediction_fields(
            priors,
            cls_logits,
            reg,
            self.prior_ys,
            self.img_w,
            self.img_h,
            self.n_offsets,
        )
        predictions_lists.append(predictions)

        if stage != self.refine_layers - 1:
            priors = prediction_lines.detach()
            priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

    return predictions_lists[-1]


def patch_model_for_onnx_export(model: torch.nn.Module) -> None:
    """Patch the model in-place with an ONNX/TensorRT-friendly head forward."""
    model.heads.forward = types.MethodType(export_safe_clr_head_forward, model.heads)
