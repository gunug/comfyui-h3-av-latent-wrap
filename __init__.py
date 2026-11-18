"""Wrap a plain MiniMax H3 video latent (from VAEEncode) into the AV NestedTensor
the H3 video models expect, with a silent audio track — enables img2img / refine."""
import torch

import comfy.model_management
import comfy.nested_tensor
from comfy_extras.nodes_minimax_h3 import temporal_shape


class MiniMaxH3WrapAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "wrap"
    CATEGORY = "model/latent/minimax"
    DESCRIPTION = "VAEEncode latent -> MiniMax H3 AV latent (silent audio). Encode 5+ frames (17k+5) for the video grid."

    def wrap(self, latent):
        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            return (latent,)
        if samples.ndim == 4:
            samples = samples.unsqueeze(2)
        b, _, t = samples.shape[:3]
        if t == 1:
            audio_t = 2  # still image, as p-h3-image does
        else:
            frame_count = 5 if t <= 2 else ((t - 2) // 5) * 17 + 5
            _, _, audio_t = temporal_shape(frame_count)
        audio = torch.zeros([b, 32, 2, audio_t], device=samples.device, dtype=samples.dtype)
        out = {k: v for k, v in latent.items() if k != "noise_mask"}
        out["samples"] = comfy.nested_tensor.NestedTensor((samples, audio))
        return (out,)


class PasteCropXYXYFeather:
    """Paste a (refined) crop back at an xyxy bbox (KJ ImageCropByMaskAndResize) with a feathered edge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original_image": ("IMAGE",),
            "crop_image": ("IMAGE",),
            "bbox": ("BBOX",),
            "feather": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                                  "tooltip": "Edge falloff as a fraction of the crop's shorter side"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste"
    CATEGORY = "image"

    def paste(self, original_image, crop_image, bbox, feather):
        import torch.nn.functional as F
        out = original_image.clone()
        boxes = bbox if isinstance(bbox, (list, tuple)) and bbox and isinstance(bbox[0], (list, tuple)) else [bbox]
        for i in range(out.shape[0]):
            x0, y0, x1, y1 = [int(v) for v in boxes[min(i, len(boxes) - 1)]]
            w, h = x1 - x0, y1 - y0
            crop = crop_image[min(i, crop_image.shape[0] - 1)].movedim(-1, 0).unsqueeze(0)
            crop = F.interpolate(crop, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
            crop = crop.squeeze(0).movedim(0, -1).to(out.device, out.dtype)
            f = max(1, int(min(w, h) * feather))
            ramp_x = torch.clamp(torch.minimum(torch.arange(w), torch.arange(w).flip(0)).float() / f, 0, 1)
            ramp_y = torch.clamp(torch.minimum(torch.arange(h), torch.arange(h).flip(0)).float() / f, 0, 1)
            alpha = (ramp_y[:, None] * ramp_x[None, :]).to(out.device, out.dtype).unsqueeze(-1)
            region = out[i, y0:y1, x0:x1, :]
            out[i, y0:y1, x0:x1, :] = region * (1 - alpha) + crop[..., :region.shape[-1]] * alpha
        return (out,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3WrapAVLatent": MiniMaxH3WrapAVLatent, "PasteCropXYXYFeather": PasteCropXYXYFeather}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3WrapAVLatent": "MiniMax H3 Wrap AV Latent (img2img)",
                              "PasteCropXYXYFeather": "Paste Crop (xyxy bbox, feather)"}
