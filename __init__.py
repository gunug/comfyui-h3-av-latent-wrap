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


def _seg_bbox(seg):
    x1, y1, x2, y2 = [int(v) for v in seg.bbox]
    return x1, y1, x2, y2


def _color_match(target, ref):
    """Per-channel mean/std transfer of target [1,H,W,C] onto ref's statistics."""
    t_mean, t_std = target.mean(dim=(1, 2), keepdim=True), target.std(dim=(1, 2), keepdim=True) + 1e-5
    r_mean, r_std = ref.mean(dim=(1, 2), keepdim=True), ref.std(dim=(1, 2), keepdim=True) + 1e-5
    return ((target - t_mean) / t_std * r_std + r_mean).clamp(0, 1)


class FaceSegsSplitByReference:
    """Split detected face SEGS into the one face that matches a reference (main) and the rest (others)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "segs": ("SEGS",),
            "image": ("IMAGE",),
            "reference_face": ("IMAGE",),
            "min_face_size": ("INT", {"default": 48, "min": 0, "max": 4096,
                                      "tooltip": "Faces whose shorter bbox side is below this are dropped from 'others'"}),
        }, "optional": {
            "clip_vision": ("CLIP_VISION", {"tooltip": "Used to pick the face most similar to reference_face. Without it the largest face is main."}),
        }}

    RETURN_TYPES = ("SEGS", "SEGS", "STRING")
    RETURN_NAMES = ("main_segs", "other_segs", "report")
    FUNCTION = "split"
    CATEGORY = "ImpactPack/Util"

    def split(self, segs, image, reference_face, min_face_size, clip_vision=None):
        shape, items = segs
        items = list(items)
        if not items:
            return ((shape, []), (shape, []), "no faces detected")

        if clip_vision is not None:
            crops = []
            for seg in items:
                x1, y1, x2, y2 = _seg_bbox(seg)
                crops.append(image[0:1, y1:y2, x1:x2, :])
            ref = clip_vision.encode_image(reference_face[0:1])["image_embeds"]
            ref = torch.nn.functional.normalize(ref.float(), dim=-1)
            scores = []
            for c in crops:
                e = clip_vision.encode_image(c)["image_embeds"]
                scores.append(float((torch.nn.functional.normalize(e.float(), dim=-1) * ref).sum()))
            main_idx = max(range(len(items)), key=lambda i: scores[i])
            how = "clip similarity"
        else:
            scores = [(lambda b: (b[2] - b[0]) * (b[3] - b[1]))(_seg_bbox(s)) for s in items]
            main_idx = max(range(len(items)), key=lambda i: scores[i])
            how = "largest area"

        others, lines = [], []
        for i, seg in enumerate(items):
            x1, y1, x2, y2 = _seg_bbox(seg)
            size = min(x2 - x1, y2 - y1)
            tag = "MAIN" if i == main_idx else ("refine" if size >= min_face_size else "skip(small)")
            lines.append(f"#{i} bbox=({x1},{y1},{x2},{y2}) size={size} score={scores[i]:.3f} -> {tag}")
            if i != main_idx and size >= min_face_size:
                others.append(seg)
        report = f"main picked by {how}\n" + "\n".join(lines)
        return ((shape, [items[main_idx]]), (shape, others), report)


class H3RefineFaces:
    """Refine every face in SEGS with a MiniMax H3 img2img pass (no identity reference) and paste them back."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "segs": ("SEGS",),
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS", {"tooltip": "Low start sigma = light touch, e.g. 0.3, 0.2, 0.1, 0.0"}),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "prompt": ("STRING", {"multiline": True, "default": "Close-up portrait photo of a person, same face, same expression, sharp detailed eyes, natural skin texture, realistic photo."}),
            "resolution": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32,
                                   "tooltip": "Long side of each face crop for the refine pass"}),
            "padding": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05,
                                  "tooltip": "Context around the face as a fraction of the bbox's longer side"}),
            "feather": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01}),
            "color_match": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "refine"
    CATEGORY = "model/latent/minimax"

    def refine(self, image, segs, model, clip, vae, sampler, sigmas, noise_seed, prompt,
               resolution, padding, feather, color_match):
        import torch.nn.functional as F
        import nodes
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

        out = image[0:1].clone()
        _, H, W, _ = out.shape
        cond = None
        for i, seg in enumerate(segs[1]):
            x1, y1, x2, y2 = _seg_bbox(seg)
            side = int(max(x2 - x1, y2 - y1) * (1 + padding))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bx0, by0 = max(0, cx - side // 2), max(0, cy - side // 2)
            bx1, by1 = min(W, bx0 + side), min(H, by0 + side)
            bx0, by0 = max(0, bx1 - side), max(0, by1 - side)
            crop = out[:, by0:by1, bx0:bx1, :]
            ch, cw = crop.shape[1], crop.shape[2]
            if ch < 8 or cw < 8:
                continue
            s = resolution / max(ch, cw)
            tw, th = max(256, round(cw * s / 32) * 32), max(256, round(ch * s / 32) * 32)
            big = F.interpolate(crop.movedim(-1, 1), size=(th, tw), mode="bicubic", align_corners=False).clamp(0, 1).movedim(1, -1)

            if cond is None or cond[1] != (tw, th):
                c = MiniMaxH3ReferenceToVideo.execute(clip=clip, prompt=prompt, width=tw, height=th, length=5,
                                                      ref_image_size="match", vae=vae)[0]
                cond = (c, (tw, th))
            latent = {"samples": vae.encode(big.repeat(5, 1, 1, 1)[..., :3])}
            latent = MiniMaxH3WrapAVLatent().wrap(latent)[0]
            guider = BasicGuider.execute(model, cond[0])[0]
            noise = RandomNoise.execute(noise_seed + i)[0]
            sampled = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]
            dec = nodes.VAEDecode().decode(vae, sampled)[0]
            dec = dec.reshape(-1, *dec.shape[-3:])[0:1].to(out.device, out.dtype)
            if color_match:
                dec = _color_match(dec, big.to(dec.device, dec.dtype))
            out = PasteCropXYXYFeather().paste(out, dec, [(bx0, by0, bx1, by1)], feather)[0]
        return (out,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3WrapAVLatent": MiniMaxH3WrapAVLatent, "PasteCropXYXYFeather": PasteCropXYXYFeather,
                       "FaceSegsSplitByReference": FaceSegsSplitByReference, "H3RefineFaces": H3RefineFaces}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3WrapAVLatent": "MiniMax H3 Wrap AV Latent (img2img)",
                              "PasteCropXYXYFeather": "Paste Crop (xyxy bbox, feather)",
                              "FaceSegsSplitByReference": "Face SEGS Split by Reference (main / others)",
                              "H3RefineFaces": "MiniMax H3 Refine Faces (no reference, all SEGS)"}
