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


class FaceCropLargestSEGS:
    """Crop the largest detected face (YOLO SEGS bbox) out of the original image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "segs": ("SEGS",),
            "padding": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05,
                                  "tooltip": "Margin around the bbox as a fraction of its longer side (InsightFace needs some context to find landmarks)"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("face", "info")
    FUNCTION = "crop"
    CATEGORY = "ImpactPack/Util"

    def crop(self, image, segs, padding):
        items = list(segs[1])
        if not items:
            return (image[0:1], "no face detected - using the whole image")
        seg = max(items, key=lambda s: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(_seg_bbox(s)))
        x1, y1, x2, y2 = _seg_bbox(seg)
        _, H, W, _ = image.shape
        p = int(max(x2 - x1, y2 - y1) * padding)
        cx0, cy0, cx1, cy1 = max(0, x1 - p), max(0, y1 - p), min(W, x2 + p), min(H, y2 + p)
        return (image[0:1, cy0:cy1, cx0:cx1, :], f"{len(items)} face(s), largest bbox=({x1},{y1},{x2},{y2}), crop=({cx0},{cy0},{cx1},{cy1})")


_ARCFACE_DST = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]]
_INSIGHTFACE_CACHE = {}


def _insightface_dir():
    import os
    import folder_paths
    cands = []
    try:
        cands += folder_paths.get_folder_paths("insightface")
    except Exception:
        pass
    cands.append(os.path.join(folder_paths.models_dir, "insightface"))
    for base in cands:
        for sub in ("models/buffalo_l", "buffalo_l", ""):
            d = os.path.join(base, sub)
            if os.path.isfile(os.path.join(d, "det_10g.onnx")) and os.path.isfile(os.path.join(d, "w600k_r50.onnx")):
                return d
    raise FileNotFoundError("buffalo_l det_10g.onnx + w600k_r50.onnx not found under models/insightface/models/buffalo_l")


def _insightface_sessions():
    if "s" not in _INSIGHTFACE_CACHE:
        import os
        import onnxruntime as ort
        d = _insightface_dir()
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        det = ort.InferenceSession(os.path.join(d, "det_10g.onnx"), providers=prov)
        rec = ort.InferenceSession(os.path.join(d, "w600k_r50.onnx"), providers=prov)
        # antelopev2 recognition model (glintr100), optional: models/insightface/models/antelopev2/glintr100.onnx
        import folder_paths
        glint = None
        bases = [os.path.dirname(d)]
        try:
            bases += [os.path.join(b, "models") for b in folder_paths.get_folder_paths("insightface")]
            bases += folder_paths.get_folder_paths("insightface")
        except Exception:
            pass
        bases.append(os.path.join(folder_paths.models_dir, "insightface", "models"))
        for g in [os.path.join(b, "antelopev2", "glintr100.onnx") for b in bases] + [os.path.join(d, "glintr100.onnx")]:
            if os.path.isfile(g):
                glint = ort.InferenceSession(g, providers=prov)
                break
        _INSIGHTFACE_CACHE["s"] = (det, rec, glint)
    return _INSIGHTFACE_CACHE["s"]


def _vision_embedding(vision, image):
    """Pooled, L2-normalized embedding from a CLIP_VISION model (CLIP or DINOv3)."""
    if getattr(vision, "model_type", "") in ("dinov2", "dinov3"):
        # core encode_image() calls .to() on DINO's None hidden-state slot, so run the encoder directly
        import comfy.model_management
        import comfy.clip_model
        comfy.model_management.load_model_gpu(vision.patcher)
        pixels = comfy.clip_model.clip_preprocess(image[0:1].to(vision.load_device), size=vision.image_size,
                                                  mean=vision.image_mean, std=vision.image_std, crop=True).float()
        seq, _, pooled, _ = vision.model(pixel_values=pixels)
        e = pooled if pooled is not None else seq[:, 0]
        return torch.nn.functional.normalize(e.float().to(comfy.model_management.intermediate_device()), dim=-1)
    out = vision.encode_image(image[0:1])
    e = out["image_embeds"]
    if e is None or e.ndim != 2:
        e = out["last_hidden_state"][:, 0]
    return torch.nn.functional.normalize(e.float(), dim=-1)


def _scrfd_landmarks(det, rgb):
    """SCRFD (det_10g) on an RGB uint8 image -> 5 landmarks of the best face, or None."""
    import numpy as np
    import cv2
    # tight crops (face filling the frame) are missed by SCRFD: add a border, then shift landmarks back
    pad = int(0.35 * max(rgb.shape[:2]))
    rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    h, w = rgb.shape[:2]
    size = 640
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(rgb, (nw, nh))
    blob = ((canvas.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
    outs = det.run(None, {det.get_inputs()[0].name: blob})
    outs = [o[0] if o.ndim == 3 else o for o in outs]
    best = None
    for i, stride in enumerate((8, 16, 32)):
        scores, bbox, kps = outs[i][:, 0], outs[i + 3] * stride, outs[i + 6] * stride
        fh, fw = size // stride, size // stride
        centers = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).reshape(-1, 2).astype(np.float32) * stride
        centers = np.repeat(centers, 2, axis=0)
        j = int(np.argmax(scores))
        if scores[j] > 0.3 and (best is None or scores[j] > best[0]):
            pts = centers[j] + kps[j].reshape(5, 2)
            best = (float(scores[j]), pts / scale - pad)
    return best


def _arcface_embedding(rgb):
    embs, note = _face_embeddings(rgb)
    return embs["buffalo"], note


def _face_embeddings(rgb):
    """Aligned 112px face -> {'buffalo': w600k_r50 emb, 'glint': glintr100 emb or None}, note."""
    import numpy as np
    import cv2
    det, rec, glint = _insightface_sessions()
    found = _scrfd_landmarks(det, rgb)
    if found is not None:
        M, _ = cv2.estimateAffinePartial2D(found[1].astype(np.float32), np.array(_ARCFACE_DST, dtype=np.float32), method=cv2.LMEDS)
        face = cv2.warpAffine(rgb, M, (112, 112), borderValue=0)
        note = f"landmarks ok (det {found[0]:.2f})"
    else:
        face = cv2.resize(rgb, (112, 112))
        note = "NO landmarks - unaligned, score unreliable"
    blob = ((face.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
    run = lambda s: (lambda e: e / (np.linalg.norm(e) + 1e-8))(s.run(None, {s.get_inputs()[0].name: blob})[0][0])
    return {"buffalo": run(rec), "glint": run(glint) if glint is not None else None}, note


class FaceSimilarityInsightFaceCLIP:
    """Two face crops -> InsightFace (ArcFace buffalo_l) cosine and CLIP Vision cosine."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "face_a": ("IMAGE",),
            "face_b": ("IMAGE",),
            "clip_vision": ("CLIP_VISION",),
        }, "optional": {
            "dino_vision": ("CLIP_VISION", {"tooltip": "DINOv3 loaded with CLIPVisionLoader (dino_v3_vit_h.safetensors)"}),
        }}

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("insightface_score", "clip_score", "report", "antelopev2_score", "dino_score")
    FUNCTION = "compare"
    CATEGORY = "image/analysis"

    def compare(self, face_a, face_b, clip_vision, dino_vision=None):
        import numpy as np
        to_u8 = lambda t: (t[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        ea, na = _face_embeddings(to_u8(face_a))
        eb, nb = _face_embeddings(to_u8(face_b))
        arc = float(np.dot(ea["buffalo"], eb["buffalo"]))
        glint = float(np.dot(ea["glint"], eb["glint"])) if ea["glint"] is not None else float("nan")
        clip = float((_vision_embedding(clip_vision, face_a) * _vision_embedding(clip_vision, face_b)).sum())
        dino = float((_vision_embedding(dino_vision, face_a) * _vision_embedding(dino_vision, face_b)).sum()) if dino_vision is not None else float("nan")
        verdict = lambda v: "same person (high)" if v >= 0.5 else ("similar / uncertain" if v >= 0.3 else "different person")
        report = (f"InsightFace buffalo_l (w600k_r50) cosine: {arc:.3f} -> {verdict(arc)}\n"
                  + (f"InsightFace antelopev2 (glintr100) cosine: {glint:.3f} -> {verdict(glint)}\n" if glint == glint
                     else "InsightFace antelopev2: glintr100.onnx not found (models/insightface/models/antelopev2)\n")
                  + f"CLIP Vision cosine: {clip:.3f} (general look: hair, light, style; not identity)\n"
                  + (f"DINOv3 cosine: {dino:.3f} (fine visual look; not identity)\n" if dino == dino else "DINOv3: not connected\n")
                  + f"face_a: {na}\nface_b: {nb}\n"
                  "Face-ID guide (rough): >=0.5 same, 0.3-0.5 uncertain, <0.3 different")
        return (arc, clip, report, glint, dino)


def _label_font(size):
    from PIL import ImageFont
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class FaceSimilarityGrid:
    """Every detected face in an image vs one reference face -> tiled grid with InsightFace + CLIP scores drawn on."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference_face": ("IMAGE",),
            "image": ("IMAGE",),
            "segs": ("SEGS",),
            "clip_vision": ("CLIP_VISION",),
            "padding": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
            "tile_size": ("INT", {"default": 320, "min": 128, "max": 1024, "step": 32}),
            "columns": ("INT", {"default": 4, "min": 1, "max": 12}),
        }, "optional": {
            "dino_vision": ("CLIP_VISION", {"tooltip": "DINOv3 loaded with CLIPVisionLoader (dino_v3_vit_h.safetensors)"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("grid", "report")
    FUNCTION = "grid"
    CATEGORY = "image/analysis"

    def grid(self, reference_face, image, segs, clip_vision, padding, tile_size, columns, dino_vision=None):
        import numpy as np
        from PIL import Image, ImageDraw
        to_u8 = lambda t: (t[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        clip_emb = lambda t: _vision_embedding(clip_vision, t)
        dino_emb = lambda t: _vision_embedding(dino_vision, t)

        ref_u8 = to_u8(reference_face)
        ref_embs, ref_note = _face_embeddings(ref_u8)
        ref_clip = clip_emb(reference_face)
        ref_dino = dino_emb(reference_face) if dino_vision is not None else None
        has_glint = ref_embs["glint"] is not None

        _, H, W, _ = image.shape
        faces = []
        for i, seg in enumerate(segs[1]):
            x1, y1, x2, y2 = _seg_bbox(seg)
            p = int(max(x2 - x1, y2 - y1) * padding)
            cx0, cy0, cx1, cy1 = max(0, x1 - p), max(0, y1 - p), min(W, x2 + p), min(H, y2 + p)
            crop = image[0:1, cy0:cy1, cx0:cx1, :]
            embs, note = _face_embeddings(to_u8(crop))
            arc = float(np.dot(ref_embs["buffalo"], embs["buffalo"]))
            glint = float(np.dot(ref_embs["glint"], embs["glint"])) if has_glint else None
            clip = float((ref_clip * clip_emb(crop)).sum())
            dino = float((ref_dino * dino_emb(crop)).sum()) if ref_dino is not None else None
            faces.append({"i": i, "crop": to_u8(crop), "arc": arc, "glint": glint, "clip": clip, "dino": dino, "note": note,
                          "bbox": (x1, y1, x2, y2), "size": min(x2 - x1, y2 - y1),
                          "id": (arc + glint) / 2 if has_glint else arc})
        # BEST = highest face-ID score (mean of buffalo_l and antelopev2 when both exist)
        best = max(range(len(faces)), key=lambda k: faces[k]["id"]) if faces else -1

        GREEN, YELLOW, RED, WHITE, GRAY = (80, 220, 120), (240, 200, 60), (230, 80, 80), (235, 235, 235), (160, 160, 160)
        arc_color = lambda v: GREEN if v >= 0.5 else (YELLOW if v >= 0.3 else RED)
        clip_color = lambda v: GREEN if v >= 0.75 else (YELLOW if v >= 0.55 else RED)
        dino_color = lambda v: GREEN if v >= 0.6 else (YELLOW if v >= 0.4 else RED)
        n_rows = 2 + int(has_glint) + int(ref_dino is not None)
        font_big, font_small = _label_font(max(12, tile_size // 11)), _label_font(max(10, tile_size // 16))
        fs = lambda f: getattr(f, "size", 12)
        label_h = fs(font_small) + n_rows * (fs(font_big) + 4) + 14
        pad = 10

        def tile(rgb, header, header_color, rows, border):
            """rows: [(label, value_text, value_color)] -> label left, value right-aligned."""
            im = Image.fromarray(rgb)
            s = (tile_size - 8) / max(im.width, im.height)  # small faces are enlarged so every tile reads the same
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
            t = Image.new("RGB", (tile_size, tile_size + label_h), (24, 24, 24))
            t.paste(im, ((tile_size - im.width) // 2, (tile_size - im.height) // 2))
            d = ImageDraw.Draw(t)
            d.rectangle([0, 0, tile_size - 1, tile_size + label_h - 1], outline=border, width=max(3, tile_size // 60))
            y = tile_size + 4
            d.text((pad, y), header, fill=header_color, font=font_small)
            y += fs(font_small) + 6
            for label, value, vcolor in rows:
                d.text((pad, y), label, fill=WHITE, font=font_big)
                d.text((tile_size - pad - d.textlength(value, font=font_big), y), value, fill=vcolor, font=font_big)
                y += fs(font_big) + 4
            return t

        tiles = [tile(ref_u8, "REFERENCE", (120, 170, 255), [], (120, 170, 255))]
        for k, f in enumerate(faces):
            ok = f["note"].startswith("landmarks ok")
            header = f"#{f['i']}   {f['size']}px" + ("   BEST" if k == best else "") + ("" if ok else "   NO landmarks")
            rows = [("Arc", f"{f['arc']:.3f}", arc_color(f["arc"]))]
            if has_glint:
                rows.append(("Glint", f"{f['glint']:.3f}", arc_color(f["glint"])))
            rows.append(("CLIP", f"{f['clip']:.3f}", clip_color(f["clip"])))
            if f["dino"] is not None:
                rows.append(("DINO", f"{f['dino']:.3f}", dino_color(f["dino"])))
            tiles.append(tile(f["crop"], header, WHITE if ok else RED, rows, arc_color(f["id"])))

        cols = max(1, min(columns, len(tiles)))
        rows = (len(tiles) + cols - 1) // cols
        th = tile_size + label_h
        canvas = Image.new("RGB", (cols * tile_size, rows * th), (12, 12, 12))
        for k, t in enumerate(tiles):
            canvas.paste(t, ((k % cols) * tile_size, (k // cols) * th))
        out = torch.from_numpy(np.asarray(canvas).astype(np.float32) / 255.0)[None]

        lines = [f"reference: {ref_note}", f"{len(faces)} face(s) detected",
                 "antelopev2 (glintr100): " + ("on" if has_glint else "not found - models/insightface/models/antelopev2/glintr100.onnx"),
                 "DINOv3: " + ("on" if ref_dino is not None else "not connected")]
        for k, f in enumerate(faces):
            lines.append(f"#{f['i']} bbox={f['bbox']} size={f['size']}px  Arc(buffalo_l)={f['arc']:.3f}"
                         + (f"  Glint(antelopev2)={f['glint']:.3f}" if has_glint else "")
                         + f"  CLIP={f['clip']:.3f}" + (f"  DINO={f['dino']:.3f}" if f["dino"] is not None else "")
                         + ("  <- BEST" if k == best else "") + ("" if f["note"].startswith("landmarks ok") else "  (no landmarks)"))
        lines.append("Arc / Glint (face ID, rough): >=0.5 same (green), 0.3-0.5 uncertain (yellow), <0.3 different (red). BEST and border = mean of both")
        lines.append("CLIP (heuristic, general look not identity): >=0.75 green, 0.55-0.75 yellow, <0.55 red")
        lines.append("DINO (heuristic, fine visual look not identity): >=0.6 green, 0.4-0.6 yellow, <0.4 red")
        return (out, "\n".join(lines))


NODE_CLASS_MAPPINGS = {"MiniMaxH3WrapAVLatent": MiniMaxH3WrapAVLatent, "PasteCropXYXYFeather": PasteCropXYXYFeather,
                       "FaceSimilarityGrid": FaceSimilarityGrid,
                       "FaceSegsSplitByReference": FaceSegsSplitByReference, "H3RefineFaces": H3RefineFaces,
                       "FaceCropLargestSEGS": FaceCropLargestSEGS,
                       "FaceSimilarityInsightFaceCLIP": FaceSimilarityInsightFaceCLIP}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3WrapAVLatent": "MiniMax H3 Wrap AV Latent (img2img)",
                              "PasteCropXYXYFeather": "Paste Crop (xyxy bbox, feather)",
                              "FaceSegsSplitByReference": "Face SEGS Split by Reference (main / others)",
                              "H3RefineFaces": "MiniMax H3 Refine Faces (no reference, all SEGS)",
                              "FaceCropLargestSEGS": "Face Crop (largest SEGS bbox)",
                              "FaceSimilarityInsightFaceCLIP": "Face Similarity (InsightFace + CLIP Vision)",
                              "FaceSimilarityGrid": "Face Similarity Grid (all faces vs reference)"}
