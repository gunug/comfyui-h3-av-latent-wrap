# comfyui-h3-av-latent-wrap

Small ComfyUI nodes for **MiniMax H3** face refine and other img2img work, plus two example workflows.

## Nodes

### MiniMax H3 Wrap AV Latent (img2img)
`MiniMaxH3WrapAVLatent` · category `model/latent/minimax`

The H3 video DiT expects a joint video+audio latent (`NestedTensor((video, audio))`).
A plain `VAEEncode` latent fails in the sampler with `index 1 is out of bounds for dimension 0 with size 1`.
This node wraps the encoded video latent with a silent audio track sized to the H3 frame grid.

- Encode **5 frames** (or any `17k+5`) for the video grid. For a still image, `RepeatImageBatch` ×5 → `VAEEncode` → this node.
- A 1-latent-frame input gets the 2-step still-image audio track that p-h3-image uses.
- An input that is already an AV latent passes through unchanged.

### Paste Crop (xyxy bbox, feather)
`PasteCropXYXYFeather` · category `image`

This node pastes a processed crop back onto the original image with a feathered edge.
It takes the `BBOX` from KJNodes `ImageCropByMaskAndResize`, which uses the `(x0, y0, x1, y1)` format.
Easy-Use `imageUncropFromBBOX` reads `(x, y, w, h)`, so with this bbox it pastes at the wrong place and size.

- `feather`: edge falloff as a fraction of the crop's shorter side (default 0.15).

### Face SEGS Split by Reference (main / others)
`FaceSegsSplitByReference` · category `ImpactPack/Util`

Splits detected face SEGS (Impact Pack) into one **main** face and the **others**.
With a `CLIP_VISION` input (for example `clip_vision_h.safetensors`), the main face is the one most similar to `reference_face`.
Without it, the largest face is the main face. This stops a bigger face of another person from being refined into the reference identity.

- `min_face_size`: faces whose shorter bbox side is below this are skipped (not refined). Very small, blurred or turned faces are unreliable.
- `report`: one line per face (bbox, size, score, MAIN / refine / skip).

### MiniMax H3 Refine Faces (no reference, all SEGS)
`H3RefineFaces` · category `model/latent/minimax`

Runs a light MiniMax H3 img2img pass on every face in SEGS without an identity reference, then pastes each one back with a feathered edge.
Use it on `other_segs` so the other people keep their own faces and only get sharper.

- `sigmas`: a low start sigma keeps the face, for example `0.45, 0.3, 0.15, 0.0`.
- `resolution` (long side of each crop), `padding`, `feather`, `color_match`.

### Face Crop (largest SEGS bbox)
`FaceCropLargestSEGS` · category `ImpactPack/Util`

Crops the largest detected face (Impact Pack SEGS bbox) out of the original image, with a margin (`padding`, fraction of the bbox's longer side).

### Face Similarity (InsightFace + CLIP Vision)
`FaceSimilarityInsightFaceCLIP` · category `image/analysis`

Two face crops → four cosine scores and a text report.

- **Arc**: InsightFace `buffalo_l` (`w600k_r50.onnx`). Face identity.
- **Glint**: InsightFace `antelopev2` (`glintr100.onnx`), optional. A second face identity score.
- **CLIP**: `CLIP_VISION` (for example `clip_vision_h.safetensors`). General look: hair, light, style. Not identity.
- **DINO**: optional second `CLIP_VISION` input with DINOv3 (`dino_v3_vit_h.safetensors`). Fine visual look. Not identity.

Faces are aligned with the 5 landmarks of `det_10g.onnx` before Arc / Glint. Tight crops get a border first, because SCRFD misses faces that fill the frame.
The ONNX models run with `onnxruntime`. The `insightface` Python package is not needed.

### Face Similarity Grid (all faces vs reference)
`FaceSimilarityGrid` · category `image/analysis`

Compares every face in SEGS with one reference face. It returns a tiled image with the scores drawn on each tile, and a text report (bbox, size, scores).

- Colors (rough, not official): Arc / Glint ≥ 0.5 green, 0.3–0.5 yellow, < 0.3 red. CLIP ≥ 0.75 / 0.55. DINO ≥ 0.6 / 0.4.
- `BEST` and the tile border use the mean of Arc and Glint.
- `NO landmarks` (red header) means alignment failed and the score is unreliable.
- `padding`, `tile_size`, `columns`.

Face similarity models (folder under `ComfyUI/models`):

| File | Folder |
| --- | --- |
| [face_yolov8m.pt](https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt) | `ultralytics/bbox` |
| [det_10g.onnx](https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx), [w600k_r50.onnx](https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx) | `insightface/models/buffalo_l` |
| [glintr100.onnx](https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/glintr100.onnx) (optional) | `insightface/models/antelopev2` |
| [clip_vision_h.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors) | `clip_vision` |
| [dino_v3_vit_h.safetensors](https://huggingface.co/VAST-AI/TripoSplat/resolve/main/clip_vision/dino_v3_vit_h.safetensors) (optional) | `clip_vision` |

The InsightFace pretrained models are released for **non-commercial research only** ([InsightFace license](https://github.com/deepinsight/insightface#license)).

## Example workflow

`example_workflows/pika_h3_ref2va_face_refine.json` is a MiniMax H3 ref2va still image with a face refine pass.

1. References: `<Picture 1>` is the character sheet and `<Picture 2>` is the face.
2. `MiniMaxH3ReferenceToVideo` generates 5 frames, and frame 0 is saved as the image.
3. Face detect: Impact `face_yolov8m`, then the KJ `ImageCropByMaskAndResize` crop (1024, padding 96).
4. Refine: a second `MiniMaxH3ReferenceToVideo` call (face reference) → `RepeatImageBatch` ×5 → `VAEEncode` → **Wrap AV Latent**. Then `SamplerCustomAdvanced` runs with sigmas `0.45, 0.333, 0.2, 0` (about 0.45 denoise).
5. Composite: KJ `ColorMatch` → **Paste Crop (xyxy)** → save.

`example_workflows/pika_h3_ref2va_multi_face_refine.json` is the multi-face version (recommended for photos with several people).

- **Face SEGS Split by Reference** picks the main face with CLIP Vision. It is refined with the face reference (sigmas start 0.6).
- **Refine Faces** lightly refines the other faces without a reference (sigmas start 0.45).
- Faces smaller than 48 px are skipped.
- Extra model: `clip_vision_h.safetensors` in `models/clip_vision`.

`example_workflows/face_similarity_insightface_clip_dino.json` measures how much a generated image looks like a reference face.
Image A is the reference face and image B is the image to check. It gives a 1:1 report (largest faces) and a grid of every face in B. Needs ComfyUI-Impact-Pack and ComfyUI-Impact-Subpack.

Required (face refine): ComfyUI ≥ 0.45, ComfyUI-KJNodes ≥ 1.5.4, ComfyUI-Impact-Pack, and PlagueKind-Nodes (AdaLN LoRA Fix).
The models are from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3): the `ref2va` pruned UNET, the Qwen3-VL text encoder, and the sol VAE.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/gunug/comfyui-h3-av-latent-wrap
```
Restart ComfyUI after cloning. The face refine nodes need no extra Python packages.
The face similarity nodes need `onnxruntime` and `opencv-python` (`pip install onnxruntime opencv-python` in the ComfyUI Python).

## License
MIT
