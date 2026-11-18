# comfyui-h3-av-latent-wrap

Two small ComfyUI nodes for **MiniMax H3** face refine and other img2img work, plus an example workflow.

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

## Example workflow

`example_workflows/pika_h3_ref2va_face_refine.json` is a MiniMax H3 ref2va still image with a face refine pass.

1. References: `<Picture 1>` is the character sheet and `<Picture 2>` is the face.
2. `MiniMaxH3ReferenceToVideo` generates 5 frames, and frame 0 is saved as the image.
3. Face detect: Impact `face_yolov8m`, then the KJ `ImageCropByMaskAndResize` crop (1024, padding 96).
4. Refine: a second `MiniMaxH3ReferenceToVideo` call (face reference) → `RepeatImageBatch` ×5 → `VAEEncode` → **Wrap AV Latent**. Then `SamplerCustomAdvanced` runs with sigmas `0.45, 0.333, 0.2, 0` (about 0.45 denoise).
5. Composite: KJ `ColorMatch` → **Paste Crop (xyxy)** → save.

Required: ComfyUI ≥ 0.45, ComfyUI-KJNodes ≥ 1.5.4, ComfyUI-Impact-Pack, and PlagueKind-Nodes (AdaLN LoRA Fix).
The models are from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3): the `ref2va` pruned UNET, the Qwen3-VL text encoder, and the sol VAE.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/gunug/comfyui-h3-av-latent-wrap
```
Restart ComfyUI after cloning. There are no extra Python dependencies.

## License
MIT
