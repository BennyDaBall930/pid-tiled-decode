# ComfyUI core `LatentCrop` silently mangles 5-dim latents

Discovered 2026-08-04 while building this tool. Not yet filed upstream.

## Symptom

Cropping a **qwen-image / Wan-family latent** (shape `[B, C, T, H, W]`, with `T=1`
for images) with the core `LatentCrop` node produces garbage: the "crop" you get
is a horizontal band of the scene stretched to the requested size, or pure
smeared noise, depending on the requested `x`.

## Cause

`LatentCrop.crop()` in `nodes.py` indexes the latent as `[B, C, H, W]`:

```python
samples['samples'] = samples['samples'][:, :, y:y+h_lat, x:x+w_lat]
```

On a 5-dim latent, dim 2 is **T** (temporal, size 1) and dim 3 is **H** — so the
node crops T with the y range (harmless-looking because T=1 survives most
slices) and crops **H with the x range**. The requested `x` offset walks down
the image instead of across it, and the width is never cropped at all. The
node's clamp logic (`if x > samples.shape[3] - 8`) also reads dim 3 as W when
it is actually H, so large `x` values clamp to a bottom-of-image sliver.

Because the output is still a valid-looking latent dict, downstream nodes
(e.g. `PiDConditioning`, which nearest-resizes whatever it receives to the
patch grid) run without error — the failure is silent and only visible in the
decoded image.

## Repro

1. Generate any qwen-image latent (e.g. `EmptySD3LatentImage` 1344x768 through a
   qwen-family KSampler) — resulting shape `[1, 16, 1, 96, 168]`.
2. `LatentCrop` width=496 height=416 x=424 y=352.
3. Decode the crop: you get the bottom half of the full scene squeezed to
   496x416, not the (424, 352) region.

## Fix / workaround

Slice the last two dims regardless of rank (what this repo does in
`crop_latent_tensor`):

```python
def crop_latent_tensor(t, x, y, w, h):
    return t[..., y // 8:(y + h) // 8, x // 8:(x + w) // 8]
```

A core fix would be the same one-liner in `LatentCrop.crop` (plus moving the
min-size clamps to `shape[-1]`/`shape[-2]`).
