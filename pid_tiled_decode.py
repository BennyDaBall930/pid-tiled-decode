#!/usr/bin/env python3
"""Latent-native tiled decoding for NVIDIA PiD (Pixel Diffusion Decoder).

Single-shot PiD decoding breaks down when the 4x output grows past the trained
resolution envelope (1024px -> 4096px per the PiD tech report, arXiv:2605.23902):
past roughly 4K output, color collapses (desaturated midtones, white -> pink,
chroma noise). This tool instead slices the ORIGINAL generation latent into
overlapping tiles and decodes each tile as a normal-sized PiD job — every tile
stays inside the trained envelope, and no VAE re-encode round-trip is involved
(unlike image-space tiled upscalers). RoPE is relative, so tiles need no
position hacks; the sigma-gated LQ injection keeps tiles globally coherent
because each is anchored to its crop of the one shared latent. Tiles are then
feather-blended (complementary raised-cosine, weights sum to 1) in pixel space.

Drives a stock ComfyUI server over HTTP — no custom nodes required.

Pipeline (each phase is its own ComfyUI submission):
  A. optional stage-1 render -> SaveLatent + VAE-decode baseline (skip with --from-latent)
  B. slice the saved latent locally into per-tile .latent files (layout-agnostic —
     core LatentCrop mangles 5-dim qwen-family latents, see README), then per tile:
     LoadLatent -> PiDConditioning -> 4-step lcm decode
  C. optional --full-frame single-shot PiD decode (A/B reference)
Then locally: feather blend, numeric QA (overlap PSNR, seam gradient ratio,
shadow green-bias and midtone chroma delta vs the VAE twin), and a review folder
with 100%-crop side-by-side strips, tile contact sheet, seam map, review.html.

Run with a python that has numpy + Pillow (the ComfyUI venv works):
  python pid_tiled_decode.py --self-test
  python pid_tiled_decode.py --from-latent "latents/my_shot_00001_.latent" \
      --label my_shot_8k --out runs/my_shot_8k --full-frame
  python pid_tiled_decode.py --prompt "..." --unet-name <model>.safetensors \
      --clip-name <te>.safetensors --label demo --out runs/demo --width 1920 --height 1088

Start your ComfyUI server first; this tool never launches one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import os

__version__ = "1.0.0"

DEFAULT_COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
DEFAULT_COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", str(Path.home() / "ComfyUI")))

STAGE1_VAE = "qwen_image_vae.safetensors"

# PiD decode constants (the standard ComfyUI PiD release checkpoints).
PID_UNET = "pid_1.5_qwenimage_1024_to_4096_4step_bf16.safetensors"
PID_CLIP = "gemma_2_2b_it_elm_bf16.safetensors"
PID_SIGMAS = "0.999,0.866,0.634,0.342,0"
PID_SCALE = 4  # stage-1 px -> output px
ALIGN = 8      # stage-1 tile positions/sizes stay multiples of 8 (latent + patch alignment)

GREEN_BIAS_GATE_PTS = 3.0   # |PiD - VAE| shadow green-bias gate (0-255 scale)
SEAM_RATIO_FLAG = 1.15      # seam gradient ratio above this gets flagged
CHROMA_GATE_PTS = 20.0      # midtone chroma delta vs VAE twin (0-255 scale); healthy
                            # decodes calibrate at 10-15, past-envelope collapse at ~29


# ---------------------------------------------------------------------------
# ComfyUI client
# ---------------------------------------------------------------------------

def get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    request = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def require_server(base_url: str) -> None:
    try:
        get_json(f"{base_url}/queue", timeout=5)
    except Exception as exc:
        raise SystemExit(
            f"ComfyUI is not reachable at {base_url} ({exc!r}).\n"
            "Start your ComfyUI server (this tool never auto-launches one), then re-run."
        )


def require_idle(base_url: str) -> None:
    queue = get_json(f"{base_url}/queue", timeout=20)
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise SystemExit("ComfyUI queue is busy — finish/cancel the running job first:\n"
                         + json.dumps({k: len(queue.get(k) or []) for k in
                                       ("queue_running", "queue_pending")}))


def submit(base_url: str, graph: dict[str, Any]) -> str:
    result = post_json(f"{base_url}/prompt", {"prompt": graph}, timeout=120)
    if "prompt_id" not in result:
        raise RuntimeError("Submit failed: " + json.dumps(result, indent=2)[:4000])
    return result["prompt_id"]


def wait_for_history(base_url: str, prompt_id: str, timeout_s: int, poll_s: float,
                     tag: str) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        time.sleep(poll_s)
        queue = get_json(f"{base_url}/queue", timeout=60)
        history = get_json(f"{base_url}/history/{prompt_id}", timeout=60)
        entry = history.get(prompt_id)
        running = len(queue.get("queue_running") or [])
        pending = len(queue.get("queue_pending") or [])
        if entry:
            status = entry.get("status", {})
            line = f"{status.get('status_str')} completed={status.get('completed')}"
            if line != last:
                print(f"[{tag}] {line} running={running} pending={pending}", flush=True)
                last = line
            if status.get("completed"):
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(f"[{tag}] ComfyUI error:\n" + json.dumps(entry, indent=2)[:6000])
        else:
            line = f"queued running={running} pending={pending}"
            if line != last:
                print(f"[{tag}] {line}", flush=True)
                last = line
    raise TimeoutError(f"[{tag}] timed out waiting for {prompt_id}")


def collect_files(comfy_root: Path, entry: dict[str, Any], node_id: str,
                  kind: str = "images") -> list[Path]:
    found: list[Path] = []
    node_output = (entry.get("outputs") or {}).get(node_id, {})
    for item in node_output.get(kind) or []:
        filename = item.get("filename")
        if not filename:
            continue
        subfolder = str(item.get("subfolder") or "").replace("\\", "/").strip("/")
        source = comfy_root / "output" / subfolder / filename
        if source.exists():
            found.append(source)
    return found


# ---------------------------------------------------------------------------
# Latent file inspection (safetensors header only, no deps)
# ---------------------------------------------------------------------------

def latent_file_dims(path: Path) -> tuple[int, int]:
    """Return stage-1 (width, height) in px from a saved .latent safetensors file."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    shape = header["latent_tensor"]["shape"]
    if len(shape) == 5:  # [B, C, T, h, w]
        h, w = shape[3], shape[4]
    else:                # [B, C, h, w]
        h, w = shape[2], shape[3]
    return w * 8, h * 8


def resolve_latent_ref(spec: str, comfy_root: Path) -> tuple[str, Path]:
    """Map --from-latent to (LoadLatent annotated name, local file path)."""
    spec = spec.strip()
    if spec.endswith(" [output]"):
        rel = spec[: -len(" [output]")]
        return spec, comfy_root / "output" / rel
    p = Path(spec)
    if p.is_absolute():
        try:
            rel = p.relative_to(comfy_root / "output")
        except ValueError:
            raise SystemExit(f"--from-latent absolute path must live under {comfy_root / 'output'}")
        return f"{rel.as_posix()} [output]", p
    return f"{Path(spec).as_posix()} [output]", comfy_root / "output" / spec


def crop_latent_tensor(t, x: int, y: int, w: int, h: int):
    """Crop the SPATIAL dims of a latent tensor, layout-agnostic.

    Qwen/Wan-family latents are 5-dim [B, C, T, H, W]; flux/sd3 are 4-dim
    [B, C, H, W]. Core LatentCrop indexes dims 2/3 and silently mangles the
    5-dim case (crops T/H instead of H/W) — so we always slice the LAST two
    dims ourselves. Coords are stage-1 px, /8 for latent units.
    """
    return t[..., y // 8:(y + h) // 8, x // 8:(x + w) // 8]


def write_tile_latents(latent_file: Path, plan: TilePlan, comfy_root: Path,
                       label: str) -> dict[str, str]:
    """Slice the full stage-1 latent into per-tile .latent files in ComfyUI's
    input dir. Returns {tile.name: LoadLatent ref}. Requires torch+safetensors
    (present in the ComfyUI venv this tool runs under)."""
    import torch  # noqa: F401 (safetensors.torch needs it)
    import safetensors.torch as st
    tensors = st.load_file(latent_file, device="cpu")
    full = tensors["latent_tensor"]
    lh, lw = full.shape[-2], full.shape[-1]
    if (lw * 8, lh * 8) != (plan.image_w, plan.image_h):
        raise RuntimeError(f"latent spatial dims {lw * 8}x{lh * 8} != plan "
                           f"{plan.image_w}x{plan.image_h}")
    crops_dir = comfy_root / "input" / "pidtile_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    refs: dict[str, str] = {}
    marker = torch.tensor([])  # keep LoadLatent's multiplier at 1.0
    for tile in plan.tiles:
        crop = crop_latent_tensor(full, tile.x, tile.y, tile.w, tile.h).contiguous()
        dest = crops_dir / f"{label}_{tile.name}.latent"
        st.save_file({"latent_tensor": crop, "latent_format_version_0": marker}, dest)
        refs[tile.name] = f"pidtile_crops/{dest.name}"
    print(f"[crops] wrote {len(refs)} tile latents (full shape {list(full.shape)}) "
          f"-> {crops_dir}")
    return refs


# ---------------------------------------------------------------------------
# Tile planner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    x: int      # stage-1 px
    y: int
    w: int
    h: int
    seed: int

    @property
    def name(self) -> str:
        return f"r{self.row}c{self.col}"


@dataclass
class TilePlan:
    image_w: int
    image_h: int
    overlap: int
    xs: list[tuple[int, int]] = field(default_factory=list)  # (pos, size) per column
    ys: list[tuple[int, int]] = field(default_factory=list)
    tiles: list[Tile] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return len(self.ys)

    @property
    def cols(self) -> int:
        return len(self.xs)

    def seam_centers_x(self) -> list[int]:
        """Vertical seam centers (stage-1 px): middle of each column overlap band."""
        out = []
        for (x0, w0), (x1, _w1) in zip(self.xs, self.xs[1:]):
            out.append((x1 + (x0 + w0)) // 2)
        return out

    def seam_centers_y(self) -> list[int]:
        out = []
        for (y0, h0), (y1, _h1) in zip(self.ys, self.ys[1:]):
            out.append((y1 + (y0 + h0)) // 2)
        return out


def _ceil_to(v: float, align: int) -> int:
    return int(math.ceil(v / align)) * align


def plan_axis(size: int, max_tile: int, overlap: int, align: int = ALIGN) -> list[tuple[int, int]]:
    """Cover [0, size) with uniform tiles of size <= max_tile, pairwise overlap >= ~overlap.

    All positions and sizes are multiples of `align`. size/max_tile/overlap must be
    multiples of align already.
    """
    for name, v in (("size", size), ("max_tile", max_tile), ("overlap", overlap)):
        if v % align:
            raise ValueError(f"{name}={v} must be a multiple of {align}")
    if max_tile <= overlap:
        raise ValueError(f"max_tile ({max_tile}) must exceed overlap ({overlap})")
    if size <= max_tile:
        return [(0, size)]
    n = max(2, math.ceil((size - overlap) / (max_tile - overlap)))
    while True:
        tile = _ceil_to((size + (n - 1) * overlap) / n, align)
        if tile <= max_tile:
            break
        n += 1
    stride = (size - tile) / (n - 1)
    positions = []
    for i in range(n):
        x = int(round(i * stride / align)) * align
        x = min(x, size - tile)
        positions.append((x, tile))
    # invariants
    assert positions[0][0] == 0 and positions[-1][0] + tile == size
    for (a, w), (b, _t) in zip(positions, positions[1:]):
        got = a + w - b
        assert got >= max(align * 2, overlap - align), f"overlap collapsed: {got} at {b}"
    return positions


def make_plan(width: int, height: int, max_tile: int, overlap: int, base_seed: int) -> TilePlan:
    plan = TilePlan(width, height, overlap)
    plan.xs = plan_axis(width, max_tile, overlap)
    plan.ys = plan_axis(height, max_tile, overlap)
    idx = 0
    for r, (y, h) in enumerate(plan.ys):
        for c, (x, w) in enumerate(plan.xs):
            plan.tiles.append(Tile(r, c, x, y, w, h, base_seed + 101 * (idx + 1)))
            idx += 1
    return plan


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def parse_loras(spec: str | None) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        head, sep, tail = part.rpartition(":")
        if sep:
            try:
                out.append((head, float(tail)))
                continue
            except ValueError:
                pass
        out.append((part, 1.0))
    return out


def stage1_graph(args: argparse.Namespace, loras: list[tuple[str, float]],
                 latent_prefix: str) -> dict[str, Any]:
    g: dict[str, Any] = {
        "10": {"class_type": "UNETLoader",
               "inputs": {"unet_name": args.unet_name, "weight_dtype": "default"}},
        "11": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": args.clip_name, "type": args.clip_type, "device": "default"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": args.prompt}},
        "13": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "5": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": args.width, "height": args.height, "batch_size": 1}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": args.vae_name}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["12", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": f"pidtile/{args.label}_vae"}},
        "7": {"class_type": "SaveLatent",
              "inputs": {"samples": ["3", 0], "filename_prefix": latent_prefix}},
    }
    prev = ["10", 0]
    for idx, (ln, ls) in enumerate(loras):
        nid = str(40 + idx)
        g[nid] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": prev, "lora_name": ln, "strength_model": ls}}
        prev = [nid, 0]
    g["3"] = {"class_type": "KSampler",
              "inputs": {"model": prev, "positive": ["6", 0], "negative": ["13", 0],
                         "latent_image": ["5", 0], "seed": args.seed, "steps": args.steps,
                         "cfg": args.cfg, "sampler_name": args.sampler_name,
                         "scheduler": args.scheduler, "denoise": 1.0}}
    return g


def pid_common_nodes(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "20": {"class_type": "UNETLoader",
               "inputs": {"unet_name": args.pid_unet, "weight_dtype": "default"}},
        "21": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": PID_CLIP, "type": "pixeldit", "device": "default"}},
        "22": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["21", 0], "text": args.pid_prompt}},
        "23": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["22", 0]}},
        "26": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}},
        "27": {"class_type": "ManualSigmas", "inputs": {"sigmas": PID_SIGMAS}},
    }


def tile_graph(args: argparse.Namespace, tile_refs: dict[str, str],
               plan: TilePlan) -> tuple[dict[str, Any], dict[str, Tile]]:
    g = pid_common_nodes(args)
    g["30"] = {"class_type": "VAELoader", "inputs": {"vae_name": "pixel_space"}}
    save_nodes: dict[str, Tile] = {}
    nid = 100
    for tile in plan.tiles:
        load, cond, empty, samp, dec, save = (str(nid + i) for i in range(6))
        nid += 10
        g[load] = {"class_type": "LoadLatent",
                   "inputs": {"latent": tile_refs[tile.name]}}
        g[cond] = {"class_type": "PiDConditioning",
                   "inputs": {"positive": ["22", 0], "latent": [load, 0],
                              "latent_format": args.latent_format,
                              "degrade_sigma": args.degrade_sigma}}
        g[empty] = {"class_type": "EmptyChromaRadianceLatentImage",
                    "inputs": {"width": tile.w * PID_SCALE, "height": tile.h * PID_SCALE,
                               "batch_size": 1}}
        g[samp] = {"class_type": "SamplerCustom",
                   "inputs": {"model": ["20", 0], "add_noise": True, "noise_seed": tile.seed,
                              "cfg": 1.0, "positive": [cond, 0], "negative": ["23", 0],
                              "sampler": ["26", 0], "sigmas": ["27", 0],
                              "latent_image": [empty, 0]}}
        g[dec] = {"class_type": "VAEDecode", "inputs": {"samples": [samp, 0], "vae": ["30", 0]}}
        g[save] = {"class_type": "SaveImage",
                   "inputs": {"images": [dec, 0],
                              "filename_prefix": f"pidtile/{args.label}_tile_{tile.name}"}}
        save_nodes[save] = tile
    return g, save_nodes


def fullframe_graph(args: argparse.Namespace, latent_ref: str, width: int, height: int) -> dict[str, Any]:
    g = pid_common_nodes(args)
    g["50"] = {"class_type": "LoadLatent", "inputs": {"latent": latent_ref}}
    g["30"] = {"class_type": "VAELoader", "inputs": {"vae_name": "pixel_space"}}
    g["60"] = {"class_type": "PiDConditioning",
               "inputs": {"positive": ["22", 0], "latent": ["50", 0],
                          "latent_format": args.latent_format,
                          "degrade_sigma": args.degrade_sigma}}
    g["61"] = {"class_type": "EmptyChromaRadianceLatentImage",
               "inputs": {"width": width * PID_SCALE, "height": height * PID_SCALE,
                          "batch_size": 1}}
    g["62"] = {"class_type": "SamplerCustom",
               "inputs": {"model": ["20", 0], "add_noise": True, "noise_seed": args.seed,
                          "cfg": 1.0, "positive": ["60", 0], "negative": ["23", 0],
                          "sampler": ["26", 0], "sigmas": ["27", 0],
                          "latent_image": ["61", 0]}}
    g["63"] = {"class_type": "VAEDecode", "inputs": {"samples": ["62", 0], "vae": ["30", 0]}}
    g["64"] = {"class_type": "SaveImage",
               "inputs": {"images": ["63", 0],
                          "filename_prefix": f"pidtile/{args.label}_full"}}
    return g


# ---------------------------------------------------------------------------
# Blend
# ---------------------------------------------------------------------------

def _axis_weights(pos_sizes: list[tuple[int, int]], scale: int) -> list["np.ndarray"]:
    """Per-tile 1D weight vectors (output px). Raised-cosine ramps over the exact
    overlap band with each neighbor; complementary ramps sum to 1."""
    import numpy as np
    weights = []
    for i, (pos, size) in enumerate(pos_sizes):
        w = np.ones(size * scale, dtype=np.float32)
        if i > 0:
            prev_pos, prev_size = pos_sizes[i - 1]
            band = (prev_pos + prev_size - pos) * scale
            if band > 0:
                u = (np.arange(band, dtype=np.float32) + 0.5) / band
                w[:band] = 0.5 - 0.5 * np.cos(np.pi * u)
        if i < len(pos_sizes) - 1:
            next_pos, _next_size = pos_sizes[i + 1]
            band = (pos + size - next_pos) * scale
            if band > 0:
                u = (np.arange(band, dtype=np.float32) + 0.5) / band
                w[-band:] = np.minimum(w[-band:], (0.5 + 0.5 * np.cos(np.pi * u)))
        weights.append(w)
    return weights


def blend_tiles(plan: TilePlan, tile_images: dict[str, "np.ndarray"]) -> "np.ndarray":
    """Feather-blend tiles (float32 HxWx3 in [0,1]) into the full output image."""
    import numpy as np
    scale = PID_SCALE
    W, H = plan.image_w * scale, plan.image_h * scale
    acc = np.zeros((H, W, 3), dtype=np.float32)
    den = np.zeros((H, W, 1), dtype=np.float32)
    wx = _axis_weights(plan.xs, scale)
    wy = _axis_weights(plan.ys, scale)
    for tile in plan.tiles:
        img = tile_images[tile.name]
        th, tw = tile.h * scale, tile.w * scale
        if img.shape[:2] != (th, tw):
            raise ValueError(f"tile {tile.name} is {img.shape[1]}x{img.shape[0]}, expected {tw}x{th}")
        w2d = (wy[tile.row][:, None] * wx[tile.col][None, :]).astype(np.float32)[..., None]
        ys, xs = tile.y * scale, tile.x * scale
        acc[ys:ys + th, xs:xs + tw] += img * w2d
        den[ys:ys + th, xs:xs + tw] += w2d
    if float(den.min()) <= 0:
        raise RuntimeError("blend hole: some output pixels are covered by no tile")
    return acc / den


# ---------------------------------------------------------------------------
# QA metrics
# ---------------------------------------------------------------------------

def shadow_green_bias(img: "np.ndarray", luma_thresh: float = 0.24) -> float:
    """Mean (G - (R+B)/2) over shadow pixels, on the 0-255 scale. img float [0,1]."""
    import numpy as np
    luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    mask = luma < luma_thresh
    if not bool(mask.any()):
        return float("nan")
    bias = img[..., 1][mask] - 0.5 * (img[..., 0][mask] + img[..., 2][mask])
    return float(bias.mean() * 255.0)


def midtone_chroma_delta(img: "np.ndarray", ref: "np.ndarray") -> float:
    """Mean opponent-chroma distance vs ref over ref midtones (0-255 scale).

    Complements shadow_green_bias: the past-envelope PiD collapse at large
    outputs shows up as a MIDTONE magenta/desaturation shift (white trim goes
    pink) that a shadow-green mean misses entirely. img must already be
    resized to ref's resolution."""
    import numpy as np
    luma = 0.2126 * ref[..., 0] + 0.7152 * ref[..., 1] + 0.0722 * ref[..., 2]
    mask = (luma > 0.25) & (luma < 0.8)
    if not bool(mask.any()):
        return float("nan")

    def opp(x):
        return x[..., 0] - x[..., 1], 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]

    rg_a, yb_a = opp(img)
    rg_b, yb_b = opp(ref)
    d = np.sqrt((rg_a - rg_b) ** 2 + (yb_a - yb_b) ** 2)
    return float(d[mask].mean() * 255.0)


def resize_to(img: "np.ndarray", w: int, h: int) -> "np.ndarray":
    import numpy as np
    from PIL import Image
    im = Image.fromarray(np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return np.asarray(im.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0


def psnr(a: "np.ndarray", b: "np.ndarray") -> float:
    import numpy as np
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def overlap_agreement(plan: TilePlan, tile_images: dict[str, "np.ndarray"]) -> list[dict[str, Any]]:
    """PSNR between neighboring raw tiles on their shared overlap rectangle."""
    out = []
    scale = PID_SCALE
    by_rc = {(t.row, t.col): t for t in plan.tiles}
    for t in plan.tiles:
        for dr, dc in ((0, 1), (1, 0)):
            n = by_rc.get((t.row + dr, t.col + dc))
            if n is None:
                continue
            x0, y0 = max(t.x, n.x), max(t.y, n.y)
            x1, y1 = min(t.x + t.w, n.x + n.w), min(t.y + t.h, n.y + n.h)
            if x1 <= x0 or y1 <= y0:
                continue
            a = tile_images[t.name][(y0 - t.y) * scale:(y1 - t.y) * scale,
                                    (x0 - t.x) * scale:(x1 - t.x) * scale]
            b = tile_images[n.name][(y0 - n.y) * scale:(y1 - n.y) * scale,
                                    (x0 - n.x) * scale:(x1 - n.x) * scale]
            out.append({"pair": f"{t.name}~{n.name}",
                        "axis": "v" if dc else "h",
                        "overlap_px": [(x1 - x0) * scale, (y1 - y0) * scale],
                        "psnr_db": round(psnr(a, b), 2)})
    return out


def seam_gradient_ratios(plan: TilePlan, blended: "np.ndarray") -> list[dict[str, Any]]:
    """|gradient| across each seam center vs nearby baseline. ~1.0 = invisible."""
    import numpy as np
    scale = PID_SCALE
    luma = (0.2126 * blended[..., 0] + 0.7152 * blended[..., 1] + 0.0722 * blended[..., 2])
    H, W = luma.shape
    out = []

    def ratio(grad_line: float, grads_ref: list[float]) -> float:
        ref = max(1e-6, sum(grads_ref) / max(1, len(grads_ref)))
        return grad_line / ref

    gx = np.abs(np.diff(luma, axis=1))  # gx[:, x] = |I[:,x+1]-I[:,x]|
    for sx in plan.seam_centers_x():
        x = sx * scale
        line = float(gx[:, max(0, x - 1):x + 2].mean())
        refs = [float(gx[:, max(0, x + off - 1):x + off + 2].mean())
                for off in (-160, -96, -48, 48, 96, 160)
                if 1 <= x + off < W - 2]
        out.append({"seam": f"v@{x}px", "ratio": round(ratio(line, refs), 3)})
    gy = np.abs(np.diff(luma, axis=0))
    for sy in plan.seam_centers_y():
        y = sy * scale
        line = float(gy[max(0, y - 1):y + 2, :].mean())
        refs = [float(gy[max(0, y + off - 1):y + off + 2, :].mean())
                for off in (-160, -96, -48, 48, 96, 160)
                if 1 <= y + off < H - 2]
        out.append({"seam": f"h@{y}px", "ratio": round(ratio(line, refs), 3)})
    return out


# ---------------------------------------------------------------------------
# Review assets
# ---------------------------------------------------------------------------

def _font(size: int):
    from PIL import ImageFont
    for name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def labeled_panel(img: "Image.Image", label: str, bar_h: int = 44) -> "Image.Image":
    from PIL import Image, ImageDraw
    panel = Image.new("RGB", (img.width, img.height + bar_h), (18, 18, 18))
    panel.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), label, fill=(240, 240, 240), font=_font(26))
    return panel


def make_strip(panels: list[tuple["Image.Image", str]], gap: int = 8) -> "Image.Image":
    from PIL import Image
    labeled = [labeled_panel(im, lb) for im, lb in panels]
    h = max(p.height for p in labeled)
    w = sum(p.width for p in labeled) + gap * (len(labeled) - 1)
    strip = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for p in labeled:
        strip.paste(p, (x, 0))
        x += p.width + gap
    return strip


def crop_at(img: "Image.Image", cx: int, cy: int, size: int) -> "Image.Image":
    x0 = max(0, min(img.width - size, cx - size // 2))
    y0 = max(0, min(img.height - size, cy - size // 2))
    return img.crop((x0, y0, x0 + size, y0 + size))


def crop_locations(plan: TilePlan) -> list[tuple[int, int, str]]:
    """Output-px (cx, cy, name): image center, seam midpoints, first 4-corner junction."""
    s = PID_SCALE
    W, H = plan.image_w * s, plan.image_h * s
    locs = [(W // 2, H // 2, "center")]
    for i, sx in enumerate(plan.seam_centers_x()):
        locs.append((sx * s, H // 2, f"vseam{i}"))
    for j, sy in enumerate(plan.seam_centers_y()):
        locs.append((W // 2, sy * s, f"hseam{j}"))
    if plan.seam_centers_x() and plan.seam_centers_y():
        locs.append((plan.seam_centers_x()[0] * s, plan.seam_centers_y()[0] * s, "junction"))
    return locs[:6]


def contact_sheet(plan: TilePlan, tile_paths: dict[str, Path], thumb_w: int = 448) -> "Image.Image":
    from PIL import Image, ImageDraw
    thumbs = []
    for tile in plan.tiles:
        im = Image.open(tile_paths[tile.name]).convert("RGB")
        th = round(im.height * thumb_w / im.width)
        im = im.resize((thumb_w, th), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width - 1, im.height - 1], outline=(255, 90, 90), width=3)
        d.text((10, 8), f"{tile.name}  {tile.w}x{tile.h}@({tile.x},{tile.y})  seed {tile.seed}",
               fill=(255, 240, 200), font=_font(20))
        thumbs.append(im)
    th = max(t.height for t in thumbs)
    gap = 6
    sheet = Image.new("RGB", (plan.cols * thumb_w + gap * (plan.cols - 1),
                              plan.rows * th + gap * (plan.rows - 1)), (24, 24, 24))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, plan.cols)
        sheet.paste(t, (c * (thumb_w + gap), r * (th + gap)))
    return sheet


def seam_map(plan: TilePlan, blended_preview: "Image.Image") -> "Image.Image":
    from PIL import ImageDraw
    im = blended_preview.copy()
    sx = im.width / (plan.image_w * PID_SCALE)
    sy = im.height / (plan.image_h * PID_SCALE)
    d = ImageDraw.Draw(im)
    for tile in plan.tiles:
        d.rectangle([tile.x * PID_SCALE * sx, tile.y * PID_SCALE * sy,
                     (tile.x + tile.w) * PID_SCALE * sx - 1,
                     (tile.y + tile.h) * PID_SCALE * sy - 1],
                    outline=(80, 255, 120), width=2)
    for xc in plan.seam_centers_x():
        d.line([xc * PID_SCALE * sx, 0, xc * PID_SCALE * sx, im.height],
               fill=(255, 160, 40), width=2)
    for yc in plan.seam_centers_y():
        d.line([0, yc * PID_SCALE * sy, im.width, yc * PID_SCALE * sy],
               fill=(255, 160, 40), width=2)
    d.text((12, 10), f"{plan.cols}x{plan.rows} tiles, overlap {plan.overlap}px(s1) "
                     f"= {plan.overlap * PID_SCALE}px(out)", fill=(255, 255, 255), font=_font(24))
    return im


def write_review_html(out_dir: Path, ctx: dict[str, Any]) -> Path:
    rows = "".join(
        f"<tr><td>{m['name']}</td><td>{m['value']}</td><td class='{m.get('cls','')}'>"
        f"{m.get('verdict','')}</td></tr>"
        for m in ctx["metrics"])
    sections = []
    for title, items in ctx["sections"]:
        imgs = "".join(f"<figure><img src='{p}' loading='lazy'><figcaption>{cap}</figcaption></figure>"
                       for p, cap in items)
        sections.append(f"<h2>{title}</h2>{imgs}")
    html = f"""<!doctype html><meta charset="utf-8">
<title>PiD tiled decode review — {ctx['label']}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#141414;color:#eee;margin:24px;max-width:1720px}}
h1{{font-size:22px}} h2{{font-size:18px;margin-top:28px;border-bottom:1px solid #333;padding-bottom:4px}}
table{{border-collapse:collapse;margin:12px 0}} td,th{{border:1px solid #333;padding:6px 12px;font-size:14px}}
img{{max-width:100%;height:auto;border:1px solid #333;margin:6px 0}}
figure{{margin:10px 0}} figcaption{{font-size:13px;color:#aaa}}
.pass{{color:#7CFC90}} .warn{{color:#FFB347}} .fail{{color:#FF6B6B}}
code{{background:#222;padding:1px 5px;border-radius:3px}}
</style>
<h1>PiD tiled decode — {ctx['label']}</h1>
<p>{ctx['summary']}</p>
<table><tr><th>metric</th><th>value</th><th>verdict</th></tr>{rows}</table>
{''.join(sections)}
<p style="color:#888">Generated {ctx['when']} by tools/pid_tiled_decode.py</p>
"""
    path = out_dir / "review.html"
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Self-test (no server, no GPU)
# ---------------------------------------------------------------------------

def self_test() -> int:
    import numpy as np
    rng = np.random.default_rng(7)
    # planner invariants over a spread of sizes
    cases = [(1344, 768, 512, 64), (1344, 768, 768, 64), (1920, 1088, 1024, 64),
             (1920, 1088, 1024, 96), (2048, 1152, 1024, 64), (1000, 568, 512, 64),
             (4096, 2304, 1024, 128), (768, 768, 1024, 64)]
    for w, h, mt, ov in cases:
        w, h = _ceil_to(w, ALIGN), _ceil_to(h, ALIGN)
        plan = make_plan(w, h, mt, ov, 1)
        for (pos, size) in plan.xs + plan.ys:
            assert pos % ALIGN == 0 and size % ALIGN == 0
        assert plan.xs[-1][0] + plan.xs[-1][1] == w
        assert plan.ys[-1][0] + plan.ys[-1][1] == h
        print(f"[plan] {w}x{h} max={mt} ov={ov} -> {plan.cols}x{plan.rows} "
              f"tiles of {plan.xs[0][1]}x{plan.ys[0][1]}")
    # blend: constant tiles must reproduce exactly; weight field must have no holes
    plan = make_plan(1344, 768, 512, 64, 1)
    tiles = {t.name: np.full((t.h * PID_SCALE, t.w * PID_SCALE, 3), 0.5, dtype=np.float32)
             for t in plan.tiles}
    out = blend_tiles(plan, tiles)
    assert abs(float(out.mean()) - 0.5) < 1e-5 and float(np.ptp(out)) < 1e-5
    # blend: smooth gradient tiles cut from one global image must reconstruct it
    W, H = plan.image_w * PID_SCALE, plan.image_h * PID_SCALE
    gx = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    gy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    world = np.concatenate([gx * np.ones((H, 1, 1), np.float32),
                            gy * np.ones((1, W, 1), np.float32),
                            0.25 * np.ones((H, W, 1), np.float32)], axis=2)
    tiles = {t.name: world[t.y * PID_SCALE:(t.y + t.h) * PID_SCALE,
                           t.x * PID_SCALE:(t.x + t.w) * PID_SCALE].copy()
             for t in plan.tiles}
    out = blend_tiles(plan, tiles)
    assert float(np.abs(out - world).max()) < 1e-4
    # overlap agreement on identical crops = 99 dB; seam ratios ~1 on noise
    agree = overlap_agreement(plan, tiles)
    assert agree and all(a["psnr_db"] >= 99.0 for a in agree)
    noise_world = rng.random((H, W, 3), dtype=np.float32)
    ratios = seam_gradient_ratios(plan, noise_world)
    assert all(0.8 < r["ratio"] < 1.2 for r in ratios), ratios
    # green bias sanity: pure gray = 0, green-tinted shadows > 3
    gray = np.full((64, 64, 3), 0.1, np.float32)
    assert abs(shadow_green_bias(gray)) < 1e-6
    tinted = gray.copy()
    tinted[..., 1] += 0.02
    assert shadow_green_bias(tinted) > 3.0
    # latent crop slices the LAST two dims for both 4-dim and 5-dim layouts
    lat5 = np.arange(2 * 16 * 1 * 96 * 168).reshape(2, 16, 1, 96, 168)
    c = crop_latent_tensor(lat5, x=424, y=352, w=496, h=416)
    assert c.shape == (2, 16, 1, 52, 62), c.shape
    assert (c == lat5[:, :, :, 44:96, 53:115]).all()
    lat4 = np.arange(1 * 16 * 96 * 168).reshape(1, 16, 96, 168)
    c4 = crop_latent_tensor(lat4, x=0, y=0, w=496, h=416)
    assert c4.shape == (1, 16, 52, 62), c4.shape
    print("SELF-TEST PASS")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_rgb(path: Path) -> "np.ndarray":
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(img: "np.ndarray", path: Path, quality: int | None = None) -> None:
    import numpy as np
    from PIL import Image
    arr = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if quality is not None:
        im.save(path, quality=quality)
    else:
        im.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--prompt", help="stage-1 prompt (fresh render)")
    src.add_argument("--from-latent", help="existing .latent (e.g. 'latents/x.latent' under ComfyUI output)")
    ap.add_argument("--vae-image", type=Path, default=None,
                    help="stage-1 VAE decode PNG for QA/review (only with --from-latent)")
    ap.add_argument("--label", default=None, help="run label (required unless --self-test)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (required unless --self-test)")
    ap.add_argument("--comfy-url", default=DEFAULT_COMFY)
    ap.add_argument("--comfy-root", default=DEFAULT_COMFY_ROOT, type=Path)
    # stage-1 (only used with --prompt; bring your own qwen-family model)
    ap.add_argument("--unet-name", default=None, help="stage-1 UNET checkpoint (required with --prompt)")
    ap.add_argument("--clip-name", default=None, help="stage-1 text encoder (required with --prompt)")
    ap.add_argument("--clip-type", default="qwen_image", help="CLIPLoader type for the stage-1 text encoder")
    ap.add_argument("--vae-name", default=STAGE1_VAE)
    ap.add_argument("--lora", default="", help="FILE[:STRENGTH],... ('' = none)")
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--sampler-name", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--seed", type=int, default=508040000)
    # PiD decode
    ap.add_argument("--pid-unet", default=PID_UNET)
    ap.add_argument("--pid-prompt", default="")
    ap.add_argument("--latent-format", default="qwenimage",
                    choices=["flux", "sd3", "sdxl", "qwenimage"])
    ap.add_argument("--degrade-sigma", type=float, default=0.0)
    # tiling
    ap.add_argument("--max-tile", type=int, default=1024,
                    help="max tile size per axis, stage-1 px (1024 = PiD trained envelope)")
    ap.add_argument("--overlap", type=int, default=64,
                    help="tile overlap, stage-1 px (x4 in output)")
    ap.add_argument("--full-frame", action="store_true",
                    help="also run the single-shot full-frame PiD decode for A/B")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--poll", type=float, default=4.0)
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.label or not args.out or not (args.prompt or args.from_latent):
        ap.error("--label, --out and one of --prompt/--from-latent are required")
    if args.prompt and not (args.unet_name and args.clip_name):
        ap.error("--prompt mode needs --unet-name and --clip-name (or use --from-latent "
                 "with a latent saved from your own workflow via SaveLatent)")
    if args.width % ALIGN or args.height % ALIGN:
        ap.error(f"--width/--height must be multiples of {ALIGN}")
    if args.max_tile % ALIGN or args.overlap % ALIGN:
        ap.error(f"--max-tile/--overlap must be multiples of {ALIGN}")

    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    t_start = time.time()
    out_dir: Path = args.out
    tiles_dir = out_dir / "tiles"
    review_dir = out_dir / "review"
    for d in (out_dir, tiles_dir, review_dir):
        d.mkdir(parents=True, exist_ok=True)

    require_server(args.comfy_url)
    require_idle(args.comfy_url)

    manifest: dict[str, Any] = {
        "tool": "pid_tiled_decode",
        "when": dt.datetime.now().isoformat(timespec="seconds"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "pid": {"unet": args.pid_unet, "clip": PID_CLIP, "sigmas": PID_SIGMAS,
                "sampler": "lcm", "cfg": 1.0, "scale": PID_SCALE},
        "phases": {},
    }
    timings: dict[str, float] = {}

    # ---- Phase A: stage-1 render (or reuse latent) --------------------------
    vae_path: Path | None = None
    if args.from_latent:
        latent_ref, latent_file = resolve_latent_ref(args.from_latent, args.comfy_root)
        if not latent_file.exists():
            raise SystemExit(f"latent file not found: {latent_file}")
        args.width, args.height = latent_file_dims(latent_file)
        print(f"[latent] reusing {latent_file.name} -> stage-1 {args.width}x{args.height}")
        if args.vae_image:
            if not args.vae_image.exists():
                raise SystemExit(f"--vae-image not found: {args.vae_image}")
            vae_path = out_dir / "stage1_vae.png"
            shutil.copy2(args.vae_image, vae_path)
    else:
        t0 = time.time()
        loras = parse_loras(args.lora)
        graph = stage1_graph(args, loras, latent_prefix=f"latents/pidtile_{args.label}")
        pid_a = submit(args.comfy_url, graph)
        print(f"[stage1] submitted {pid_a} ({args.width}x{args.height}, seed {args.seed})")
        entry = wait_for_history(args.comfy_url, pid_a, args.timeout, args.poll, "stage1")
        latents = collect_files(args.comfy_root, entry, "7", kind="latents")
        vaes = collect_files(args.comfy_root, entry, "9", kind="images")
        if not latents or not vaes:
            raise RuntimeError("stage-1 outputs missing (latent or VAE baseline)")
        latent_file = latents[0]
        rel = latent_file.relative_to(args.comfy_root / "output").as_posix()
        latent_ref = f"{rel} [output]"
        vae_path = out_dir / "stage1_vae.png"
        shutil.copy2(vaes[0], vae_path)
        timings["stage1_s"] = round(time.time() - t0, 1)
        manifest["phases"]["stage1"] = {"prompt_id": pid_a, "latent": rel,
                                        "vae_baseline": vae_path.name}
        print(f"[stage1] done in {timings['stage1_s']}s -> {rel}")

    manifest["latent_ref"] = latent_ref
    manifest["stage1_dims"] = [args.width, args.height]
    out_w, out_h = args.width * PID_SCALE, args.height * PID_SCALE

    # ---- Plan tiles ---------------------------------------------------------
    plan = make_plan(args.width, args.height, args.max_tile, args.overlap, args.seed)
    print(f"[plan] {args.width}x{args.height} -> {out_w}x{out_h} output, "
          f"{plan.cols}x{plan.rows} tiles of {plan.xs[0][1]}x{plan.ys[0][1]} "
          f"(overlap {args.overlap} s1-px)")
    manifest["plan"] = {
        "cols": plan.cols, "rows": plan.rows, "overlap_s1": args.overlap,
        "tiles": [{"name": t.name, "x": t.x, "y": t.y, "w": t.w, "h": t.h,
                   "seed": t.seed} for t in plan.tiles],
    }

    # ---- Phase B: tile decodes ---------------------------------------------
    t0 = time.time()
    tile_refs = write_tile_latents(latent_file, plan, args.comfy_root, args.label)
    manifest["tile_latents"] = tile_refs
    graph, save_nodes = tile_graph(args, tile_refs, plan)
    pid_b = submit(args.comfy_url, graph)
    print(f"[tiles] submitted {pid_b} ({len(plan.tiles)} tiles)")
    entry = wait_for_history(args.comfy_url, pid_b, args.timeout, args.poll, "tiles")
    tile_paths: dict[str, Path] = {}
    for node_id, tile in save_nodes.items():
        files = collect_files(args.comfy_root, entry, node_id)
        if not files:
            raise RuntimeError(f"tile {tile.name} produced no image")
        dest = tiles_dir / f"{tile.name}.png"
        shutil.copy2(files[0], dest)
        tile_paths[tile.name] = dest
    timings["tiles_s"] = round(time.time() - t0, 1)
    manifest["phases"]["tiles"] = {"prompt_id": pid_b, "count": len(tile_paths)}
    print(f"[tiles] done in {timings['tiles_s']}s")

    # ---- Blend --------------------------------------------------------------
    t0 = time.time()
    tile_images = {name: load_rgb(p) for name, p in tile_paths.items()}
    agreement = overlap_agreement(plan, tile_images)
    blended = blend_tiles(plan, tile_images)
    master_path = out_dir / "pid_tiled.png"
    save_rgb(blended, master_path)
    preview = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
    pw = 1680
    preview_im = preview.resize((pw, round(preview.height * pw / preview.width)), Image.LANCZOS)
    preview_path = out_dir / "pid_tiled_preview.jpg"
    preview_im.save(preview_path, quality=90)
    timings["blend_s"] = round(time.time() - t0, 1)
    print(f"[blend] {out_w}x{out_h} master written in {timings['blend_s']}s")

    # ---- Phase C: optional full-frame A/B ----------------------------------
    full_path: Path | None = None
    full_error: str | None = None
    if args.full_frame:
        t0 = time.time()
        try:
            graph = fullframe_graph(args, latent_ref, args.width, args.height)
            pid_c = submit(args.comfy_url, graph)
            print(f"[full] submitted {pid_c} (single-shot {out_w}x{out_h})")
            entry = wait_for_history(args.comfy_url, pid_c, args.timeout, args.poll, "full")
            files = collect_files(args.comfy_root, entry, "64")
            if files:
                full_path = out_dir / "pid_full.png"
                shutil.copy2(files[0], full_path)
            timings["full_s"] = round(time.time() - t0, 1)
            manifest["phases"]["full_frame"] = {"prompt_id": pid_c, "ok": full_path is not None}
        except Exception as exc:
            full_error = f"{type(exc).__name__}: {exc}"
            manifest["phases"]["full_frame"] = {"ok": False, "error": full_error[:2000]}
            print(f"[full] single-shot decode FAILED (continuing): {full_error[:300]}")

    # ---- QA -----------------------------------------------------------------
    qa: dict[str, Any] = {"overlap_agreement": agreement}
    qa["seam_ratios"] = seam_gradient_ratios(plan, blended)
    qa["green_bias_tiled"] = shadow_green_bias(blended)
    vae_img = load_rgb(vae_path) if vae_path else None
    if vae_img is not None:
        qa["green_bias_vae"] = shadow_green_bias(vae_img)
        qa["green_bias_delta_tiled"] = round(qa["green_bias_tiled"] - qa["green_bias_vae"], 2)
        qa["chroma_delta_tiled"] = round(midtone_chroma_delta(
            resize_to(blended, args.width, args.height), vae_img), 2)
        per_tile = []
        for t in plan.tiles:
            crop = vae_img[t.y:t.y + t.h, t.x:t.x + t.w]
            delta = shadow_green_bias(tile_images[t.name]) - shadow_green_bias(crop)
            per_tile.append({"tile": t.name, "delta_vs_vae": None if math.isnan(delta)
                             else round(delta, 2)})
        qa["green_bias_per_tile"] = per_tile
    full_img = load_rgb(full_path) if full_path else None
    if full_img is not None:
        qa["green_bias_full"] = shadow_green_bias(full_img)
        if vae_img is not None:
            qa["green_bias_delta_full"] = round(qa["green_bias_full"] - qa["green_bias_vae"], 2)
            qa["chroma_delta_full"] = round(midtone_chroma_delta(
                resize_to(full_img, args.width, args.height), vae_img), 2)

    worst_seam = max((r["ratio"] for r in qa["seam_ratios"]), default=1.0)
    min_agree = min((a["psnr_db"] for a in agreement), default=99.0)
    delta_tiled = qa.get("green_bias_delta_tiled")
    chroma_tiled = qa.get("chroma_delta_tiled")
    qa["gates"] = {
        "seam_worst_ratio": worst_seam,
        "seam_pass": worst_seam <= SEAM_RATIO_FLAG,
        "overlap_min_psnr_db": min_agree,
        "green_gate_pts": GREEN_BIAS_GATE_PTS,
        "green_pass_tiled": (delta_tiled is None) or (abs(delta_tiled) <= GREEN_BIAS_GATE_PTS),
        "chroma_gate_pts": CHROMA_GATE_PTS,
        "chroma_pass_tiled": (chroma_tiled is None) or (chroma_tiled <= CHROMA_GATE_PTS),
    }
    (out_dir / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")

    # ---- Review assets ------------------------------------------------------
    print("[review] building crops / sheets / html")
    sheet = contact_sheet(plan, tile_paths)
    sheet_path = review_dir / "tile_contact_sheet.jpg"
    sheet.save(sheet_path, quality=88)
    smap = seam_map(plan, preview_im)
    smap_path = review_dir / "seam_map.jpg"
    smap.save(smap_path, quality=90)

    tiled_pil = Image.open(master_path).convert("RGB")
    full_pil = Image.open(full_path).convert("RGB") if full_path else None
    vae_up = None
    if vae_path:
        with Image.open(vae_path) as v:
            vae_up = v.convert("RGB").resize((out_w, out_h), Image.LANCZOS)

    crop_files: list[tuple[str, str]] = []
    crop_px = 512
    for cx, cy, name in crop_locations(plan):
        panels: list[tuple[Image.Image, str]] = []
        if vae_up is not None:
            panels.append((crop_at(vae_up, cx, cy, crop_px), "VAE decode, lanczos x4"))
        if full_pil is not None:
            panels.append((crop_at(full_pil, cx, cy, crop_px), "PiD single-shot"))
        panels.append((crop_at(tiled_pil, cx, cy, crop_px), "PiD TILED (this tool)"))
        strip = make_strip(panels)
        p = review_dir / f"crop_{name}.png"
        strip.save(p)
        crop_files.append((f"review/{p.name}", f"100% crop at {name} ({cx},{cy})"))

    def verdict(ok: bool | None) -> tuple[str, str]:
        if ok is None:
            return "n/a", "warn"
        return ("PASS", "pass") if ok else ("CHECK", "fail")

    g = qa["gates"]
    metrics = [
        {"name": "output size", "value": f"{out_w} x {out_h} ({out_w*out_h/1e6:.1f} MP)"},
        {"name": "tiles", "value": f"{plan.cols} x {plan.rows} of "
         f"{plan.xs[0][1]*PID_SCALE} x {plan.ys[0][1]*PID_SCALE} out-px, "
         f"overlap {args.overlap*PID_SCALE} out-px"},
        {"name": "worst seam gradient ratio", "value": f"{g['seam_worst_ratio']:.3f} "
         f"(flag > {SEAM_RATIO_FLAG})", **dict(zip(("verdict", "cls"), verdict(g["seam_pass"])))},
        {"name": "min overlap agreement", "value": f"{g['overlap_min_psnr_db']:.1f} dB"},
        {"name": "shadow green-bias delta vs VAE (tiled)",
         "value": f"{delta_tiled if delta_tiled is not None else 'n/a'} pts "
         f"(gate +/-{GREEN_BIAS_GATE_PTS})",
         **dict(zip(("verdict", "cls"), verdict(g["green_pass_tiled"] if delta_tiled is not None else None)))},
    ]
    if chroma_tiled is not None:
        metrics.append({"name": "midtone chroma delta vs VAE (tiled)",
                        "value": f"{chroma_tiled} pts (flag > {CHROMA_GATE_PTS})",
                        **dict(zip(("verdict", "cls"), verdict(qa["gates"]["chroma_pass_tiled"])))})
    if "green_bias_delta_full" in qa:
        ok = abs(qa["green_bias_delta_full"]) <= GREEN_BIAS_GATE_PTS
        metrics.append({"name": "shadow green-bias delta vs VAE (single-shot)",
                        "value": f"{qa['green_bias_delta_full']} pts",
                        **dict(zip(("verdict", "cls"), verdict(ok)))})
    if "chroma_delta_full" in qa:
        ok = qa["chroma_delta_full"] <= CHROMA_GATE_PTS
        metrics.append({"name": "midtone chroma delta vs VAE (single-shot)",
                        "value": f"{qa['chroma_delta_full']} pts",
                        **dict(zip(("verdict", "cls"), verdict(ok)))})
    if full_error:
        metrics.append({"name": "single-shot full-frame decode",
                        "value": full_error[:160], "verdict": "FAILED", "cls": "fail"})

    sections: list[tuple[str, list[tuple[str, str]]]] = []
    overview: list[tuple[str, str]] = [("pid_tiled_preview.jpg",
                                       f"Tiled PiD master (preview; full res: pid_tiled.png)")]
    if vae_path:
        overview.append((vae_path.name, "Stage-1 VAE baseline (native res)"))
    sections.append(("Overview", overview))
    sections.append(("100% crops — VAE x4 vs single-shot vs tiled", crop_files))
    sections.append(("Tiling", [("review/seam_map.jpg", "Tile layout + seam centers"),
                                ("review/tile_contact_sheet.jpg", "Raw tiles (pre-blend)")]))

    html_path = write_review_html(out_dir, {
        "label": args.label,
        "summary": (f"Stage-1 {args.width}x{args.height} -> PiD 4x {out_w}x{out_h}. "
                    f"{len(plan.tiles)} tiles, each decoded inside the trained PiD envelope, "
                    f"raised-cosine feather blend."),
        "metrics": metrics,
        "sections": sections,
        "when": dt.datetime.now().isoformat(timespec="seconds"),
    })

    # qa.md
    lines = [f"# PiD tiled decode QA — {args.label}", "",
             f"- output: {out_w}x{out_h} ({out_w*out_h/1e6:.1f} MP), "
             f"tiles {plan.cols}x{plan.rows}, overlap {args.overlap} s1-px",
             f"- worst seam ratio: {g['seam_worst_ratio']:.3f} "
             f"({'PASS' if g['seam_pass'] else 'CHECK'} at {SEAM_RATIO_FLAG})",
             f"- min overlap agreement: {g['overlap_min_psnr_db']:.1f} dB",
             f"- green-bias delta vs VAE (tiled): {delta_tiled} pts "
             f"({'PASS' if g['green_pass_tiled'] else 'CHECK'} at +/-{GREEN_BIAS_GATE_PTS})"]
    if chroma_tiled is not None:
        lines.append(f"- midtone chroma delta vs VAE (tiled): {chroma_tiled} pts "
                     f"({'PASS' if qa['gates']['chroma_pass_tiled'] else 'CHECK'} at {CHROMA_GATE_PTS})")
    if "green_bias_delta_full" in qa:
        lines.append(f"- green-bias delta vs VAE (single-shot): {qa['green_bias_delta_full']} pts")
    if "chroma_delta_full" in qa:
        lines.append(f"- midtone chroma delta vs VAE (single-shot): {qa['chroma_delta_full']} pts "
                     f"({'PASS' if qa['chroma_delta_full'] <= CHROMA_GATE_PTS else 'CHECK'} at {CHROMA_GATE_PTS})")
    if full_error:
        lines.append(f"- single-shot decode FAILED: {full_error[:200]}")
    lines += ["", "| seam | gradient ratio |", "|---|---|"]
    lines += [f"| {r['seam']} | {r['ratio']} |" for r in qa["seam_ratios"]]
    lines += ["", "| tile pair | axis | overlap PSNR (dB) |", "|---|---|---|"]
    lines += [f"| {a['pair']} | {a['axis']} | {a['psnr_db']} |" for a in agreement]
    (out_dir / "qa.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest["timings_s"] = timings
    manifest["qa_gates"] = qa["gates"]
    manifest["outputs"] = {
        "master": master_path.name, "preview": preview_path.name,
        "vae_baseline": vae_path.name if vae_path else None,
        "full_frame": full_path.name if full_path else None,
        "review_html": html_path.name,
    }
    manifest["total_s"] = round(time.time() - t_start, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[done] {out_dir}  total {manifest['total_s']}s")
    print(f"[done] review: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
