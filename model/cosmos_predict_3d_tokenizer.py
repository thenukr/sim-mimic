#!/usr/bin/env python3
"""Use NVIDIA Cosmos 3D video tokenizer VAE for encode/decode workflows.

Backends:
- `jit`: Cosmos tokenizer JIT checkpoints (encoder.jit + decoder.jit).
- `diffusers`: Cosmos Predict2 VAE (`vae/` subfolder on Hugging Face).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download


DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_download = sub.add_parser("download", help="Download tokenizer/vae checkpoints from Hugging Face.")
    p_download.add_argument(
        "--target",
        choices=["auto", "predict2", "cv4"],
        default="auto",
        help="`predict2` is exact repo used by Cosmos Predict2 (gated). `cv4` is Cosmos-0.1 CV4x8x8 tokenizer.",
    )
    p_download.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="HF token. Defaults to $HF_TOKEN.")
    p_download.add_argument("--out-dir", type=Path, default=Path("checkpoints/nvidia"))

    p_encode = sub.add_parser("encode", help="Encode a video to tokenizer latents.")
    add_common_model_args(p_encode)
    p_encode.add_argument("--video", type=Path, required=True)
    p_encode.add_argument("--out", type=Path, required=True, help="Output .pt latents file.")
    p_encode.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    p_encode.add_argument("--resize-height", type=int, default=0)
    p_encode.add_argument("--resize-width", type=int, default=0)
    p_encode.add_argument("--sample-latent", action="store_true", help="Only for diffusers backend.")

    p_decode = sub.add_parser("decode", help="Decode latents (.pt) to a video file.")
    add_common_model_args(p_decode)
    p_decode.add_argument("--latents", type=Path, required=True, help="Latents file from `encode`.")
    p_decode.add_argument("--out", type=Path, required=True, help="Output video path (.mp4).")
    p_decode.add_argument("--fps", type=float, default=0.0, help="Override output FPS. 0 uses metadata from latents.")

    p_roundtrip = sub.add_parser("roundtrip", help="Encode and decode a video in one command.")
    add_common_model_args(p_roundtrip)
    p_roundtrip.add_argument("--video", type=Path, required=True)
    p_roundtrip.add_argument("--out", type=Path, required=True, help="Output reconstructed video path.")
    p_roundtrip.add_argument("--save-latents", type=Path, default=None, help="Optional latents save path.")
    p_roundtrip.add_argument("--max-frames", type=int, default=0)
    p_roundtrip.add_argument("--resize-height", type=int, default=0)
    p_roundtrip.add_argument("--resize-width", type=int, default=0)
    p_roundtrip.add_argument("--sample-latent", action="store_true", help="Only for diffusers backend.")

    return parser.parse_args()


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["jit", "diffusers"], default="jit")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/nvidia/Cosmos-0.1-Tokenizer-CV4x8x8"),
        help="For jit backend: directory containing encoder.jit and decoder.jit.",
    )
    parser.add_argument(
        "--model-id",
        default="nvidia/Cosmos-Predict2-2B-Video2World",
        help="For diffusers backend: HF model ID containing `vae/` subfolder.",
    )
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="HF token. Defaults to $HF_TOKEN.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP.keys()), default="bfloat16")


def ensure_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")


def read_video(path: Path, max_frames: int, resize_hw: tuple[int, int] | None) -> tuple[torch.Tensor, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_hw is not None:
            frame = cv2.resize(frame, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames found in {path}")

    # [T, H, W, C] -> [1, C, T, H, W], values in [-1, 1]
    arr = np.stack(frames, axis=0).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0) / 127.5 - 1.0
    return tensor, float(fps)


def write_video(tensor: torch.Tensor, out_path: Path, fps: float, crop_hw: tuple[int, int] | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vid = tensor.detach().cpu().clamp(-1.0, 1.0)
    vid = ((vid + 1.0) * 127.5).round().to(torch.uint8)
    # [1, C, T, H, W] -> [T, H, W, C]
    frames = vid[0].permute(1, 2, 3, 0).numpy()
    if crop_hw is not None:
        h, w = crop_hw
        frames = frames[:, :h, :w, :]

    h, w = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer for: {out_path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def pad_spatial_to_16(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, _, h, w = x.shape
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    # pad order: (W_left, W_right, H_top, H_bottom)
    padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (pad_h, pad_w)


def load_jit_tokenizer(checkpoint_dir: Path, device: str, dtype_name: str):
    if not device.startswith("cuda"):
        raise RuntimeError("`jit` backend requires CUDA for this checkpoint.")
    enc = checkpoint_dir / "encoder.jit"
    dec = checkpoint_dir / "decoder.jit"
    if not enc.exists() or not dec.exists():
        raise FileNotFoundError(f"Missing encoder/decoder JIT files in {checkpoint_dir}")
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer

    return CausalVideoTokenizer(
        checkpoint_enc=str(enc),
        checkpoint_dec=str(dec),
        device=device,
        dtype=dtype_name,
    )


def load_diffusers_vae(model_id_or_path: str, hf_token: str | None, device: str, dtype_name: str):
    from diffusers import AutoencoderKLCosmos

    dtype = DTYPE_MAP[dtype_name]
    vae = AutoencoderKLCosmos.from_pretrained(
        model_id_or_path,
        subfolder="vae",
        torch_dtype=dtype,
        token=hf_token,
    )
    vae = vae.to(device)
    vae.eval()
    return vae


def encode_latents(args: argparse.Namespace) -> None:
    ensure_device(args.device)
    resize_hw = None
    if args.resize_height > 0 and args.resize_width > 0:
        resize_hw = (args.resize_height, args.resize_width)

    video, fps = read_video(args.video, args.max_frames, resize_hw)
    orig_h, orig_w = int(video.shape[-2]), int(video.shape[-1])
    video, pad_hw = pad_spatial_to_16(video)
    dtype = DTYPE_MAP[args.dtype]

    if args.backend == "jit":
        tokenizer = load_jit_tokenizer(args.checkpoint_dir, args.device, args.dtype)
        latents = tokenizer.encode(video.to(dtype=dtype, device=args.device))
        metadata: dict[str, Any] = {"checkpoint_dir": str(args.checkpoint_dir)}
    else:
        vae = load_diffusers_vae(args.model_id, args.hf_token, args.device, args.dtype)
        video_t = video.to(dtype=dtype, device=args.device)
        with torch.no_grad():
            posterior = vae.encode(video_t).latent_dist
            latents = posterior.sample() if args.sample_latent else posterior.mode()
            scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
            latents = latents * scaling_factor
        metadata = {"model_id": args.model_id, "scaling_factor": scaling_factor}

    payload = {
        "latents": latents.detach().cpu(),
        "backend": args.backend,
        "fps": fps,
        "orig_hw": (orig_h, orig_w),
        "pad_hw": pad_hw,
        "metadata": metadata,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"Saved latents: {args.out}")
    print(f"Latent shape: {tuple(payload['latents'].shape)}")


def decode_latents(args: argparse.Namespace) -> None:
    ensure_device(args.device)
    payload = torch.load(args.latents, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        payload = {"latents": payload}
    latents = payload["latents"]
    backend = payload.get("backend", args.backend)
    dtype = DTYPE_MAP[args.dtype]

    if backend == "jit":
        tokenizer = load_jit_tokenizer(args.checkpoint_dir, args.device, args.dtype)
        recon = tokenizer.decode(latents.to(dtype=dtype, device=args.device))
    else:
        meta = payload.get("metadata", {})
        model_id = meta.get("model_id", args.model_id)
        scaling_factor = float(meta.get("scaling_factor", 1.0))
        vae = load_diffusers_vae(model_id, args.hf_token, args.device, args.dtype)
        with torch.no_grad():
            recon = vae.decode(latents.to(dtype=dtype, device=args.device) / scaling_factor).sample

    fps = float(args.fps) if args.fps > 0 else float(payload.get("fps", 30.0))
    crop_hw = payload.get("orig_hw", None)
    write_video(recon, args.out, fps=fps, crop_hw=crop_hw)
    print(f"Saved video: {args.out}")


def roundtrip(args: argparse.Namespace) -> None:
    tmp_latents = args.save_latents if args.save_latents is not None else Path("tmp_cosmos_latents.pt")
    enc_ns = argparse.Namespace(**vars(args))
    enc_ns.out = tmp_latents
    encode_latents(enc_ns)

    dec_ns = argparse.Namespace(**vars(args))
    dec_ns.latents = tmp_latents
    decode_latents(dec_ns)

    if args.save_latents is None and tmp_latents.exists():
        tmp_latents.unlink()


def download_target(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def _download(repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            token=args.hf_token,
            allow_patterns=allow_patterns,
        )
        print(f"Downloaded: {repo_id} -> {local_dir}")

    if args.target == "predict2":
        _download(
            "nvidia/Cosmos-Predict2-2B-Video2World",
            args.out_dir / "Cosmos-Predict2-2B-Video2World",
            allow_patterns=["vae/*", "tokenizer/tokenizer.pth"],
        )
        return

    if args.target == "cv4":
        _download("nvidia/Cosmos-0.1-Tokenizer-CV4x8x8", args.out_dir / "Cosmos-0.1-Tokenizer-CV4x8x8")
        return

    # auto: exact Predict2 first, then CV4 fallback.
    try:
        _download(
            "nvidia/Cosmos-Predict2-2B-Video2World",
            args.out_dir / "Cosmos-Predict2-2B-Video2World",
            allow_patterns=["vae/*", "tokenizer/tokenizer.pth"],
        )
    except Exception as exc:
        print(f"Predict2 gated download failed: {exc}")
        print("Falling back to nvidia/Cosmos-0.1-Tokenizer-CV4x8x8 ...")
        _download("nvidia/Cosmos-0.1-Tokenizer-CV4x8x8", args.out_dir / "Cosmos-0.1-Tokenizer-CV4x8x8")


def main() -> None:
    args = parse_args()
    if args.cmd == "download":
        download_target(args)
    elif args.cmd == "encode":
        encode_latents(args)
    elif args.cmd == "decode":
        decode_latents(args)
    elif args.cmd == "roundtrip":
        roundtrip(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

