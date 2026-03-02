#!/usr/bin/env python3
"""Overlay BEHAVIOR-1K R1Pro proprio data on top of an RGB episode video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# R1Pro mapping from StanfordVL/BEHAVIOR-1K eval_utils.py
JOINT_QPOS = slice(0, 28)
JOINT_QVEL = slice(84, 112)
R1PRO_PROPRIO_QPOS = {
    "torso": slice(6, 10),
    "left_arm": slice(10, 24, 2),
    "right_arm": slice(11, 24, 2),
    "left_gripper": slice(24, 26),
    "right_gripper": slice(26, 28),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("b1k_preview/data/task-0000/episode_00000010.parquet"),
        help="Path to episode parquet file.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("b1k_preview/videos/task-0000/observation.images.rgb.head/episode_00000010.mp4"),
        help="Path to episode RGB video file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("b1k_preview/episode_00000010_proprio_overlay.mp4"),
        help="Output annotated mp4 path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on number of frames to render (0 means all).",
    )
    return parser.parse_args()


def fmt_vec(v: np.ndarray, digits: int = 2) -> str:
    return "[" + ", ".join(f"{x:+.{digits}f}" for x in v) + "]"


def draw_panel(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    panel = out.copy()
    margin = 10
    line_h = 24
    panel_h = margin * 2 + line_h * len(lines)
    cv2.rectangle(panel, (8, 8), (out.shape[1] - 8, 8 + panel_h), (0, 0, 0), -1)
    out = cv2.addWeighted(panel, 0.55, out, 0.45, 0)

    y = 8 + margin + 14
    for line in lines:
        cv2.putText(
            out,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_h
    return out


def main() -> None:
    args = parse_args()
    if not args.parquet.exists():
        raise FileNotFoundError(f"Parquet not found: {args.parquet}")
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    df = pd.read_parquet(args.parquet)
    if "observation.state" not in df.columns:
        raise KeyError("Missing 'observation.state' column in parquet.")

    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    timestamps = df["timestamp"].to_numpy(dtype=np.float64)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    usable_frames = min(frame_count, len(df))
    if args.max_frames > 0:
        usable_frames = min(usable_frames, args.max_frames)
    if usable_frames <= 0:
        raise RuntimeError("No frames available for rendering.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    for i in range(usable_frames):
        ok, frame = cap.read()
        if not ok:
            break

        state = states[i]
        qpos = state[JOINT_QPOS]
        qvel = state[JOINT_QVEL]

        lines = [
            f"frame {i + 1}/{usable_frames}  t={timestamps[i]:.3f}s",
            f"torso qpos: {fmt_vec(qpos[R1PRO_PROPRIO_QPOS['torso']])}",
            f"left arm qpos: {fmt_vec(qpos[R1PRO_PROPRIO_QPOS['left_arm']])}",
            f"right arm qpos: {fmt_vec(qpos[R1PRO_PROPRIO_QPOS['right_arm']])}",
            (
                "gripper qpos L/R: "
                f"{fmt_vec(qpos[R1PRO_PROPRIO_QPOS['left_gripper']])} / "
                f"{fmt_vec(qpos[R1PRO_PROPRIO_QPOS['right_gripper']])}"
            ),
            f"left arm qvel: {fmt_vec(qvel[R1PRO_PROPRIO_QPOS['left_arm']])}",
            f"right arm qvel: {fmt_vec(qvel[R1PRO_PROPRIO_QPOS['right_arm']])}",
        ]

        writer.write(draw_panel(frame, lines))

        if (i + 1) % 300 == 0:
            print(f"Rendered {i + 1}/{usable_frames} frames...")

    writer.release()
    cap.release()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

