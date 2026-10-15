import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cv2
import os
from pathlib import Path


def compute_chained_factors(factors, chunk_size, start_frame=0):
    chained = []
    frame = start_frame
    while frame < len(factors):
        f = float(factors[frame])
        chained.append((frame, f))
        skip = max(1, int(chunk_size * f))
        frame += skip
    return chained


def render_factor_video(
    demo_frames,
    factors,
    output_path,
    chunk_size=12,
    fps=10,
):
    N = len(factors)
    chained = compute_chained_factors(factors, chunk_size)

    max_factor_for_plot = max(3.0, np.max(factors) * 1.1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = demo_frames[0].shape[:2]
    plot_height = 200
    total_height = height + plot_height
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, total_height))

    fig, ax = plt.subplots(figsize=(width / 100.0, plot_height / 100.0), dpi=100)
    ax.set_xlim(0, N)
    ax.set_ylim(0, max_factor_for_plot)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Factor")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    factor_line, = ax.plot([], [], "b-", linewidth=1, alpha=0.7)
    for _, f in chained:
        ax.plot([], [])
    cursor, = ax.plot([], [], "ro", markersize=6)

    for t in range(N):
        factor_line.set_data(range(t + 1), factors[: t + 1])

        chained_x = [c[0] for c in chained if c[0] <= t]
        chained_y = [c[1] for c in chained if c[0] <= t]
        if chained_x:
            for coll in list(ax.collections):
                coll.remove()
            ax.scatter(chained_x, chained_y, c="red", marker="D", s=20, zorder=5)

        cursor.set_data([t], [factors[t]])

        fig.canvas.draw()
        plot_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plot_img = plot_img.astype(np.uint8)
        plot_img = cv2.resize(plot_img, (width, plot_height))

        main_frame = demo_frames[t]
        if main_frame.dtype != np.uint8:
            main_frame = np.clip(main_frame, 0, 255).astype(np.uint8)
        if main_frame.shape[1] != width:
            main_frame = cv2.resize(main_frame, (width, height))

        combined = np.vstack([main_frame, plot_img])
        writer.write(combined)

    plt.close(fig)
    writer.release()


def visualize_factors_for_demos(demos, factors_list, output_dir, chunk_size=12):
    output_dir = Path(output_dir) / "factor_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx, (demo, factors) in enumerate(zip(demos, factors_list)):
        frames = []
        for step in demo:
            obs = step[0]
            if "rgb_head" in obs:
                head = obs["rgb_head"].transpose(1, 2, 0)
                frames.append(head)
            elif "head" in obs:
                head = obs["head"].transpose(1, 2, 0) if isinstance(obs["head"], np.ndarray) and obs["head"].ndim >= 3 else obs["head"]
                frames.append(head)
            elif "image" in obs:
                frames.append(obs["image"])

        if not frames:
            print(f"No frames found for episode {ep_idx}, step[0] keys: {list(demo[0][0].keys())}")
            continue

        output_path = str(output_dir / f"factors_ep{ep_idx}.mp4")
        render_factor_video(frames, factors, output_path, chunk_size=chunk_size)
        print(f"Saved: {output_path}")
