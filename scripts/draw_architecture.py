"""Draw the architecture diagram (assets/architecture.png)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 160)
ax.set_ylim(-3, 90)
ax.axis("off")


def box(x, y, w, h, text, fc="#dbeafe", ec="#1e3a8a", ls="-", fs=8.5, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=1.5, ls=ls))
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(x1, y1, x2, y2, label=None, ls="-", color="#374151"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color, ls=ls))
    if label:
        ax.text((x1 + x2) / 2.0, (y1 + y2) / 2.0 + 1.8, label, ha="center", va="bottom", fontsize=7.5, color=color)


ax.text(1, 87, "A. Training, evaluation and benchmarks (Google Colab, Tesla T4)", fontsize=11, fontweight="bold")
box(2, 70, 24, 10, "BDD100K subset\n4,000 train / 800 val / 800 test\n10 classes")
box(32, 70, 24, 10, "YOLO11s fine-tune\n30 epochs, W&B logging")
box(62, 70, 18, 10, "best.pt\ntest mAP50 0.414")
box(88, 70, 22, 10, "ONNX export\n(36 MB, CPU capable)")
box(118, 70, 26, 10, "TensorRT FP16 engine\n2.5 ms vs 9.4 ms (T4 only)")
arrow(26, 75, 32, 75)
arrow(56, 75, 62, 75)
arrow(80, 75, 88, 75)
arrow(110, 75, 118, 75)

box(2, 56, 24, 10, "Dashcam clips\n(Pexels stock footage)")
box(32, 56, 24, 10, "ByteTrack (Ultralytics)\nvehicles + people")
box(62, 56, 18, 10, "Tracked videos\n+ track statistics")
arrow(26, 61, 32, 61)
arrow(56, 61, 62, 61)
arrow(66, 69.6, 48, 66.4, ls="--", color="#1e3a8a")

box(2, 40, 24, 10, "TuSimple\n3,626 frames, split by drive")
box(32, 40, 24, 10, "Classical OpenCV\nvs U-Net (ResNet18)")
box(62, 40, 18, 10, "Lane comparison\nF1 0.48 vs 0.90")
arrow(26, 45, 32, 45)
arrow(56, 45, 62, 45)
box(100, 40, 22, 10, "KITTI subset\n140 frames, true distance")
box(128, 40, 28, 10, "MiDaS_small (512 px)\ndistance error evaluation")
arrow(122, 45, 128, 45)

ax.plot([0, 160], [36, 36], color="#9ca3af", lw=1, ls=":")
ax.text(1, 33.5, "B. Demo pipeline (Docker, ROS2 Humble, CPU only, ONNX Runtime)", fontsize=11, fontweight="bold")
G, GE = "#dcfce7", "#14532d"
box(1, 12, 16, 12, "Recorded video\n(demo clip)", G, GE)
box(24, 12, 18, 12, "camera_node\n4 frames/s\n960 px wide", G, GE)
box(58, 5, 40, 26, "perception_node\n\nONNX Runtime detector (CPU)\n+ IoU tracker\n+ warning rule (corridor)", G, GE)
box(116, 12, 18, 12, "recorder_node\n(decode JPEG,\nwrite mp4)", G, GE)
box(142, 12, 16, 12, "annotated mp4\n+ stats JSON", G, GE)
arrow(17, 18, 24, 18)
arrow(42, 18, 58, 18, "/camera/image_raw")
arrow(98, 18, 116, 18, "/perception/annotated\n/compressed")
arrow(134, 18, 142, 18)
box(1, 25.5, 34, 6, "CARLA simulator: not used\n(4 GB GPU, 6 GB minimum)", "#f3f4f6", "#9ca3af", "--", 7.5, "#6b7280")
arrow(20, 25.5, 30, 24.2, ls="--", color="#9ca3af")
arrow(94, 70, 94, 31.5, ls="--", color="#1e3a8a")
ax.text(96, 55, "models/yolo11s_bdd.onnx", ha="left", fontsize=7.5, color="#1e3a8a")
ax.text(78, 2.2, "also publishes /perception/detections (JSON) and /perception/warning;\n/perception/status (DONE) ends the camera and recorder nodes",
        ha="center", va="center", fontsize=7.5, color="#374151")
fig.savefig("assets/architecture.png", dpi=130, bbox_inches="tight")
print("saved assets/architecture.png")

