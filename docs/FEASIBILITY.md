# Feasibility spikes (Phase 0, 7 Nov 2026)

| Item | Result | Decision |
|---|---|---|
| Laptop GPU | RTX 3050 laptop, 4 GB VRAM, 30 W cap | Inference only; all training on Colab T4 |
| Free disk | about 45 GB | Datasets live on Colab/Drive, not on the laptop |
| WSL2 / Docker | Docker Desktop works; only the docker-desktop WSL distro | ROS2 runs in Docker |
| ROS2 | `ros:humble` container ran `ros2 --help` successfully | Use ROS2 Humble in Docker (Tier 1) |
| CARLA | NOT installed. Official docs ask for a 6 GB GPU minimum (8 GB recommended) and 20+ GB disk; this laptop has 4 GB VRAM and limited disk | Not run. Frames come from recorded driving data instead. Documented honestly |
| TensorRT | Not tested yet | Planned on Colab T4 (Phase 7) |
| Accounts | Kaggle: yes. Weights & Biases: yes | Keys stored only in Colab Secrets |
