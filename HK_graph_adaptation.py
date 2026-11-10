import numpy as np
import matplotlib.pyplot as plt
from ant_custom_env_hk_cpg import AntPositionHKEnv

# -----------------------------
# Simulation Parameters
# -----------------------------
STEPS = 6000           # total simulation steps (~20-30 min depending on frame_skip)
LOG_INTERVAL = 10      # log C and h every N steps

# -----------------------------
# Initialize Environment
# -----------------------------
env = AntPositionHKEnv(render_mode=None)  # No GUI for faster logging
obs, _ = env.reset()

# -----------------------------
# Data Storage
# -----------------------------
C_history = []
h_history = []
time_steps = []

# -----------------------------
# Run Simulation
# -----------------------------
for step in range(STEPS):
    obs, _, _, _, _ = env.step(None)  # HK + gait automatically applied

    # Log every LOG_INTERVAL steps
    if step % LOG_INTERVAL == 0:
        C_history.append(env.C.copy())
        h_history.append(env.h.copy())
        time_steps.append(step)

# Convert to numpy arrays
C_history = np.array(C_history)  # shape: [num_logged_steps, 8, 8]
h_history = np.array(h_history)  # shape: [num_logged_steps, 8]

# -----------------------------
# Plot HK Matrix (C) Heatmaps Over Time
# -----------------------------
num_snapshots = 5  # show 5 snapshots across simulation
snapshots = np.linspace(0, C_history.shape[0]-1, num_snapshots, dtype=int)

plt.figure(figsize=(15, 3))
for i, idx in enumerate(snapshots):
    plt.subplot(1, num_snapshots, i+1)
    plt.imshow(C_history[idx], cmap='coolwarm', vmin=-0.05, vmax=0.05)
    plt.title(f"C at step {time_steps[idx]}")
    plt.colorbar(label="C values")
plt.suptitle("Homeokinesis Matrix C Evolution")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot HK Bias Vector (h) Over Time
# -----------------------------
plt.figure(figsize=(10,5))
for joint_idx in range(8):
    plt.plot(time_steps, h_history[:, joint_idx], label=f"Joint {joint_idx+1}")
plt.xlabel("Simulation Step")
plt.ylabel("h value")
plt.title("Homeokinesis Bias Vector h Evolution")
plt.legend()
plt.grid(True)
plt.show()
