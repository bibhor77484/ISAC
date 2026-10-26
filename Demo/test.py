import numpy as np
import matplotlib.pyplot as plt

plt.ion()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1
sc1 = ax1.scatter([], [])
ax1.set(xlim=(-10, 10), ylim=(-10, 10),
        title="Ground Truth", xlabel="X (m)", ylabel="Y (m)")
ax1.grid(True)

# Plot 2
sc2 = ax2.scatter([], [])
ax2.set(xlim=(-10, 10), ylim=(-10, 10),
        title="Estimated Positions", xlabel="X (m)", ylabel="Y (m)")
ax2.grid(True)

# Plot 3
sc3 = ax3.scatter([], [])
sc4 = ax3.scatter([], [], color='red')
ax3.set(xlim=(-10, 10), ylim=(-10, 10),
        title="Tracked Positions", xlabel="X (m)", ylabel="Y (m)")
ax3.grid(True)

plt.tight_layout()
plt.show()

# Fixed point at (0, 0)
sc4 = ax3.scatter([0], [0], color='red', label='Base Station')

ax3.annotate(
    'Base Station',
    xy=(0, 0),
    xytext=(5, 5),
    textcoords='offset points',
    fontsize=9
)

for i in range(100):
    # Replace with your actual data
    x1, y1 = np.random.uniform(-10, 10, 10), np.random.uniform(-10, 10, 10)
    x2, y2 = np.random.uniform(-10, 10, 5), np.random.uniform(-10, 10, 5)
    x3, y3 = np.random.uniform(-10, 10, 3), np.random.uniform(-10, 10, 3)

    sc1.set_offsets(np.column_stack((x1, y1)))
    sc2.set_offsets(np.column_stack((x2, y2)))
    sc3.set_offsets(np.column_stack((x3, y3)))

    if i%2 == 0:
        sc4.set_offsets(np.column_stack((0, 0)))
#     else:
#         sc4.set_offsets(np.column_stack(([], [])))

    fig.canvas.draw_idle()
    plt.pause(0.1)

plt.ioff()
plt.show()