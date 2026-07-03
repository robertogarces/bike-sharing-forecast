import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def plot_cyclic_encoding() -> HTML:
    """
    Animated illustration of cyclic (sin/cos) encoding for hour-of-day.
    Each frame shows one hour moving around a unit circle, with its
    corresponding sin and cos values. Intended for use in Jupyter notebooks.

    Returns
    -------
    HTML
        A jshtml animation renderable inline in a Jupyter notebook.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    ax.axis("off")

    # Static circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#B4B2A9", linewidth=1.5)

    # All hour dots
    for h in range(24):
        angle = 2 * np.pi * h / 24 - np.pi / 2
        ax.scatter(np.cos(angle), np.sin(angle), color="#B4B2A9", s=30, zorder=3)
        if h % 6 == 0:
            ax.text(
                np.cos(angle) * 1.2,
                np.sin(angle) * 1.2,
                str(h),
                ha="center",
                va="center",
                fontsize=9,
                color="#5F5E5A",
            )

    # Animated elements
    (dot,) = ax.plot([], [], "o", color="#378ADD", markersize=12, zorder=5)
    label = ax.text(0, -1.4, "", ha="center", fontsize=11, color="#378ADD")
    (trail,) = ax.plot([], [], "-", color="#378ADD", alpha=0.3, linewidth=1.5)

    trail_x, trail_y = [], []

    def animate(h):
        angle = 2 * np.pi * h / 24 - np.pi / 2
        x, y = np.cos(angle), np.sin(angle)
        dot.set_data([x], [y])
        label.set_text(
            f"hr = {h}   →   sin={np.sin(2 * np.pi * h / 24):.2f}, cos={np.cos(2 * np.pi * h / 24):.2f}"
        )
        trail_x.append(x)
        trail_y.append(y)
        trail.set_data(trail_x, trail_y)
        return dot, label, trail

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.6, 1.5)
    ax.set_title("Each hour maps to a unique point on the circle", fontsize=11)

    ani = animation.FuncAnimation(fig, animate, frames=24, interval=300, blit=True)
    plt.close()
    return HTML(ani.to_jshtml())
