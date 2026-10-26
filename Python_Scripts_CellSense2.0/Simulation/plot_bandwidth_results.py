import numpy as np
import matplotlib.pyplot as plt

# ---------------- Data ----------------
tof_diff = [4, 8, 16, 32]  # ToF difference between LoS and object path (ns)
bandwidths = ["Full bandwidth", "Half bandwidth", "Quarter bandwidth"]

aoa_rmse = {  # degrees
    "Full bandwidth":    [3.50, 3.66, 2.98, 1.72],
    "Half bandwidth":    [3.41, 3.79, 3.40, 2.20],
    "Quarter bandwidth": [3.60, 3.19, 3.88, 3.25],
}
tof_rmse = {  # ns
    "Full bandwidth":    [2.68, 3.24, 7.02, 6.10],
    "Half bandwidth":    [2.47, 3.32, 9.05, 8.93],
    "Quarter bandwidth": [5.70, 3.78, 16.3, 14.4],
}
det_prob = {  # %
    "Full bandwidth":    [96, 97, 100, 100],
    "Half bandwidth":    [97, 98, 97, 100],
    "Quarter bandwidth": [96, 95, 93, 100],
}

# ---------------- Style ----------------
colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (CVD-safe)
hatches = ["", "//", ".."]                  # secondary encoding for B/W print

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
})


def grouped_bar(data, ylabel, title, fname, ylim=None, value_fmt="{:.2f}"):
    x = np.arange(len(tof_diff))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 7))

    for i, bw in enumerate(bandwidths):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[bw], width * 0.92, label=bw,
                      color=colors[i], hatch=hatches[i],
                      edgecolor="white", linewidth=1.2, zorder=3)
        ax.bar_label(bars, labels=[value_fmt.format(v) for v in data[bw]],
                     padding=3, fontsize=14, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in tof_diff])
    ax.set_xlabel("ToF difference between LoS and object path (ns)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=55)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, max(max(v) for v in data.values()) * 1.15)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              ncol=3, handlelength=1.5, columnspacing=1.2)

    fig.tight_layout()
    fig.savefig(fname + ".png", dpi=300)
    fig.savefig(fname + ".pdf")
    return fig, ax


grouped_bar(aoa_rmse, "AoA RMSE (degrees)", "AoA RMSE vs ToF difference",
            "bar_aoa_rmse")
grouped_bar(tof_rmse, "ToF RMSE (ns)", "ToF RMSE vs ToF difference",
            "bar_tof_rmse")
grouped_bar(det_prob, "Detection probability (%)",
            "Detection probability vs ToF difference", "bar_detection_prob",
            ylim=(80, 102.5), value_fmt="{:.0f}")

plt.show()
