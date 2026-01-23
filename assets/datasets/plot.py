import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"  #
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid", {"grid.linestyle": "--"})


def plot_convergence_experiment(file_baseline, file_wide_clamp):
    df_base = pd.read_csv(file_baseline)
    df_wide = pd.read_csv(file_wide_clamp)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    window = 5  # 平滑窗口
    color_base = "#2E5077"  # Baseline
    color_wide = "#E94560"  # Wide Clamp

    ax.plot(
        df_base["Step"],
        df_base["arm_std"].rolling(window).mean(),
        label="Baseline (Standard Reward)",
        color=color_base,
        lw=2,
        alpha=0.8,
    )

    ax.plot(
        df_wide["Step"],
        df_wide["arm_std"].rolling(window).mean(),
        label="Wide Clamp Reward (Proposed)",
        color=color_wide,
        lw=2.5,
    )

    # 5. 精细化标注
    ax.set_xlabel("Training Steps", fontsize=14, labelpad=10)
    ax.set_ylabel("Action Std ($\sigma$)", fontsize=14, labelpad=10)
    ax.set_title(
        "Action Convergence Analysis: Overcoming Reward Saturation",
        fontsize=16,
        pad=20,
        fontweight="bold",
    )

    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, loc="upper right", fontsize=12)

    ax.annotate(
        "Faster Convergence",
        xy=(df_wide["Step"].iloc[len(df_wide) // 4], 0.2),
        xytext=(df_wide["Step"].iloc[len(df_wide) // 2], 0.6),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
        fontsize=12,
        color=color_wide,
    )

    # 8. 移除多余边框
    sns.despine()

    plt.tight_layout()
    plt.savefig("convergence_comparison.pdf", bbox_inches="tight")
    plt.show()


plot_convergence_experiment("no_virtual_torque.csv", "virtual_torque.csv")
