"""共享 matplotlib 样式：GitHub Dark 配色 + 中文字体配置。"""

import matplotlib.pyplot as plt

# ── GitHub Dark 配色 ──────────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_FG    = "#e6edf3"
C_GREEN = "#39d353"
C_RED   = "#f85149"
C_BLUE  = "#58a6ff"
C_GOLD  = "#e3b341"
C_MUTED = "#484f58"

COLORS = dict(bg=C_BG, fg=C_FG, green=C_GREEN, red=C_RED,
              blue=C_BLUE, gold=C_GOLD, muted=C_MUTED)


def setup_matplotlib():
    """设置中文字体和负号显示。"""
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "STHeiti", "Microsoft YaHei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def style_ax(ax):
    """对单个 Axes 应用 GitHub Dark 主题。"""
    ax.set_facecolor(C_BG)
    ax.tick_params(colors=C_FG, labelsize=8)
    ax.spines[:].set_color(C_MUTED)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.yaxis.label.set_color(C_FG)
    ax.grid(color=C_MUTED, linewidth=0.3, alpha=0.5)
