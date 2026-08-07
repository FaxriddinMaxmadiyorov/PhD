import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12
})

fig, ax = plt.subplots(figsize=(10, 2.8))

ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")


def box(x, y, w, h, text):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
        )
    )

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
    )


# ----------------------------
# Blocks
# ----------------------------

box(0.2, 1.8, 2.3, 0.7,
    "Panxromatik")

box(0.2, 0.5, 2.3, 0.7,
    "Multispektral")

box(4.0, 1.15, 2.2, 0.7,
    "Birlashtirish")

box(7.3, 1.15, 2.2, 0.7,
    "Baholash\nERGAS, SAM, Q4")

# ----------------------------
# Connections
# ----------------------------

arrow = dict(
    arrowstyle="->",
    lw=1.2,
    color="black",
    shrinkA=0,
    shrinkB=0
)

# Pan -> Fusion
ax.annotate(
    "",
    xy=(4.0, 1.5),
    xytext=(2.5, 2.15),
    arrowprops=arrow,
)

# MS -> Fusion
ax.annotate(
    "",
    xy=(4.0, 1.5),
    xytext=(2.5, 0.85),
    arrowprops=arrow,
)

# Fusion -> Quality
ax.annotate(
    "",
    xy=(7.3, 1.5),
    xytext=(6.2, 1.5),
    arrowprops=arrow,
)

plt.tight_layout()

plt.savefig(
    "fusion_pipeline_minimal.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()