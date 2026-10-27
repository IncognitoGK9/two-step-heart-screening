"""Paper-only figures: two-step flow diagram and the confusion/reliability composite (built from results figures)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
FIG = Path(__file__).resolve().parents[2] / "figures"
OUT = Path(__file__).resolve().parent
plt.rcParams["font.family"] = "DejaVu Sans"

def box(ax, x, y, w, h, text, fc, ec="#333", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))

fig, ax = plt.subplots(figsize=(7.8, 3.5)); ax.set_xlim(0, 15.4); ax.set_ylim(0, 6); ax.axis("off")
GREEN, RED, AMB, BLUE, GREY = "#DDEFE0", "#F6D9D5", "#FBEFC9", "#DCE8F7", "#F1F1EF"
box(ax, 0.1, 2.4, 1.9, 1.2, "Patient at a\nbasic clinic", GREY, fs=7, bold=True)
box(ax, 2.5, 1.9, 3.6, 2.2, "STEP 1\nAge, sex, chest pain,\nresting BP, fasting sugar\n(no cholesterol)", BLUE, fs=7)
arrow(ax, 2.0, 3.0, 2.5, 3.0)
box(ax, 6.9, 4.55, 2.4, 1.05, "Low risk:\nsend home", GREEN, fs=7)
box(ax, 6.9, 2.5, 2.4, 1.0, "Not sure: tests,\nnormal priority", AMB, fs=7)
box(ax, 6.9, 0.4, 2.4, 1.05, "High risk: tests,\nurgent", RED, fs=7)
for yy in (5.07, 3.0, 0.92):
    arrow(ax, 6.1, 3.0, 6.9, yy)
box(ax, 10.1, 1.3, 3.0, 3.4, "STEP 2\nAll 11 inputs:\nresting ECG, max HR,\nexercise angina,\nST depression, ST slope\n(+ cholesterol,\nif measured)", BLUE, fs=7)
arrow(ax, 9.3, 3.0, 10.1, 3.0); arrow(ax, 9.3, 0.92, 10.1, 2.0)
box(ax, 13.4, 4.2, 1.9, 1.0, "Low risk:\nhome", GREEN, fs=7)
box(ax, 13.4, 2.5, 1.9, 1.0, "Still not sure:\nspecialist", AMB, fs=7)
box(ax, 13.4, 0.8, 1.9, 1.0, "High risk:\nspecialist", RED, fs=7)
for yy in (4.7, 3.0, 1.3):
    arrow(ax, 13.1, 2.95, 13.4, yy)
ax.text(6.9, 5.85, "Dangerous error: a sick patient sent home here", fontsize=7, color="#8A2B1E", ha="left", style="italic")
fig.savefig(OUT / "fig_flow.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1, 1]})
for a, name in zip(axs, ("fig_confusion.png", "fig_reliability.png")):
    a.imshow(mpimg.imread(FIG / name)); a.axis("off")
axs[0].set_title("(a)", fontsize=9, loc="left"); axs[1].set_title("(b)", fontsize=9, loc="left")
fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.93, bottom=0.01)
fig.savefig(OUT / "fig_cm_rel.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("figures ok")
