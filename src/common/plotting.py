import matplotlib.pyplot as plt

# Colour-blind-safe palette (Okabe-Ito based)
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
RAG = {"Green": "#009E73", "Amber": "#E69F00", "Red": "#D55E00"}


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
        "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
    })
