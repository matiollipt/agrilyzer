# Write style files

from pathlib import Path

def make_styles(styles, output_dir):
    for name, style in styles.items():
        with open(output_dir / name, "w") as f:
            f.write(style.strip() + "\n")
            

           
# Directory where style files should be placed (aligned with tests)
output_dir = Path(__file__).parent / "mplstyles"
output_dir.mkdir(parents=True, exist_ok=True)

# Corrected style contents using double quotes for hex to avoid comment parsing
styles = {
    "agrilizer_precipitation.mplstyle": """
# === agrilizer: Precipitation & Humidity ===
axes.facecolor: white
figure.facecolor: white
axes.edgecolor: "#2e8b57"
axes.prop_cycle: cycler('color', ["#7FDBFF", "#2E8B57", "#87CEEB", "#006994"])
grid.color: "#87CEEB"
grid.linestyle: --
grid.alpha: 0.4
xtick.color: "#006994"
ytick.color: "#006994"
axes.labelcolor: "#006994"
text.color: "#006994"
font.size: 12
legend.frameon: False
date.autoformatter.day: "%b %-d"
""",
    "agrilizer_dewpoint.mplstyle": """
# === agrilizer: Dew Point ===
axes.facecolor: "#F0FFF0"
figure.facecolor: "#F0FFF0"
axes.edgecolor: "#228B22"
axes.prop_cycle: cycler('color', ["#20B2AA", "#3CB371", "#2E8B57"])
grid.color: "#3CB371"
grid.linestyle: :
grid.alpha: 0.5
xtick.color: "#2E8B57"
ytick.color: "#2E8B57"
axes.labelcolor: "#2E8B57"
text.color: "#2E8B57"
font.size: 12
legend.frameon: False
date.autoformatter.day: "%b %-d"
""",
    "agrilizer_temperature.mplstyle": """
# === agrilizer: Temperature & Heat ===
axes.facecolor: "#FFF5E6"
figure.facecolor: "#FFF5E6"
axes.edgecolor: "#D62728"
axes.prop_cycle: cycler('color', ["#FF7F0E", "#D62728", "#FFBB78", "#E6550D"])
grid.color: "#FFBB78"
grid.linestyle: --
grid.alpha: 0.3
xtick.color: "#D62728"
ytick.color: "#D62728"
axes.labelcolor: "#D62728"
text.color: "#D62728"
font.size: 12
legend.frameon: False
date.autoformatter.day: "%b %-d"
""",
    "agrilizer_clouds.mplstyle": """
# === agrilizer: Clouds ===
axes.facecolor: white
figure.facecolor: white
axes.edgecolor: none
axes.prop_cycle: cycler('color', ["#FFFFFF", "#DDDDDD", "#BBBBBB"])
grid.color: none
xtick.color: none
ytick.color: none
axes.labelcolor: none
text.color: "#555555"
font.size: 12
legend.frameon: False
axes.spines.top: False
axes.spines.right: False
axes.spines.left: False
axes.spines.bottom: False
""",
}

make_styles(styles, output_dir)