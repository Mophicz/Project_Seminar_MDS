import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# Daten laden
df = pd.read_csv("./Evaluation/levenshtein_scores_3.csv", delimiter=";")

# Sicherstellen, dass Levenshtein Score numerisch ist
df["Levenshtein Score"] = pd.to_numeric(df["Levenshtein Score"], errors="coerce")

# ---------------------------------------------------------------------
# Dateinamen bereinigen
df["raw_number"] = df["filename"].str.extract(r'^(\d+)\.')

df["filename"] = (
    df["filename"]
    .str.replace(r'^\d+\.\s*', '', regex=True)
    .str.replace(r'\.txt$', '', regex=True)
    .str.strip()
)

# ---------------------------------------------------------------------
# Mittelwert pro Arztbrief berechnen
group = df.groupby("filename")["Levenshtein Score"].mean().reset_index()

# ---------------------------------------------------------------------
# Median und MAD berechnen
median_ls = group["Levenshtein Score"].median()
mad_ls = np.median(np.abs(group["Levenshtein Score"] - median_ls))

print("Median:", median_ls)
print("MAD:", mad_ls)

# 1. Setup Figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

# 2. Colors
bar_color = "#4A90E2"       # Blue for bars
median_color = "#D32F2F"    # Red for line/text
range_fill_color = "#FFCDD2" # Light Red for shading

# 3. Define the "Data Area"
# We want the lines to span only the width of the bars, not the whole plot.
# Bars are at indices 0, 1, ... len(group)-1.
# We'll start slightly before the first bar (-0.5) and end after the last (len(group)-0.5).
line_start = -0.6
line_end = len(group) - 0.4

# 4. Plot Bars
x_pos = np.arange(len(group))
ax.bar(x_pos, group["Levenshtein Score"], 
       color=bar_color, width=0.6, label="Score", zorder=3)

# 5. Bounded Visuals (Stop before the text!)

# Shading (MAD) - Using fill_between to control the X range
ax.fill_between([line_start, line_end], 
                median_ls - mad_ls, 
                median_ls + mad_ls,
                color=range_fill_color, alpha=0.3, zorder=1, linewidth=0,
                label="Median ± MAD Range")

# Median Line - Using hlines to control start/end points
ax.hlines(median_ls, line_start, line_end, 
          colors=median_color, linewidth=2, linestyles="-", zorder=2,
          label="Median")

# 6. Annotations (Placed safely to the right)
text_x = line_end + 0.2  # Small gap between line end and text

# Median Label
ax.text(text_x, median_ls, f'Median: {median_ls:.1f}', 
        color=median_color, va='center', fontweight='bold', fontsize=10)

# MAD Bounds
ax.text(text_x, median_ls + mad_ls, f'+MAD: {median_ls + mad_ls:.1f}', 
        color=median_color, va='center', fontsize=9, alpha=0.7)
ax.text(text_x, median_ls - mad_ls, f'-MAD: {median_ls - mad_ls:.1f}', 
        color=median_color, va='center', fontsize=9, alpha=0.7)

# 7. Formatting
# Extend x-axis to make room for text
ax.set_xlim(line_start, len(group) + 2.5)

# Ticks & Spines
ax.set_xticks(x_pos)
ax.set_xticklabels(group["filename"], rotation=45, ha="right", fontsize=9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#888888')
ax.spines['bottom'].set_color('#888888')

# Grid
ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.25, zorder=0)

# Labels
ax.set_title("Levenshtein Scores pro Arztbrief", fontweight="bold", fontsize=14, loc='left')
ax.set_xlabel("Arztbrief", fontsize=10)
ax.set_ylabel("Levenshtein Score", fontsize=10)

plt.tight_layout()
plt.show()