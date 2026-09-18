"""
Author: Stefan Stiller
Affiliation: Leibniz-Centre for Agricultural Landscape Research (ZALF) e.V.
ORCID: https://orcid.org/0009-0004-7468-1678
License: GPL-3.0

Two-question figure split for Experiment 3 LLM-judge results.

Figure 1 (Q1 — structure): paired bars (per judge) for unstructured all vs
  MCP all, with a slope overlay for the mean across judges.

Figure 2 (Q2 — modality dose under MCP): singles → tuples → MCP all,
  with overall + per-criterion panels. No unstructured or image-only baselines.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

AVG_LINE_COLOR = "#4a5c6e"
JUDGE_LINE_COLORS = {
  "GPT-5.2": "#6b8cae",
  "Claude Sonnet 4.5": "#c49a6c",
  "Gemini 3": "#7a9e8e",
}
JUDGE_LINE_WIDTH = 1.6
AVG_LINE_WIDTH_SCALE = 2.0
BASELINE_FADE = 0.62
BASELINE_GRAY_MIX = "#d4d4d4"

# Marker shape encodes modality / resource count (as in earlier figures)
MODALITY_MARKERS = {
  1: "s",  # square — one resource
  2: "^",  # triangle — two resources
  3: "D",  # diamond — three resources
}
MODALITY_MARKER_LABELS = {
  1: "1 modality",
  2: "2 modalities",
  3: "3 modalities",
}
# Visual size multipliers (triangles read smaller at equal area)
MODALITY_MARKER_SIZE_SCALE = {
  1: 1.0,
  2: 1.4,
  3: 1.05,
}

OUTPUT_DIR = Path(__file__).resolve().parent / "Evaluation_Reports"

CSV_NAME = "2025_LLM-MCP-Study_Results_LLM-Judges - Refined Study - Adde.csv"

OVERALL_CRITERIA = [
  "Claim Factuality",
  "Relevance & Correct Referencing",
  "Reasoning Quality",
  "Cross-source Integration",
]

PER_CRITERION_FIGURES = [
  ("Claim Factuality", "Claim factuality", "Claim_Factuality"),
  (
    "Relevance & Correct Referencing",
    "Relevance & correct referencing",
    "Relevance_Correct_Referencing",
  ),
  ("Reasoning Quality", "Reasoning quality", "Reasoning_Quality"),
  (
    "Cross-source Integration",
    "Cross-source integration",
    "Cross_Source_Integration",
  ),
]

JUDGE_ORDER = [
  "GPT-5.2",
  "Claude Sonnet 4.5",
  "Gemini 3",
]

# Display labels after CSV rename
LABEL_UNSTRUCTURED_ALL = "Baseline \nunstructured \nall"
LABEL_IMAGE_ONLY = "Baseline \nimage \nonly"
LABEL_MCP_ALL = "All \nresources, \nstructured"

FIG1_ORDER = [LABEL_UNSTRUCTURED_ALL, LABEL_MCP_ALL]
FIG1_XTICK_LABELS = [
  "Baseline\nunstructured access\nall resources",
  "MCP\nstructured access\nall resources",
]

FIG2_ORDER = [
  "Yield \nprediction",
  "Literature",
  "Soil \ndata",
  "Soil data & \nliterature",
  "Soil data & \nyield \nprediction",
  "Literature & \nyield \nprediction",
  LABEL_MCP_ALL,
]


def output_path(filename):
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  return str(OUTPUT_DIR / filename)


def normalize_label(value):
  return " ".join(str(value).split())


def _hex_to_rgb(hex_color):
  hex_color = hex_color.lstrip("#")
  return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
  return "#{:02x}{:02x}{:02x}".format(
    *[max(0, min(255, int(round(c)))) for c in rgb]
  )


def blend_hex(color, mix_color, amount):
  c1 = _hex_to_rgb(color)
  c2 = _hex_to_rgb(mix_color)
  return _rgb_to_hex(
    tuple((1.0 - amount) * a + amount * b for a, b in zip(c1, c2))
  )


def fade_baseline_color(color, amount=BASELINE_FADE):
  return blend_hex(color, BASELINE_GRAY_MIX, amount)


def format_xtick_label(label, max_lines=2):
  parts = [part.strip() for part in str(label).split("\n") if part.strip()]
  if len(parts) <= max_lines:
    return "\n".join(parts)
  if max_lines == 1:
    return " ".join(parts)
  merged_head = " ".join(parts[: len(parts) - (max_lines - 1)])
  tail = " ".join(parts[len(parts) - (max_lines - 1) :])
  return f"{merged_head}\n{tail}"


PRESENTATION_NULL_LABELS = {
  LABEL_IMAGE_ONLY: {
    "top": "Baseline \nimage \nonly",
    "bottom": "Baseline \nimage only",
  },
}


def format_fig2_xtick_labels(labels, panel_tier="top", max_lines=3):
  formatted = []
  for label in labels:
    if label in PRESENTATION_NULL_LABELS:
      formatted.append(PRESENTATION_NULL_LABELS[label][panel_tier])
    else:
      formatted.append(format_xtick_label(label, max_lines=max_lines))
  return formatted


def load_experiment3_dataframe():
  path = OUTPUT_DIR / CSV_NAME
  df = pd.read_csv(path)
  label_row = df.iloc[0]

  rename_map = {}
  for col in df.columns:
    if col in ["Experiment", "Judge"]:
      continue
    if col in label_row.index:
      rename_map[col] = label_row[col]
  df = df.rename(columns=rename_map)

  df = df.drop(index=0).reset_index(drop=True)
  df["Experiment"] = df["Experiment"].ffill()
  df["Experiment"] = (
    df["Experiment"].str.replace(r"\s+", " ", regex=True).str.strip()
  )
  df["Judge"] = df["Judge"].replace(
    {
      "GPT-5.5 Instant": "GPT-5.2",
      "ChatGPT 5.2 Plus Instant": "GPT-5.2",
      "Claude Sonnet 4.6 Low": "Claude Sonnet 4.5",
      "Gemini 3.6": "Gemini 3",
    }
  )
  df["Experiment"] = df["Experiment"].replace(
    {
      "Baseline": LABEL_IMAGE_ONLY,
      "Baseline+": LABEL_UNSTRUCTURED_ALL,
      "Yield prediction tool": "Yield \nprediction",
      "Scientific literature": "Literature",
      "Soil experimental data": "Soil \ndata",
      "CSV & PDF": "Soil data & \nliterature",
      "CSV & Yield": "Soil data & \nyield \nprediction",
      "PDF & Yield": "Literature & \nyield \nprediction",
      "Combined": LABEL_MCP_ALL,
    }
  )

  non_criterion_cols = {"Experiment", "Judge", "mean", "SEM"}
  score_cols = [c for c in df.columns if c not in non_criterion_cols]
  df = df.rename(columns={col: normalize_label(col) for col in score_cols})
  score_cols = [normalize_label(col) for col in score_cols]
  df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")

  missing = [c for c in OVERALL_CRITERIA if c not in score_cols]
  if missing:
    raise ValueError(f"Missing criteria {missing}. Found: {score_cols}")

  return df, score_cols


def build_judge_series(long_df, experiment_order, judge_order=None, with_sem=False):
  grouped = long_df.groupby(["Experiment", "Judge"])["Score"]
  by_judge = grouped.mean().unstack("Judge")
  by_judge = by_judge.reindex(experiment_order)
  if judge_order is None:
    judge_order = list(by_judge.columns)
  else:
    judge_order = [j for j in judge_order if j in by_judge.columns]
    by_judge = by_judge.reindex(columns=judge_order)
  average = by_judge.mean(axis=1)

  if not with_sem:
    return by_judge, average

  n_judges = by_judge.count(axis=1).clip(lower=1)
  average_sem = (by_judge.std(axis=1, ddof=1) / np.sqrt(n_judges)).fillna(0.0)
  average_sem = average_sem.reindex(experiment_order).fillna(0.0)
  return by_judge, average, average_sem


def null_experiment_indices(exp_labels):
  """Fade Baseline image only (null / 0 MCP sources)."""
  return [
    i
    for i, exp in enumerate(exp_labels)
    if normalize_label(exp).lower() == "baseline image only"
  ]


def modality_count(exp_label):
  """Number of MCP information modalities/resources for a condition."""
  text = normalize_label(exp_label).lower()
  if "all resources" in text:
    return 3
  # Tuple labels use "&" (embedded newlines preserved in raw label)
  if "&" in str(exp_label):
    return 2
  return 1


def modality_markers_for_order(experiment_order):
  return [MODALITY_MARKERS[modality_count(exp)] for exp in experiment_order]


def draw_multi_condition_lines(
  ax,
  by_judge,
  average,
  experiment_order,
  title,
  annotate_targets=None,
  panel_letter=None,
  show_legend=True,
  legend_loc="upper left",
  show_xlabels=True,
  show_ylabel=True,
  ylabel="Score",
  average_sem=None,
  show_errorbars=False,
  fade_null=True,
  show_modality_legend=True,
  xtick_labels=None,
  x_tick_rotation=0,
  x_tick_max_lines=2,
  x_tick_pad=None,
  x_tick_va=None,
  x_tick_label_tier=None,
  title_fontsize=18,
  label_fontsize=14,
  tick_fontsize=12,
  legend_fontsize=13,
  annotation_fontsize=14,
  panel_letter_fontsize=14,
  marker_size=110,
  linewidth=JUDGE_LINE_WIDTH,
  capsize=3.5,
):
  """Judge slopes/lines + thicker mean; optional SEM on mean only.

  Marker shape encodes modality count: square=1, triangle=2, diamond=3.
  """
  by_judge = by_judge.reindex(experiment_order)
  average = average.reindex(experiment_order)
  x_base = np.arange(len(experiment_order), dtype=float)
  exp_labels = list(experiment_order)
  avg_values = average.values
  avg_linewidth = linewidth * AVG_LINE_WIDTH_SCALE
  # Larger plot markers than earlier circle-only figures
  ms_judge = max(4, marker_size / 12) * (2.0 / 3.0)
  ms_avg = max(5, marker_size / 10) * (2.0 / 3.0)
  markers = modality_markers_for_order(exp_labels)
  modality_counts = [modality_count(exp) for exp in exp_labels]

  null_idx = null_experiment_indices(exp_labels) if fade_null else []
  mcp_start = null_idx[-1] if null_idx else 0

  if show_errorbars:
    if average_sem is None:
      raise ValueError("show_errorbars=True requires average_sem")
    average_sem = average_sem.reindex(experiment_order).fillna(0.0)

  def marker_area(base_markersize, n_modalities):
    scale = MODALITY_MARKER_SIZE_SCALE.get(n_modalities, 1.0)
    return (base_markersize * 1.35 * scale) ** 2

  def plot_series(values, color, series_linewidth, markersize, zorder, label, yerr=None):
    faded = fade_baseline_color(color)
    values = np.asarray(values, dtype=float)

    # Connecting line (no markers — shapes set per point below)
    if null_idx and mcp_start + 1 < len(values):
      i0 = null_idx[0]
      ax.plot(
        x_base[i0 : mcp_start + 2],
        values[i0 : mcp_start + 2],
        linewidth=series_linewidth,
        color=faded,
        alpha=0.85,
        zorder=zorder,
        solid_capstyle="round",
      )
      (line,) = ax.plot(
        x_base[mcp_start + 1 :],
        values[mcp_start + 1 :],
        linewidth=series_linewidth,
        color=color,
        zorder=zorder + 1,
        label=label,
      )
    else:
      (line,) = ax.plot(
        x_base,
        values,
        linewidth=series_linewidth,
        color=color,
        zorder=zorder + 1,
        label=label,
        alpha=0.85 if (null_idx and len(values) == 1) else 1.0,
      )

    for i, (xi, yi) in enumerate(zip(x_base, values)):
      if np.isnan(yi):
        continue
      is_null = i in null_idx
      pt_color = faded if is_null else color
      ax.scatter(
        [xi],
        [yi],
        marker=markers[i],
        s=marker_area(markersize, modality_counts[i]),
        color=pt_color,
        edgecolors="white",
        linewidths=0.7,
        zorder=zorder + 1.5,
        alpha=0.85 if is_null else 1.0,
      )

    if yerr is not None:
      yerr = np.asarray(yerr, dtype=float)
      for i in range(len(values)):
        col = faded if i in null_idx else color
        ax.errorbar(
          [x_base[i]],
          [values[i]],
          yerr=[yerr[i]],
          fmt="none",
          ecolor=col,
          elinewidth=max(1.0, series_linewidth * 0.7),
          capsize=capsize,
          capthick=max(1.0, series_linewidth * 0.7),
          alpha=0.8,
          zorder=zorder,
        )
    return line

  legend_handles = []
  legend_labels = []
  for judge in by_judge.columns:
    color = JUDGE_LINE_COLORS.get(judge, "#a0a0a0")
    legend_handles.append(
      plot_series(
        by_judge[judge].values,
        color,
        linewidth,
        ms_judge,
        zorder=2,
        label=judge,
      )
    )
    legend_labels.append(judge)

  legend_handles.append(
    plot_series(
      avg_values,
      AVG_LINE_COLOR,
      avg_linewidth,
      ms_avg,
      zorder=3,
      label="Mean across judges",
      yerr=average_sem.values if show_errorbars else None,
    )
  )
  legend_labels.append("Mean across judges")

  if show_legend and legend_handles:
    if show_modality_legend:
      # Blank separator then modality marker keys in the same legend box
      legend_handles.append(Line2D([0], [0], linestyle="None", marker="", label=""))
      legend_labels.append("")
      for n in (1, 2, 3):
        scale = MODALITY_MARKER_SIZE_SCALE.get(n, 1.0)
        legend_handles.append(
          Line2D(
            [0],
            [0],
            marker=MODALITY_MARKERS[n],
            color="none",
            markerfacecolor="#6e6e6e",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=max(8, (ms_avg + 2) * scale),
            linestyle="None",
            label=MODALITY_MARKER_LABELS[n],
          )
        )
        legend_labels.append(MODALITY_MARKER_LABELS[n])

    ax.legend(
      legend_handles,
      legend_labels,
      fontsize=legend_fontsize,
      frameon=True,
      loc=legend_loc,
    )

  if annotate_targets:
    for target_label in annotate_targets:
      target_clean = normalize_label(target_label).lower()
      for i, exp in enumerate(exp_labels):
        exp_clean = normalize_label(exp).lower()
        if target_clean in exp_clean or target_clean == exp_clean:
          ax.annotate(
            f"{avg_values[i]:.2f}",
            xy=(x_base[i], avg_values[i]),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=annotation_fontsize,
            fontweight="bold",
            color=AVG_LINE_COLOR,
          )
          break

  ax.set_xticks(x_base)
  if show_xlabels:
    if xtick_labels is not None:
      labels = xtick_labels
    elif x_tick_label_tier in {"top", "bottom"}:
      labels = format_fig2_xtick_labels(
        exp_labels, panel_tier=x_tick_label_tier, max_lines=x_tick_max_lines
      )
    else:
      labels = [
        format_xtick_label(label, max_lines=x_tick_max_lines)
        for label in exp_labels
      ]
    if x_tick_va is not None:
      x_ha, x_va = "center", x_tick_va
    elif x_tick_rotation == 0:
      x_ha, x_va = "center", "top"
    else:
      x_ha, x_va = "center", "baseline"
    ax.set_xticklabels(
      labels,
      fontsize=tick_fontsize,
      rotation=x_tick_rotation,
      ha=x_ha,
      va=x_va,
      rotation_mode="anchor",
    )
    tick_pad = x_tick_pad
    if tick_pad is None:
      tick_pad = 8 if x_tick_rotation == 0 else 18
    ax.tick_params(axis="x", pad=tick_pad)
  else:
    ax.set_xticklabels([])

  if show_ylabel:
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
  ax.set_ylim(0, 4.25)
  ax.set_yticks([0, 1, 2, 3, 4])
  ax.tick_params(axis="y", labelsize=tick_fontsize)
  ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
  ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=10)
  if panel_letter:
    ax.text(
      0.01,
      0.98,
      str(panel_letter),
      transform=ax.transAxes,
      ha="left",
      va="top",
      fontsize=panel_letter_fontsize,
      fontweight="bold",
      color="black",
    )


def draw_paired_bar_slope_plot(
  ax,
  by_judge,
  average,
  experiment_order,
  title,
  xtick_labels,
  average_sem=None,
  show_errorbars=True,
  show_legend=True,
  legend_loc="upper left",
  ylabel="Mean Score ± SEM",
  annotate=True,
  title_fontsize=18,
  label_fontsize=14,
  tick_fontsize=12,
  legend_fontsize=13,
  annotation_fontsize=14,
  bar_width=0.22,
  bar_alpha=0.88,
  linewidth=JUDGE_LINE_WIDTH,
  capsize=4.0,
):
  """Paired grouped bars (per judge) + mean slope for two conditions (Figure 1)."""
  if len(experiment_order) != 2:
    raise ValueError("Paired bar+slope plot expects exactly two conditions.")

  by_judge = by_judge.reindex(experiment_order)
  average = average.reindex(experiment_order)
  judges = list(by_judge.columns)
  n_judges = len(judges)
  x_centers = np.array([0.0, 1.0])
  avg_values = average.values
  avg_lw = linewidth * AVG_LINE_WIDTH_SCALE

  if show_errorbars:
    if average_sem is None:
      raise ValueError("show_errorbars=True requires average_sem")
    average_sem = average_sem.reindex(experiment_order).fillna(0.0)

  # Offset judge bars symmetrically around each condition center
  offsets = (np.arange(n_judges) - (n_judges - 1) / 2.0) * bar_width
  legend_handles = []

  for j_idx, judge in enumerate(judges):
    color = JUDGE_LINE_COLORS.get(judge, "#a0a0a0")
    vals = by_judge[judge].values
    bars = ax.bar(
      x_centers + offsets[j_idx],
      vals,
      width=bar_width * 0.92,
      color=color,
      edgecolor="white",
      linewidth=0.8,
      alpha=bar_alpha,
      zorder=2,
      label=judge,
    )
    legend_handles.append(bars)

  (avg_line,) = ax.plot(
    x_centers,
    avg_values,
    linewidth=avg_lw,
    color=AVG_LINE_COLOR,
    marker="o",
    markersize=9,
    markeredgecolor="white",
    markeredgewidth=0.9,
    zorder=4,
    label="Mean across judges",
  )
  legend_handles.append(avg_line)

  if show_errorbars:
    ax.errorbar(
      x_centers,
      avg_values,
      yerr=average_sem.values,
      fmt="none",
      ecolor=AVG_LINE_COLOR,
      elinewidth=max(1.2, avg_lw * 0.7),
      capsize=capsize,
      capthick=max(1.2, avg_lw * 0.7),
      alpha=0.9,
      zorder=4,
    )

  if annotate:
    for i, val in enumerate(avg_values):
      ax.annotate(
        f"{val:.2f}",
        xy=(x_centers[i], val),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=annotation_fontsize,
        fontweight="bold",
        color=AVG_LINE_COLOR,
        zorder=5,
      )

  if show_legend:
    ax.legend(
      legend_handles,
      [h.get_label() for h in legend_handles],
      fontsize=legend_fontsize,
      frameon=True,
      loc=legend_loc,
    )

  ax.set_xticks(x_centers)
  ax.set_xticklabels(xtick_labels, fontsize=tick_fontsize)
  half_span = max(0.55, (n_judges * bar_width) / 2.0 + 0.2)
  ax.set_xlim(x_centers[0] - half_span, x_centers[1] + half_span)
  ax.set_ylabel(ylabel, fontsize=label_fontsize)
  ax.set_ylim(0, 4.25)
  ax.set_yticks([0, 1, 2, 3, 4])
  ax.tick_params(axis="y", labelsize=tick_fontsize)
  ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
  ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=12)


def save_figure1_structure(by_judge, average, average_sem):
  """Q1: unstructured all vs MCP all (paired bars + mean slope)."""
  fig, ax = plt.subplots(figsize=(8.0, 5.8))
  draw_paired_bar_slope_plot(
    ax,
    by_judge,
    average,
    FIG1_ORDER,
    title="Structured vs. unstructured access",
    xtick_labels=FIG1_XTICK_LABELS,
    average_sem=average_sem,
    show_errorbars=True,
    legend_loc="lower right",
  )
  plt.tight_layout()
  out = output_path("2025_LLM-MCP-Study_Fig1_Structure_PairedBars_SEM.png")
  plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.25)
  plt.close(fig)
  print(f"Figure 1 saved to: {out}")

  # Presentation-sized copy
  fig, ax = plt.subplots(figsize=(12, 9))
  draw_paired_bar_slope_plot(
    ax,
    by_judge,
    average,
    FIG1_ORDER,
    title="Structured vs. unstructured access",
    xtick_labels=FIG1_XTICK_LABELS,
    average_sem=average_sem,
    show_errorbars=True,
    legend_loc="lower right",
    title_fontsize=26,
    label_fontsize=22,
    tick_fontsize=18,
    legend_fontsize=16,
    annotation_fontsize=22,
    bar_width=0.26,
    linewidth=2.4,
    capsize=6,
  )
  plt.tight_layout()
  out_pres = output_path(
    "2025_LLM-MCP-Study_Fig1_Structure_PairedBars_Presentation_SEM.png"
  )
  plt.savefig(out_pres, dpi=300, bbox_inches="tight", pad_inches=0.25)
  plt.close(fig)
  print(f"Figure 1 (presentation) saved to: {out_pres}")


def save_figure2_modality(df, judge_order):
  """Q2: MCP modality ladder with overall + criterion panels."""
  overall_long = df.melt(
    id_vars=["Experiment", "Judge"],
    value_vars=OVERALL_CRITERIA,
    var_name="Criterion",
    value_name="Score",
  )
  by_judge, average, average_sem = build_judge_series(
    overall_long, FIG2_ORDER, judge_order=judge_order, with_sem=True
  )

  annotate_targets = [
    "Soil data & yield prediction",
    "All resources, structured",
  ]

  criterion_series = {}
  for criterion_col, plot_title, file_slug in PER_CRITERION_FIGURES:
    crit_long = df.melt(
      id_vars=["Experiment", "Judge"],
      value_vars=[criterion_col],
      var_name="Criterion",
      value_name="Score",
    )
    bj, avg = build_judge_series(
      crit_long, FIG2_ORDER, judge_order=judge_order
    )
    criterion_series[file_slug] = {
      "by_judge": bj,
      "average": avg,
      "title": plot_title,
    }

  panels = [
    {
      "by_judge": by_judge,
      "average": average,
      "average_sem": average_sem,
      "show_errorbars": True,
      "title": "Mean performance across all criteria",
      "ylabel": "Mean Score ± SEM",
      "annotate_targets": annotate_targets,
      "panel_letter": "a",
    },
    {
      **criterion_series["Claim_Factuality"],
      "panel_letter": "b",
    },
    {
      **criterion_series["Relevance_Correct_Referencing"],
      "panel_letter": "c",
    },
    {
      **criterion_series["Reasoning_Quality"],
      "panel_letter": "d",
    },
    {
      **criterion_series["Cross_Source_Integration"],
      "panel_letter": "e",
    },
  ]

  # Standalone overall ladder
  fig, ax = plt.subplots(figsize=(11, 4.8))
  draw_multi_condition_lines(
    ax,
    by_judge,
    average,
    FIG2_ORDER,
    title="MCP modality ladder: mean across criteria",
    annotate_targets=annotate_targets,
    average_sem=average_sem,
    show_errorbars=True,
    fade_null=False,
    ylabel="Mean Score ± SEM",
    legend_loc="lower right",
    x_tick_max_lines=3,
  )
  plt.tight_layout()
  out_overall = output_path(
    "2025_LLM-MCP-Study_Fig2_ModalityLadder_Overall_SEM.png"
  )
  plt.savefig(out_overall, dpi=300, bbox_inches="tight", pad_inches=0.3)
  plt.close(fig)
  print(f"Figure 2 overall saved to: {out_overall}")

  # Combined top-large + 2x2 criteria
  for style_name, style in [
    (
      "publication",
      {
        "figsize": (14, 10),
        "outer_hspace": 0.35,
        "bottom_hspace": 0.22,
        "bottom_wspace": 0.18,
        "figure_bottom": 0.10,
        "dpi": 300,
        "pad_inches": 0.05,
        "top": {
          "title_fontsize": 14,
          "label_fontsize": 12,
          "tick_fontsize": 10,
          "legend_fontsize": 9,
          "annotation_fontsize": 11,
          "panel_letter_fontsize": 14,
          "marker_size": 100,
          "linewidth": 1.6,
          "title_pad": 10,
          "x_tick_max_lines": 3,
          "x_tick_label_tier": "top",
        },
        "bottom": {
          "title_fontsize": 12,
          "label_fontsize": 10,
          "tick_fontsize": 8,
          "panel_letter_fontsize": 14,
          "marker_size": 75,
          "linewidth": 1.4,
          "title_pad": 10,
          "x_tick_rotation": 60,
          "x_tick_max_lines": 2,
          "x_tick_pad": 70,
          "x_tick_va": "top",
          "x_tick_label_tier": "bottom",
        },
      },
    ),
    (
      "presentation",
      {
        "figsize": (24, 17),
        "outer_hspace": 0.35,
        "bottom_hspace": 0.28,
        "bottom_wspace": 0.18,
        "figure_bottom": 0.14,
        "dpi": 300,
        "pad_inches": 0.05,
        "top": {
          "title_fontsize": 28,
          "label_fontsize": 22,
          "tick_fontsize": 18,
          "legend_fontsize": 16,
          "annotation_fontsize": 22,
          "panel_letter_fontsize": 28,
          "marker_size": 160,
          "linewidth": 2.3,
          "title_pad": 14,
          "x_tick_max_lines": 3,
          "x_tick_label_tier": "top",
        },
        "bottom": {
          "title_fontsize": 22,
          "label_fontsize": 20,
          "tick_fontsize": 16,
          "panel_letter_fontsize": 26,
          "marker_size": 130,
          "linewidth": 2.1,
          "title_pad": 14,
          "x_tick_rotation": 60,
          "x_tick_max_lines": 2,
          "x_tick_pad": 80,
          "x_tick_va": "top",
          "x_tick_label_tier": "bottom",
        },
      },
    ),
  ]:
    top_style = style["top"]
    bottom_style = style["bottom"]
    fig = plt.figure(figsize=style["figsize"])
    gs_outer = fig.add_gridspec(
      2,
      1,
      left=0.06,
      right=0.98,
      top=0.95,
      bottom=style["figure_bottom"],
      height_ratios=[2.1, 2.3],
      hspace=style["outer_hspace"],
    )
    gs_bottom = gs_outer[1].subgridspec(
      2, 2, hspace=style["bottom_hspace"], wspace=style["bottom_wspace"]
    )
    ax_a = fig.add_subplot(gs_outer[0])
    ax_b = fig.add_subplot(gs_bottom[0, 0])
    ax_c = fig.add_subplot(gs_bottom[1, 0])
    ax_d = fig.add_subplot(gs_bottom[0, 1])
    ax_e = fig.add_subplot(gs_bottom[1, 1])

    draw_multi_condition_lines(
      ax_a,
      panels[0]["by_judge"],
      panels[0]["average"],
      FIG2_ORDER,
      panels[0]["title"],
      annotate_targets=panels[0].get("annotate_targets"),
      panel_letter=panels[0].get("panel_letter"),
      show_legend=True,
      legend_loc="lower right",
      ylabel=panels[0]["ylabel"],
      average_sem=panels[0].get("average_sem"),
      show_errorbars=True,
      fade_null=False,
      title_fontsize=top_style["title_fontsize"],
      label_fontsize=top_style["label_fontsize"],
      tick_fontsize=top_style["tick_fontsize"],
      legend_fontsize=top_style["legend_fontsize"],
      annotation_fontsize=top_style["annotation_fontsize"],
      panel_letter_fontsize=top_style["panel_letter_fontsize"],
      marker_size=top_style["marker_size"],
      linewidth=top_style["linewidth"],
      x_tick_max_lines=top_style["x_tick_max_lines"],
      x_tick_label_tier=top_style["x_tick_label_tier"],
    )
    ax_a.set_title(
      panels[0]["title"],
      fontsize=top_style["title_fontsize"],
      fontweight="bold",
      pad=top_style["title_pad"],
    )

    bottom_axes = [
      (ax_b, panels[1], False, True),
      (ax_c, panels[2], True, True),
      (ax_d, panels[3], False, False),
      (ax_e, panels[4], True, False),
    ]
    for ax, panel, show_xlabels, show_ylabel in bottom_axes:
      draw_multi_condition_lines(
        ax,
        panel["by_judge"],
        panel["average"],
        FIG2_ORDER,
        panel["title"],
        panel_letter=panel.get("panel_letter"),
        show_legend=False,
        show_xlabels=show_xlabels,
        show_ylabel=show_ylabel,
        ylabel="Score",
        fade_null=False,
        title_fontsize=bottom_style["title_fontsize"],
        label_fontsize=bottom_style["label_fontsize"],
        tick_fontsize=bottom_style["tick_fontsize"],
        panel_letter_fontsize=bottom_style["panel_letter_fontsize"],
        marker_size=bottom_style["marker_size"],
        linewidth=bottom_style["linewidth"],
        x_tick_rotation=bottom_style.get("x_tick_rotation", 0),
        x_tick_max_lines=bottom_style.get("x_tick_max_lines", 2),
        x_tick_pad=bottom_style.get("x_tick_pad"),
        x_tick_va=bottom_style.get("x_tick_va"),
        x_tick_label_tier=bottom_style.get("x_tick_label_tier"),
      )
      ax.set_title(
        panel["title"],
        fontsize=bottom_style["title_fontsize"],
        fontweight="bold",
        pad=bottom_style["title_pad"],
      )

    suffix = "" if style_name == "publication" else "_Presentation"
    out = output_path(
      f"2025_LLM-MCP-Study_Fig2_ModalityLadder_Combined{suffix}_SEM.png"
    )
    plt.savefig(
      out, dpi=style["dpi"], bbox_inches="tight", pad_inches=style["pad_inches"]
    )
    plt.close(fig)
    print(f"Figure 2 combined ({style_name}) saved to: {out}")


def write_captions():
  cap1 = """Figure 1. Structured vs. unstructured access (same information sources).

Paired bar plot comparing Baseline unstructured all (all sources available without
MCP structure) to All resources, structured (MCP). Colored bars show each LLM judge
(GPT-5.2, Claude Sonnet 4.5, Gemini 3); the thick overlay line is the
mean across judges, with SEM error bars across the three judge-level means (judge
dispersion, not independent experimental replicates). Scores are first averaged
across the four evaluation criteria for each judge. Scores range from 0 to 4."""

  cap2 = """Figure 2. MCP modality ladder: effect of increasing complementary sources.

Conditions progress under MCP from single resources through pairwise tuples to all
resources structured. Baseline conditions (image only; unstructured all) are excluded
so that this figure addresses modality dose under MCP only. Marker shapes encode the
number of modalities: square = 1, triangle = 2, diamond = 3. Panel (a) shows mean
performance across all criteria with judge lines and a thicker mean-across-judges
line (SEM across judges on the mean only). Panels (b–e) show individual criteria
without SEM. Scores range from 0 to 4."""

  p1 = output_path("2025_LLM-MCP-Study_Fig1_Structure_PairedBars_SEM_caption.txt")
  p2 = output_path("2025_LLM-MCP-Study_Fig2_ModalityLadder_SEM_caption.txt")
  with open(p1, "w", encoding="utf-8") as f:
    f.write(cap1)
  with open(p2, "w", encoding="utf-8") as f:
    f.write(cap2)
  print(f"Captions saved to:\n  {p1}\n  {p2}")
  print("\n" + "=" * 80)
  print("FIGURE 1 CAPTION")
  print("=" * 80)
  print(cap1)
  print("\n" + "=" * 80)
  print("FIGURE 2 CAPTION")
  print("=" * 80)
  print(cap2)


def main():
  df, score_cols = load_experiment3_dataframe()
  observed_judges = [j for j in JUDGE_ORDER if j in set(df["Judge"].dropna())]
  if not observed_judges:
    observed_judges = sorted(df["Judge"].dropna().unique().tolist())

  print(f"Loaded criteria: {score_cols}")
  print(f"Judges: {observed_judges}")
  print(f"Output directory: {OUTPUT_DIR}")

  # --- Figure 1 data ---
  overall_long = df.melt(
    id_vars=["Experiment", "Judge"],
    value_vars=OVERALL_CRITERIA,
    var_name="Criterion",
    value_name="Score",
  )
  fig1_by_judge, fig1_avg, fig1_sem = build_judge_series(
    overall_long, FIG1_ORDER, judge_order=observed_judges, with_sem=True
  )
  save_figure1_structure(fig1_by_judge, fig1_avg, fig1_sem)

  # --- Figure 2 data / plots ---
  missing_fig2 = [e for e in FIG2_ORDER if e not in set(df["Experiment"])]
  if missing_fig2:
    raise ValueError(f"Figure 2 conditions missing from data: {missing_fig2}")
  save_figure2_modality(df, observed_judges)

  write_captions()


if __name__ == "__main__":
  main()
