# analysis/utils.R - Shared utility functions for inverse planning analysis

library(here)
library(tidyverse)
library(tidyboot)
library(ragg)
library(jsonlite)

# Figure dimension constants for consistent sizing across all outputs
# Use FIG_WIDTH_LARGE for 4+ facets or grids, FIG_WIDTH_STANDARD for 2-3 facets
# Legends on right add to width, so widths are increased accordingly
FIG_WIDTH_LARGE <- 14
FIG_WIDTH_STANDARD <- 12
FIG_HEIGHT_TALL <- 5      # For grid layouts (e.g., desire × model)
FIG_HEIGHT_STANDARD <- 4  # For single-row faceted plots
FIG_HEIGHT_SHORT <- 3.5   # For correlation plots

# Color scheme constants
PLOT_ALPHA <- 0.95

# Intimacy color scale parameters - adjust these to test different ranges
INTIMACY_PALETTE <- "cividis"
INTIMACY_BEGIN <- 0.1
INTIMACY_END <- 0.85
# Intimacy is a purely verbal manipulation: the condition is stored as a slug
# (ascending, formal -> intimate), never a numeric code.
INTIMACY_LEVELS <- c("max_formal", "somewhat_formal", "somewhat_intimate", "max_intimate")

# Generate discrete intimacy colors
INTIMACY_COLORS <- viridisLite::viridis(
  n = length(INTIMACY_LEVELS),
  begin = INTIMACY_BEGIN,
  end = INTIMACY_END,
  option = INTIMACY_PALETTE
)
names(INTIMACY_COLORS) <- as.character(INTIMACY_LEVELS)

# Verbal labels for the intimacy levels (the experiment's relationship
# descriptors; see experiments/_lib/scenario.js). Keyed by the condition slug so
# the scales display readable text in the legend.
INTIMACY_LABELS <- c(
  "max_formal"        = "Maximally formal",
  "somewhat_formal"   = "Somewhat formal",
  "somewhat_intimate" = "Somewhat intimate",
  "max_intimate"      = "Maximally intimate"
)

# Intimacy color scales (discrete) — slug levels mapped to verbal labels. The
# legend title defaults to "Relationship" (the construct the intimacy slugs
# manipulate); pass `name` to override it for a specific plot.
scale_fill_intimacy <- function(name = "Relationship") {
  scale_fill_manual(values = INTIMACY_COLORS, labels = INTIMACY_LABELS, name = name)
}

scale_color_intimacy <- function(name = "Relationship") {
  scale_color_manual(values = INTIMACY_COLORS, labels = INTIMACY_LABELS, name = name)
}

# Motivation color scales (discrete)
MOTIVATION_COLORS <- c("Low" = "#C9A8B0", "High" = "#7A4A5A")

scale_fill_desire <- function() {
  scale_fill_manual(values = MOTIVATION_COLORS)
}

scale_color_desire <- function() {
  scale_color_manual(values = MOTIVATION_COLORS)
}

# Effort color scales (discrete) - sage green, distinct from desire (pink)
# and intimacy (cividis blue→yellow)
EFFORT_LEVELS <- c("Low effort", "High effort")
EFFORT_COLORS <- c("Low effort" = "#B5C9A8", "High effort" = "#4A7A4A")

scale_fill_effort <- function() {
  scale_fill_manual(values = EFFORT_COLORS)
}

scale_color_effort <- function() {
  scale_color_manual(values = EFFORT_COLORS)
}

# Build the effort factor from the raw low/high condition codes.
factor_effort <- function(x) factor(x, levels = c("low", "high"), labels = EFFORT_LEVELS)

# Effort pattern scale (ggpattern): solid vs. striped — the non-color companion
# to the effort color scales above, for bar plots that already use fill for a
# different variable (e.g. relationship).
EFFORT_PATTERNS <- c("Low effort" = "none", "High effort" = "stripe")
scale_pattern_effort <- function(name = "Effort of low-risk share") {
  ggpattern::scale_pattern_manual(values = EFFORT_PATTERNS, breaks = EFFORT_LEVELS, name = name)
}

# Model-comparison panels: fitted-model slugs -> display labels, and the panel
# order (the three ablations plus the human data) shared by every study's
# out-of-sample model-vs-human figure.
MODEL_LABELS <- c(base = "Base", discomfort_only = "Discomfort-only", full = "Full")
PANEL_LEVELS <- c(unname(MODEL_LABELS), "Humans")

# Standard theme setup
setup_analysis <- function() {
  theme_set(
    theme_classic(base_size = 18) +
      theme(
        strip.background = element_blank(),
        text = element_text(family = "Arial Nova"),
        panel.spacing = unit(1, "lines"),
        strip.text = element_text(size = 18),
        legend.key = element_blank()
      )
  )
  set.seed(67)
}

# Bootstrap correlation with 95% CI; used via format_correlation_labels in the
# model-vs-human panels
boot_cor <- function(x, y, n_boot = 1000) {
  complete <- complete.cases(x, y)
  x <- x[complete]
  y <- y[complete]
  n <- length(x)
  if (n < 3)
    return(list(
      r = NA_real_,
      ci_lower = NA_real_,
      ci_upper = NA_real_
    ))
  boot_rs <- replicate(n_boot, {
    idx <- sample(n, replace = TRUE)
    cor(x[idx], y[idx])
  })
  list(
    r = cor(x, y),
    ci_lower = as.numeric(quantile(boot_rs, 0.025, na.rm = TRUE)),
    ci_upper = as.numeric(quantile(boot_rs, 0.975, na.rm = TRUE))
  )
}

# Read the model pipeline's JSON / JSON Lines outputs into a tibble. The model
# code writes JSON (small structured artifacts: fit_results, *_preds_summary) and
# JSON Lines (per-record logs: cv_trial_ll, fit_restarts). Use these to consume
# model predictions in the model-vs-human panels:
#   - cv_trial_ll.jsonl   — per-trial held-out log-likelihood (primary metric),
#     keyed by subject_id (bootstrap the full-vs-ablation difference by participant)
#   - cv_preds_summary.json / preds_summary.json — per-cell model belief update
#     delta_<latent> (+ delta_effort for joint studies) for the model-vs-human
#     correlation
read_model_json <- function(path) {
  as_tibble(jsonlite::fromJSON(path, flatten = TRUE))
}

read_model_jsonl <- function(path) {
  as_tibble(jsonlite::stream_in(file(path), verbose = FALSE))
}

# Calculate belief updates for prior/posterior data
# rating_col: name of the rating column (e.g., "intimacy_rating", "p_high_reward")
# Every (subject_id, scenario_label) group must hold exactly one prior and one
# posterior row; anything else would silently recycle or NA out the subtraction,
# so malformed groups abort with the offending (subject, scenario) pairs.
calculate_belief_update <- function(df, rating_col) {
  shape <- df |>
    count(subject_id, scenario_label, stage) |>
    pivot_wider(names_from = stage, values_from = n, values_fill = 0)
  for (col in c("prior", "posterior")) {
    if (!col %in% names(shape)) shape[[col]] <- 0L
  }
  bad <- shape |> filter(prior != 1 | posterior != 1)
  if (nrow(bad) > 0) {
    stop(
      "calculate_belief_update(): each (subject_id, scenario_label) group needs ",
      "exactly 1 prior and 1 posterior row; offending pairs:\n",
      paste0("  ", bad$subject_id, " / ", bad$scenario_label,
             " (prior = ", bad$prior, ", posterior = ", bad$posterior, ")",
             collapse = "\n")
    )
  }
  df |>
    group_by(subject_id, scenario_label) |>
    mutate(belief_update = ifelse(
      stage == "posterior",
      .data[[rating_col]][stage == "posterior"][1] -
        .data[[rating_col]][stage == "prior"][1],
      NA
    )) |>
    ungroup()
}

# Cluster bootstrap (resampling subjects with replacement) of per-cell means
# for one or more update columns. Used by the joint belief-update plots in the
# two-slider studies (food-inv-joint-de, food-inv-joint-ie), where each cell
# needs a CI on both axes. Returns one row per group with the observed mean
# per column plus <col>_ci_lower / <col>_ci_upper.
boot_cluster_means <- function(df, update_cols, group_vars, n_boot = 1000) {
  subjects <- unique(df$subject_id)
  boot_means <- map_dfr(seq_len(n_boot), function(i) {
    tibble(subject_id = sample(subjects, length(subjects), replace = TRUE)) |>
      inner_join(df, by = "subject_id", relationship = "many-to-many") |>
      group_by(across(all_of(group_vars))) |>
      summarize(across(all_of(update_cols), \(x) mean(x, na.rm = TRUE)),
                .groups = "drop")
  })
  observed <- df |>
    group_by(across(all_of(group_vars))) |>
    summarize(across(all_of(update_cols), \(x) mean(x, na.rm = TRUE)),
              .groups = "drop")
  cis <- boot_means |>
    group_by(across(all_of(group_vars))) |>
    summarize(
      across(all_of(update_cols),
             list(ci_lower = \(x) quantile(x, 0.025, na.rm = TRUE),
                  ci_upper = \(x) quantile(x, 0.975, na.rm = TRUE))),
      .groups = "drop"
    )
  left_join(observed, cis, by = group_vars)
}

# Observed-action factor: raw condition slugs -> ordered display labels, shared
# wherever the action_condition column becomes a plotting factor.
ACTION_LEVELS <- c("no_share", "low_risk_share", "high_risk_share")
ACTION_LABELS <- c("No share", "Low-risk share", "High-risk share")
factor_action <- function(x) factor(x, levels = ACTION_LEVELS, labels = ACTION_LABELS)

# Shared shape mapping for observed actions in the joint belief-update plots
ACTION_SHAPES <- c("No share" = 16, "Low-risk share" = 17, "High-risk share" = 15)

# Observed-action x-axis labels wrapped onto two lines (qualifier over "share")
# so they read horizontally without rotation. Underlying factor levels are
# unchanged ("No share" etc.) — this only relabels the axis ticks.
ACTION_AXIS_LABELS <- c(
  "No share"        = "No\nshare",
  "Low-risk share"  = "Low-risk\nshare",
  "High-risk share" = "High-risk\nshare"
)

scale_x_action <- function(...) {
  scale_x_discrete(labels = ACTION_AXIS_LABELS, ...)
}

# Save a plot to the repo-root figures/ directory as a vector PDF for the
# manuscript (creates the directory if needed). Uses the cairo_pdf device, which
# embeds the theme font (Arial Nova) and renders ggpattern fills cleanly — the
# base pdf() device does neither reliably. cairo_pdf needs cairo support in R (on
# macOS, install XQuartz: brew install --cask xquartz).
#
# width/height default to the current chunk's fig-width / fig-height (Quarto
# passes those to knitr as fig.width / fig.height), so the saved PDF matches the
# on-screen plot with no duplicated numbers. Pass them explicitly to override;
# 7 is the fallback when called outside a knitr render.
save_figure <- function(plot, filename, width = NULL, height = NULL, ...) {
  width  <- width  %||% knitr::opts_current$get("fig.width")  %||% 7
  height <- height %||% knitr::opts_current$get("fig.height") %||% 7
  fig_dir <- here("figures")
  if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
  ggsave(file.path(fig_dir, filename), plot = plot, width = width, height = height,
         device = cairo_pdf, ...)
}

# Print standardized demographics block from an experiment's exit_survey.csv.
# Demographics cover everyone recruited; the retained-after-exclusions count is
# read off the analyzed data (main_trials_long.csv), which
# analysis/json_to_csv.py writes after applying the study's exclusion rule.
# That script is the single source of truth for exclusion rules — deriving the
# N from its output means a rule change there can't silently desynchronize the
# manuscript-facing retention count reported here.
report_demographics <- function(data_dir) {
  path <- here("data", data_dir, "exit_survey.csv")
  if (!file.exists(path)) {
    path <- here("data", "legacy", data_dir, "exit_survey.csv")
  }
  df_exit <- read_csv(path, show_col_types = FALSE)
  n_total <- nrow(df_exit)
  cat("Total participants recruited:", n_total, "\n")
  long_path <- file.path(dirname(path), "main_trials_long.csv")
  if (file.exists(long_path)) {
    n_retained <- read_csv(long_path, show_col_types = FALSE) |>
      distinct(subject_id) |>
      nrow()
    cat("Retained after exclusions:", n_retained, "\n")
  } else {
    cat("Retained after exclusions: unknown (no main_trials_long.csv at",
        long_path, ")\n")
  }
  cat("Mean age:", round(mean(df_exit$age, na.rm = TRUE), 1),
      "SD age:", round(sd(df_exit$age, na.rm = TRUE), 1),
      "Min age:", min(df_exit$age, na.rm = TRUE),
      "Max age:", max(df_exit$age, na.rm = TRUE))
  cat("\nGender:\n")
  print(table(df_exit$gender))
  invisible(df_exit)
}

# Build a per-group correlation tibble with bootstrap CI and a formatted label
# column ready for geom_label / geom_text. group_vars is a character vector of
# columns to group by; pass NULL or omit for an overall correlation.
format_correlation_labels <- function(df, x, y, group_vars = NULL) {
  x <- rlang::enquo(x)
  y <- rlang::enquo(y)
  grouped <- if (length(group_vars)) {
    df |> group_by(across(all_of(group_vars)))
  } else {
    df
  }
  grouped |>
    summarize(
      boot_result = list(boot_cor(!!x, !!y)),
      .groups = "drop"
    ) |>
    mutate(
      r = sapply(boot_result, function(b) b$r),
      ci_lower = sapply(boot_result, function(b) b$ci_lower),
      ci_upper = sapply(boot_result, function(b) b$ci_upper),
      label = paste0(
        "r = ", round(r, 2),
        " (", round(ci_lower, 2), ", ", round(ci_upper, 2), ")"
      )
    ) |>
    select(-boot_result)
}
