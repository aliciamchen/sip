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
INTIMACY_LEVELS <- c("max_formal", "neither", "somewhat_intimate", "max_intimate")

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
  "neither"           = "Neither formal nor intimate",
  "somewhat_intimate" = "Somewhat intimate",
  "max_intimate"      = "Maximally intimate"
)

# Intimacy color scales (discrete) — numeric levels mapped to verbal labels
scale_fill_intimacy <- function() {
  scale_fill_manual(values = INTIMACY_COLORS, labels = INTIMACY_LABELS)
}

scale_color_intimacy <- function() {
  scale_color_manual(values = INTIMACY_COLORS, labels = INTIMACY_LABELS)
}

# Motivation color scales (discrete)
MOTIVATION_LEVELS <- c("Low", "High")
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

# Combined condition colors for inv-plan-combined-correlation (desire + intimacy)
.intimacy_levels <- INTIMACY_LEVELS
.intimacy_colors <- viridisLite::viridis(
  n = length(.intimacy_levels),
  begin = INTIMACY_BEGIN,
  end = INTIMACY_END,
  option = INTIMACY_PALETTE
)
names(.intimacy_colors) <- paste0("Intimacy: ", .intimacy_levels)

COMBINED_CONDITION_COLORS <- c(
  MOTIVATION_COLORS,
  .intimacy_colors
)

scale_color_combined_condition <- function() {
  scale_color_manual(values = COMBINED_CONDITION_COLORS)
}

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
calculate_belief_update <- function(df, rating_col) {
  df |>
    group_by(subject_id, scenario_label) |>
    mutate(belief_update = ifelse(stage == "posterior", .data[[rating_col]][stage == "posterior"] - .data[[rating_col]][stage == "prior"], NA)) |>
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

# Shared shape mapping for observed actions in the joint belief-update plots
ACTION_SHAPES <- c("No share" = 16, "Low-risk share" = 17, "High-risk share" = 15)

# Reusable jitter+dodge for risk scatter panels
POS_JITTER_DODGE <- position_jitterdodge(jitter.width = 0.04, jitter.height = 0,
                                          dodge.width = 0.06, seed = 67)

# Print standardized demographics block from an experiment's exit_survey.csv
report_demographics <- function(data_dir) {
  path <- here("data", data_dir, "exit_survey.csv")
  if (!file.exists(path)) {
    path <- here("data", "legacy", data_dir, "exit_survey.csv")
  }
  df_exit <- read_csv(path, show_col_types = FALSE)
  n_total <- nrow(df_exit)
  n_passed <- df_exit |>
    filter(attention_passed == TRUE, memory_correct_count > 0) |>
    nrow()
  cat("Total participants recruited:", n_total, "\n")
  cat("Passed attention + memory checks:", n_passed, "\n")
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

# Rescale a tidyboot summary (empirical_stat, ci_lower, ci_upper) by `scale`
# and rename empirical_stat to belief_update. DV ratings are now stored on the
# 0-1 scale (belief updates already fall in [-1, 1]), so the default scale = 1 is
# an identity; pass scale = 100 only for legacy 0-100 ratings.
rescale_belief_update <- function(df, scale = 1) {
  df |>
    mutate(
      belief_update = empirical_stat / scale,
      ci_lower = ci_lower / scale,
      ci_upper = ci_upper / scale
    )
}
