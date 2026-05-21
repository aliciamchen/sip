# analysis/utils.R - Shared utility functions for inverse planning analysis

library(here)
library(tidyverse)
library(tidyboot)
library(ragg)

# Figure dimension constants for consistent sizing across all outputs
# Use FIG_WIDTH_LARGE for 4+ facets or grids, FIG_WIDTH_STANDARD for 2-3 facets
# Legends on right add to width, so widths are increased accordingly
FIG_WIDTH_LARGE <- 14
FIG_WIDTH_STANDARD <- 12
FIG_HEIGHT_TALL <- 5      # For grid layouts (e.g., motivation × model)
FIG_HEIGHT_STANDARD <- 4  # For single-row faceted plots
FIG_HEIGHT_SHORT <- 3.5   # For correlation plots

# Color scheme constants
PLOT_ALPHA <- 0.95

# Intimacy color scale parameters - adjust these to test different ranges
INTIMACY_PALETTE <- "cividis"
INTIMACY_BEGIN <- 0.1
INTIMACY_END <- 0.85
INTIMACY_LEVELS <- c(0, 50, 75, 100)

# Generate discrete intimacy colors
INTIMACY_COLORS <- viridisLite::viridis(
  n = length(INTIMACY_LEVELS),
  begin = INTIMACY_BEGIN,
  end = INTIMACY_END,
  option = INTIMACY_PALETTE
)
names(INTIMACY_COLORS) <- as.character(INTIMACY_LEVELS)

# Intimacy color scales (discrete)
scale_fill_intimacy <- function() {
  scale_fill_manual(values = INTIMACY_COLORS)
}

scale_color_intimacy <- function() {
  scale_color_manual(values = INTIMACY_COLORS)
}

# Motivation color scales (discrete)
MOTIVATION_LEVELS <- c("Low", "High")
MOTIVATION_COLORS <- c("Low" = "#C9A8B0", "High" = "#7A4A5A")

scale_fill_motivation <- function() {
  scale_fill_manual(values = MOTIVATION_COLORS)
}

scale_color_motivation <- function() {
  scale_color_manual(values = MOTIVATION_COLORS)
}

# Effort color scales (discrete) - sage green, distinct from motivation (pink)
# and intimacy (cividis blue→yellow)
EFFORT_LEVELS <- c("Low effort", "High effort")
EFFORT_COLORS <- c("Low effort" = "#B5C9A8", "High effort" = "#4A7A4A")

scale_fill_effort <- function() {
  scale_fill_manual(values = EFFORT_COLORS)
}

scale_color_effort <- function() {
  scale_color_manual(values = EFFORT_COLORS)
}

# Combined condition colors for inv-plan-combined-correlation (motivation + intimacy)
.intimacy_levels <- c(0, 50, 75, 100)
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

# Bootstrap correlation with 95% CI
# Used in: food-inv-intimacy-desire-alt, food-inv-desire-intimacy-alt, inv-plan-combined-correlation
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

# Calculate belief updates for prior/posterior data
# rating_col: name of the rating column (e.g., "intimacy_rating", "p_high_reward")
calculate_belief_update <- function(df, rating_col) {
  df |>
    group_by(subject_id, scenario_label) |>
    mutate(belief_update = ifelse(stage == "posterior", .data[[rating_col]][stage == "posterior"] - .data[[rating_col]][stage == "prior"], NA)) |>
    ungroup()
}

# Create coord_fixed with symmetric x and y limits
# Calculates shared range from data and applies to both axes
coord_fixed_symmetric <- function(x, y, expand = 0.06) {
  range_val <- range(c(x, y), na.rm = TRUE)
  padding <- diff(range_val) * expand
  limits <- c(range_val[1] - padding, range_val[2] + padding)
  coord_fixed(xlim = limits, ylim = limits)
}

# Save plot to figures directory (creates directory if needed)
# Uses cairo_pdf for better font handling (supports Arial Nova)
# Requires XQuartz on macOS: brew install --cask xquartz
# Use standardized widths (10" or 12") for consistent font scaling
save_figure <- function(plot, filename, width = 12, height = 5, ...) {
  fig_dir <- here("figures")
  if (!dir.exists(fig_dir)) {
    dir.create(fig_dir, recursive = TRUE)
  }
  ggsave(here("figures", filename), plot = plot, width = width, height = height,
         device = cairo_pdf, ...)
}

# Reusable jitter+dodge for risk/access scatter panels
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

# Add AIC and (optionally) BIC columns to a fit-results table.
# n_obs: required for BIC; if NULL, only AIC is added.
add_aic_bic <- function(fit_results, n_obs = NULL) {
  out <- fit_results |>
    mutate(AIC = 2 * n_params + 2 * nll)
  if (!is.null(n_obs)) {
    out <- out |> mutate(BIC = n_params * log(n_obs) + 2 * nll)
  }
  out
}

# One-shot model-comparison kable for the inverse-planning notebooks.
# Computes AIC (if absent), delta_AIC from the min, and renders a kable
# with the standard column ordering and labels.
kable_aic_table <- function(fit_results,
                            caption = "Model comparison (lower AIC is better)",
                            model_col = "model_label",
                            model_col_label = "Model") {
  if (!"AIC" %in% names(fit_results)) {
    fit_results <- add_aic_bic(fit_results)
  }
  fit_results |>
    mutate(delta_AIC = AIC - min(AIC)) |>
    select(all_of(model_col), nll, n_params, AIC, delta_AIC) |>
    arrange(AIC) |>
    knitr::kable(
      digits = 2,
      caption = caption,
      col.names = c(model_col_label, "NLL", "n_params", "AIC", "ΔAIC")
    )
}

# Rescale a tidyboot summary (empirical_stat, ci_lower, ci_upper) by `scale`
# and rename empirical_stat to belief_update. Default scale = 100 maps
# 0-100 ratings to 0-1 belief updates.
rescale_belief_update <- function(df, scale = 100) {
  df |>
    mutate(
      belief_update = empirical_stat / scale,
      ci_lower = ci_lower / scale,
      ci_upper = ci_upper / scale
    )
}
