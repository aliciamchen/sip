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
PLOT_ALPHA <- 0.85

# Intimacy color scales (continuous) - inferno palette
scale_fill_intimacy <- function() {
  scale_fill_viridis_c(option = "inferno", begin = 0.15, end = 0.85)
}

scale_color_intimacy <- function() {
  scale_color_viridis_c(option = "inferno", begin = 0.15, end = 0.85)
}

# Motivation color scales (discrete) - manual green values
MOTIVATION_COLORS <- c("low" = "#a1d99b", "high" = "#238b45")

scale_fill_motivation <- function() {
  scale_fill_manual(values = MOTIVATION_COLORS)
}

scale_color_motivation <- function() {
  scale_color_manual(values = MOTIVATION_COLORS)
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
# Used in: exp-2a, exp-2b, exp-2-combined
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
coord_fixed_symmetric <- function(x, y, expand = 0.05) {
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
