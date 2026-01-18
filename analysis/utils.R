# analysis/utils.R - Shared utility functions for inverse planning analysis

library(here)
library(tidyverse)
library(tidyboot)

# Standard theme setup
setup_analysis <- function() {
  theme_set(theme_classic(base_size = 18))
  set.seed(67)
}

# Bootstrap correlation with 95% CI
# Used in: exp-2a, exp-2b, exp-2-combined
boot_cor <- function(x, y, n_boot = 1000) {
  complete <- complete.cases(x, y)
  x <- x[complete]
  y <- y[complete]
  n <- length(x)
  if (n < 3) return(list(r = NA_real_, ci_lower = NA_real_, ci_upper = NA_real_))
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
    mutate(
      belief_update = ifelse(
        stage == "posterior",
        .data[[rating_col]][stage == "posterior"] - .data[[rating_col]][stage == "prior"],
        NA
      )
    ) |>
    ungroup()
}
