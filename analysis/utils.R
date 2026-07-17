# analysis/utils.R - Shared R helpers for the analysis qmds and exploratory
# scripts (signature_tests.R). All figures — and their palettes — are Python
# now (repo-root plot_style.py is the visual source of truth; the qmds report
# demographics and data checks only), so this file keeps just the data-side
# helpers.

library(here)
library(tidyverse)
library(jsonlite)

# Condition-level factor orders shared with the Python side (plot_style.py /
# model tables). Intimacy is a purely verbal manipulation: the condition is
# stored as a slug (ascending, formal -> intimate), never a numeric code.
INTIMACY_LEVELS <- c(
  "max_formal",
  "somewhat_formal",
  "somewhat_intimate",
  "max_intimate"
)
ACTION_LEVELS <- c("no_share", "low_risk_share", "high_risk_share")

# Read the model pipeline's JSON / JSON Lines outputs into a tibble. The model
# code writes JSON (small structured artifacts: fit_results, *_preds_summary) and
# JSON Lines (per-record logs: cv_trial_ll, fit_restarts).
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
      paste0(
        "  ",
        bad$subject_id,
        " / ",
        bad$scenario_label,
        " (prior = ",
        bad$prior,
        ", posterior = ",
        bad$posterior,
        ")",
        collapse = "\n"
      )
    )
  }
  df |>
    group_by(subject_id, scenario_label) |>
    mutate(
      belief_update = ifelse(
        stage == "posterior",
        .data[[rating_col]][stage == "posterior"][1] -
          .data[[rating_col]][stage == "prior"][1],
        NA
      )
    ) |>
    ungroup()
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
    cat(
      "Retained after exclusions: unknown (no main_trials_long.csv at",
      long_path,
      ")\n"
    )
  }
  cat(
    "Mean age:",
    round(mean(df_exit$age, na.rm = TRUE), 1),
    "SD age:",
    round(sd(df_exit$age, na.rm = TRUE), 1),
    "Min age:",
    min(df_exit$age, na.rm = TRUE),
    "Max age:",
    max(df_exit$age, na.rm = TRUE)
  )
  cat("\nGender:\n")
  print(table(df_exit$gender))
  invisible(df_exit)
}
