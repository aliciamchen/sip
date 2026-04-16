# Exp 2a (intimacy inference): 4-way access-model comparison.
#
# Produces:
#   figures/access_model_exp2a_comparison.pdf — dodged bar chart of belief updates
#   per action, filled by motivation, faceted by model (4 access variants + Humans).
#
# Matches the style in analysis/exp-2a-inv-plan-intimacy-analysis.qmd
# (scale_fill_motivation, geom_col+dodge, crossbar zero-line, cairo_pdf).

source(here::here("analysis", "utils.R"))
setup_analysis()

MODEL_LEVELS <- c("Base", "Access only", "Full model", "Humans")

# ---- Human belief updates --------------------------------------------------

df_humans <- read_csv(
  here("data", "inv_plan_intimacy", "main_trials_long.csv"), show_col_types = FALSE
) |>
  mutate(
    action = str_replace(action_condition, "action_", ""),
    motivation = factor(motivation, levels = c("low", "high"), labels = MOTIVATION_LEVELS),
    stage = factor(stage, levels = c("prior", "posterior")),
    model = "Humans"
  ) |>
  select(-action_condition) |>
  calculate_belief_update("intimacy_rating") |>
  filter(stage == "posterior") |>
  group_by(motivation, action, model) |>
  tidyboot_mean(belief_update) |>
  ungroup() |>
  mutate(
    belief_update = empirical_stat / 100,
    ci_lower = ci_lower / 100,
    ci_upper = ci_upper / 100
  ) |>
  select(motivation, action, model, belief_update, ci_lower, ci_upper)

# ---- Access-variant model predictions --------------------------------------

df_model <- read_csv(
  here("model", "outputs", "inv_plan_intimacy_preds_summary.csv"), show_col_types = FALSE
) |>
  rename(motivation = reward_condition) |>
  filter(model %in% c("access_full", "access_only", "no_access")) |>
  mutate(
    action = as.character(action),
    motivation = factor(motivation, levels = c("low", "high"), labels = MOTIVATION_LEVELS),
    belief_update = (expected_intimacy - 50) / 100,
    model = recode(model,
      access_full = "Full model",
      access_only = "Access only",
      no_access   = "Base"
    )
  ) |>
  group_by(motivation, action, model) |>
  summarize(belief_update = mean(belief_update), .groups = "drop") |>
  mutate(ci_lower = NA_real_, ci_upper = NA_real_)

# ---- Combine & plot --------------------------------------------------------

df_combined <- bind_rows(df_humans, df_model) |>
  mutate(
    model = factor(model, levels = MODEL_LEVELS),
    action = factor(action)
  )

p <- df_combined |>
  ggplot(aes(x = action, y = belief_update,
             group = interaction(model, motivation), fill = motivation)) +
  geom_crossbar(
    aes(y = 0, ymin = -0.01, ymax = 0.01),
    position = position_dodge(width = 0.9),
    width = 0.9, color = NA
  ) +
  geom_col(position = position_dodge(width = 0.9), alpha = PLOT_ALPHA) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    position = position_dodge(width = 0.9),
    size = 1, width = 0.1, color = "black", alpha = 0.8
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
  scale_fill_motivation() +
  labs(y = "Belief update", x = "Observed action", fill = "Desire") +
  theme(legend.position = "right") +
  facet_wrap(~ model, nrow = 1)

ggsave(
  here("figures", "access_model_exp2a_comparison.pdf"),
  plot = p,
  width = FIG_WIDTH_LARGE, height = FIG_HEIGHT_STANDARD, device = cairo_pdf
)

cat("Saved figures/access_model_exp2a_comparison.pdf\n")
