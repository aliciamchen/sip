# Exp 2b (desire inference): 4-way access-model comparison.
#
# Produces:
#   figures/access_model_exp2b_comparison.pdf — dodged bar chart of belief updates
#   per action, filled by intimacy (4 levels), faceted by model (4 access variants + Humans).
#
# Matches the style in analysis/exp-2b-inv-plan-desire-analysis.qmd
# (scale_fill_intimacy, geom_col+dodge, crossbar zero-line, cairo_pdf).

source(here::here("analysis", "utils.R"))
setup_analysis()

MODEL_LEVELS <- c("Base", "Access only", "Full model", "Humans")

# ---- Human belief updates --------------------------------------------------

df_humans <- read_csv(
  here("data", "inv_plan_desire", "main_trials_long.csv"), show_col_types = FALSE
) |>
  mutate(
    action = str_replace(action_condition, "action_", ""),
    intimacy = factor(intimacy),
    stage = factor(stage, levels = c("prior", "posterior")),
    p_high_reward = response / 100,
    model = "Humans"
  ) |>
  select(-action_condition) |>
  calculate_belief_update("p_high_reward") |>
  filter(stage == "posterior") |>
  group_by(intimacy, action, model) |>
  tidyboot_mean(belief_update) |>
  ungroup() |>
  mutate(belief_update = empirical_stat) |>
  select(intimacy, action, model, belief_update, ci_lower, ci_upper)

# ---- Access-variant model predictions --------------------------------------

df_model <- read_csv(
  here("model", "outputs", "inv_plan_desire_preds_summary.csv"), show_col_types = FALSE
) |>
  rename(intimacy = intimacy_condition) |>
  filter(model %in% c("access_full", "access_only", "no_access")) |>
  mutate(
    action = as.character(action),
    intimacy = factor(intimacy),
    p_high_reward = p_high_reward / 100,
    belief_update = p_high_reward - 0.5,
    model = recode(model,
      access_full = "Full model",
      access_only = "Access only",
      no_access   = "Base"
    )
  ) |>
  group_by(intimacy, action, model) |>
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
             group = interaction(model, intimacy), fill = intimacy)) +
  geom_crossbar(
    aes(y = 0, ymin = -0.015, ymax = 0.015),
    position = position_dodge(width = 0.9),
    width = 0.9, color = NA
  ) +
  geom_col(position = position_dodge(width = 0.9), alpha = PLOT_ALPHA) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    position = position_dodge(width = 0.9),
    size = 1, width = 0, color = "black", alpha = 0.8
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
  scale_fill_intimacy() +
  labs(y = "Belief update", x = "Observed action", fill = "Intimacy") +
  theme(legend.position = "right") +
  facet_wrap(~ model, nrow = 1)

ggsave(
  here("figures", "access_model_exp2b_comparison.pdf"),
  plot = p,
  width = FIG_WIDTH_LARGE, height = FIG_HEIGHT_STANDARD, device = cairo_pdf
)

cat("Saved figures/access_model_exp2b_comparison.pdf\n")
