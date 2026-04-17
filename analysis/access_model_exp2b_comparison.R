# Exp 2b (desire inference): access-model comparison.
#
# Produces:
#   figures/access_model_exp2b_comparison.pdf — dodged bar chart of belief updates
#   per action, filled by intimacy, faceted by (param_source, model).
#
# Three variants (Base, Access only, Full model) crossed with two parameter
# sources when the LLM CSV is available (stipulated, llm); plus Humans.
# Matches analysis/exp-2b-inv-plan-desire-analysis.qmd style conventions.

source(here::here("analysis", "utils.R"))
setup_analysis()

MODEL_LEVELS <- c("Base", "Access only", "Full model", "Humans")

model_label <- function(name) {
  recode(name,
    access_full = "Full model", access_full_llm = "Full model",
    access_only = "Access only", access_only_llm = "Access only",
    no_access   = "Base",        no_access_llm   = "Base",
  )
}
param_source <- function(name) {
  if_else(endsWith(name, "_llm"), "llm", "stipulated")
}

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
  mutate(belief_update = empirical_stat, param_source = "Humans") |>
  select(intimacy, action, model, belief_update, ci_lower, ci_upper, param_source)

# ---- Access-variant model predictions --------------------------------------

access_variants <- c("access_full", "access_only", "no_access",
                     "access_full_llm", "access_only_llm", "no_access_llm")

df_model <- read_csv(
  here("model", "outputs", "inv_plan_desire_preds_summary.csv"), show_col_types = FALSE
) |>
  rename(intimacy = intimacy_condition) |>
  filter(model %in% access_variants) |>
  mutate(
    action = as.character(action),
    intimacy = factor(intimacy),
    p_high_reward = p_high_reward / 100,
    belief_update = p_high_reward - 0.5,
    param_source = param_source(model),
    model = model_label(model)
  ) |>
  group_by(intimacy, action, model, param_source) |>
  summarize(belief_update = mean(belief_update), .groups = "drop") |>
  mutate(ci_lower = NA_real_, ci_upper = NA_real_)

# ---- Combine & plot --------------------------------------------------------

df_combined <- bind_rows(df_humans, df_model) |>
  mutate(
    model = factor(model, levels = MODEL_LEVELS),
    param_source = factor(param_source, levels = c("stipulated", "llm", "Humans")),
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
  facet_grid(param_source ~ model)

n_rows <- length(unique(df_combined$param_source))
fig_height <- max(FIG_HEIGHT_STANDARD, 2.5 * n_rows)

ggsave(
  here("figures", "access_model_exp2b_comparison.pdf"),
  plot = p,
  width = FIG_WIDTH_LARGE, height = fig_height, device = cairo_pdf
)

cat("Saved figures/access_model_exp2b_comparison.pdf\n")
