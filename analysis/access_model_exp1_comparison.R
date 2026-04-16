# Exp 1 (forward planning): access-model comparison.
#
# Produces:
#   figures/access_model_exp1_comparison.pdf — facet_grid(motivation ~ model),
#   line+point per intimacy level across actions, humans in the final facet.
#   figures/access_model_exp1_correlation.pdf — scatter of model preds vs humans
#   at condition x action level, one facet per access variant, with bootstrapped
#   correlation label in each panel.
#
# Three model variants are compared: Base (no_access), Access only, Full model
# (access_full), plus Humans. Matches the style conventions in
# analysis/exp-1-analysis.qmd (scale_color_intimacy, scale_shape_manual for
# actions, cairo_pdf, dodged lines + errorbars).

source(here::here("analysis", "utils.R"))
setup_analysis()

MODEL_LEVELS <- c("Base", "Access only", "Full model", "Humans")

# ---- Load & reshape predictions --------------------------------------------

df_preds <- read_csv(
  here("model", "outputs", "forward_planning_fits.csv"), show_col_types = FALSE
) |>
  select(
    subject_id, scenario_label, intimacy, motivation, action, p_action,
    pred_access_full, pred_access_only, pred_no_access
  ) |>
  pivot_longer(
    cols = c(p_action, pred_access_full, pred_access_only, pred_no_access),
    names_to = "model",
    values_to = "p_action"
  ) |>
  mutate(
    intimacy = factor(intimacy),
    motivation = factor(motivation, levels = c("low", "high"), labels = MOTIVATION_LEVELS),
    model = case_when(
      model == "p_action"         ~ "Humans",
      model == "pred_access_full" ~ "Full model",
      model == "pred_access_only" ~ "Access only",
      model == "pred_no_access"   ~ "Base"
    ),
    model = factor(model, levels = MODEL_LEVELS)
  )

df_preds_summary <- df_preds |>
  group_by(intimacy, motivation, action, model) |>
  tidyboot_mean(p_action) |>
  ungroup()

# ---- Plot 1: line + point comparison ---------------------------------------

p_comparison <- df_preds_summary |>
  ggplot(aes(x = action, y = empirical_stat,
             color = intimacy, shape = factor(action), group = intimacy)) +
  geom_point(position = position_dodge(width = 0.2), size = 4, alpha = PLOT_ALPHA) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    position = position_dodge(width = 0.2), size = 1.2, width = 0.5
  ) +
  geom_line(position = position_dodge(width = 0.2), linewidth = 1.2) +
  scale_color_intimacy() +
  scale_shape_manual(values = c(16, 17, 15, 18)) +
  labs(y = "P(Action)", x = "Action", color = "Intimacy", shape = "Action") +
  facet_grid(motivation ~ model)

ggsave(
  here("figures", "access_model_exp1_comparison.pdf"),
  plot = p_comparison,
  width = FIG_WIDTH_LARGE, height = FIG_HEIGHT_TALL, device = cairo_pdf
)

# ---- Plot 2: model vs human correlation scatter ----------------------------

df_humans <- df_preds_summary |>
  filter(model == "Humans") |>
  select(intimacy, motivation, action,
         Humans = empirical_stat,
         Humans_ci_lower = ci_lower,
         Humans_ci_upper = ci_upper)

df_corr <- df_preds_summary |>
  filter(model != "Humans") |>
  select(intimacy, motivation, action, model, model_pred = empirical_stat) |>
  left_join(df_humans, by = c("intimacy", "motivation", "action")) |>
  filter(!is.na(model_pred), !is.na(Humans)) |>
  mutate(
    action = factor(action),
    model = droplevels(model)
  )

correlations <- df_corr |>
  group_by(model) |>
  summarize(boot_result = list(boot_cor(model_pred, Humans)), .groups = "drop") |>
  mutate(
    r        = sapply(boot_result, function(x) x$r),
    ci_lower = sapply(boot_result, function(x) x$ci_lower),
    ci_upper = sapply(boot_result, function(x) x$ci_upper),
    label    = sprintf("r = %.2f (%.2f, %.2f)", r, ci_lower, ci_upper)
  ) |>
  select(-boot_result)

p_correlation <- ggplot(df_corr, aes(x = model_pred, y = Humans, color = intimacy, shape = action)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "lightgray") +
  geom_errorbar(aes(ymin = Humans_ci_lower, ymax = Humans_ci_upper), width = 0.02) +
  geom_point(size = 3, alpha = PLOT_ALPHA) +
  geom_label(
    data = correlations,
    aes(x = -Inf, y = Inf, label = label),
    inherit.aes = FALSE,
    hjust = -0.08, vjust = 1.5, size = 4.5,
    fill = "white", alpha = 0.5, label.size = 0
  ) +
  scale_color_intimacy() +
  scale_shape_manual(values = c(16, 17, 15, 18)) +
  labs(x = "Model", y = "Humans", color = "Intimacy", shape = "Action") +
  facet_wrap(~ model, nrow = 1) +
  coord_fixed_symmetric(df_corr$model_pred, df_corr$Humans)

ggsave(
  here("figures", "access_model_exp1_correlation.pdf"),
  plot = p_correlation,
  width = FIG_WIDTH_LARGE, height = FIG_HEIGHT_SHORT, device = cairo_pdf
)

cat("Saved figures/access_model_exp1_comparison.pdf\n")
cat("Saved figures/access_model_exp1_correlation.pdf\n")
print(correlations)
