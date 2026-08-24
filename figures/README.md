# Figures

The scripts in `figures/scripts/` generate the figures and figure components
used in the manuscript. Files whose names begin with an underscore provide
shared plotting functions and are not run directly.

The outputs are organized by how they are used:

- `panels/` contains PDF components that are assembled into the final figures
  in Adobe Illustrator.
- `si/` contains complete PDF figures that can be included directly in the
  supplementary material.
- `model-eqs/` contains equation graphics used in the model schematic.

The main figure commands are:

```bash
make figures-panels                 # results panels and legends
make figures-nonfood-domains        # Study 3 results by sharing domain
make figures-schematic              # model schematic panels
make figures-lm-si                  # language-model validation figures
make figures-si-scenarios           # results for individual scenarios
make figures-si-prior-posterior     # prior and posterior rating distributions
make figures-si-prereg-predictions  # preregistered and reported predictions
```

All plotting scripts use `figures/scripts/plot_style.py` for fonts, colors, and
output locations. Each script saves a PDF and a PNG preview. The PDFs are
included in the repository; the PNG previews are ignored by Git.

The schematic panels use the example scores in
`figures/scripts/figure_scores.json`, so they can be generated without running
the fitted models.
