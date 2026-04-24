var jsPsychProbabilitySliders = (function (jspsych) {
  "use strict";

  const info = {
    name: "probability-sliders",
    parameters: {
      labels: {
        type: jspsych.ParameterType.STRING,
        array: true,
        pretty_name: "Labels",
        default: undefined,
      },
      start: {
        type: jspsych.ParameterType.FLOAT,
        array: true,
        pretty_name: "Start probabilities [0,1]",
        default: null,
      },
      button_label: {
        type: jspsych.ParameterType.STRING,
        pretty_name: "Button label",
        default: "Continue",
      },
      show_reset: {
        type: jspsych.ParameterType.BOOL,
        default: true,
      },
      show_chips: {
        type: jspsych.ParameterType.BOOL,
        default: true,
      },
      instruction_html: {
        type: jspsych.ParameterType.HTML_STRING,
        default:
          '<p class="ps-muted">Distribute probability across the events. You can move sliders freely. When you release a slider, all values will be automatically adjusted to sum to 100%.</p>',
      },
      precision: {
        type: jspsych.ParameterType.INT,
        default: 3,
      },
      require_total_exact: {
        type: jspsych.ParameterType.BOOL,
        default: true,
      },
      trial_duration: {
        type: jspsych.ParameterType.INT,
        default: null,
      },
      slider_min: {
        type: jspsych.ParameterType.INT,
        default: 0,
      },
      slider_max: {
        type: jspsych.ParameterType.INT,
        default: 100,
      },
      slider_step: {
        type: jspsych.ParameterType.INT,
        default: 1,
      },
      slider_width: {
        type: jspsych.ParameterType.INT,
        default: null,
      },
    },
  };

  class ProbabilitySlidersPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    static info = info;

    trial(display_element, trial) {
      const n = trial.labels.length;

      const toPercentsFromProbs = (probs) => {
        if (!probs || !Array.isArray(probs)) return null;
        const raw = probs.map((p) => Math.max(0, Math.min(1, p)) * 100);
        let ints = raw.map((x) => Math.floor(x));
        let remainder = 100 - ints.reduce((a, b) => a + b, 0);
        const fracs = raw
          .map((x, i) => ({ i, frac: x - Math.floor(x) }))
          .sort((a, b) => b.frac - a.frac);
        for (let k = 0; k < remainder; k++) ints[fracs[k].i] += 1;
        return ints;
      };

      const equalPercents = (n) => {
        let vals = Array(n).fill(Math.floor(100 / n));
        for (let r = 0; r < 100 - vals.reduce((a, b) => a + b, 0); r++)
          vals[r]++;
        return vals;
      };

      const normalize = (
        values,
        min = trial.slider_min,
        max = trial.slider_max
      ) => {
        const sum = values.reduce((a, b) => a + b, 0);
        if (sum === 0) {
          // If all values are 0, distribute equally
          const equalValue = Math.floor(100 / values.length);
          const remainder = 100 - equalValue * values.length;
          const normalized = values.map(() => equalValue);
          for (let i = 0; i < remainder; i++) {
            normalized[i] += 1;
          }
          return normalized;
        }

        // Normalize to sum to 100
        const normalized = values.map((v) => Math.round((v / sum) * 100));

        // Adjust for rounding errors to ensure exact sum of 100
        const actualSum = normalized.reduce((a, b) => a + b, 0);
        const diff = 100 - actualSum;

        if (diff !== 0) {
          // Add/subtract the difference to the largest value
          const maxIndex = normalized.indexOf(Math.max(...normalized));
          normalized[maxIndex] += diff;
        }

        // Ensure all values are within bounds
        return normalized.map((v) => Math.max(min, Math.min(max, v)));
      };

      const renderHTML = (labels, percents) => {
        const sliderWidthStyle = trial.slider_width
          ? `style="max-width:${trial.slider_width}px;"`
          : "";

        const rows = labels
          .map(
            (lbl, i) => `
            <div class="ps-row">
              <div class="ps-label">${lbl}</div>
              <div class="ps-slider-wrapper">
                <input class="ps-slider" type="range" min="${trial.slider_min}" max="${trial.slider_max}" step="${trial.slider_step}" value="${percents[i]}" id="ps-slider-${i}" ${sliderWidthStyle} />
                <div class="ps-value"><span id="ps-val-${i}">${percents[i]}</span>%</div>
              </div>
            </div>
          `
          )
          .join("");

        const resetBtn = trial.show_reset
          ? `<button class="ps-btn" id="ps-reset" type="button">Reset</button>`
          : "";

        return `
            <div class="ps-instruction-container">
              ${trial.instruction_html}
            </div>
            <div class="ps-slider-container">
              ${rows}
              <div class="ps-totalbar">
                <div>
                  ${resetBtn}
                  <button class="ps-btn" id="ps-continue" type="button" disabled>${trial.button_label}</button>
                </div>
              </div>
            </div>`;
      };

      // Initial percents
      let percents =
        trial.start && trial.start.length === n
          ? toPercentsFromProbs(trial.start)
          : equalPercents(n);

      display_element.innerHTML = renderHTML(trial.labels, percents);

      // Cache elements
      const btnCont = display_element.querySelector("#ps-continue");
      const btnReset = display_element.querySelector("#ps-reset");

      let hasSliderMoved = false;

      const updateUI = () => {
        for (let i = 0; i < n; i++) {
          const s = display_element.querySelector(`#ps-slider-${i}`);
          const v = display_element.querySelector(`#ps-val-${i}`);
          if (s) s.value = percents[i];
          if (v) v.textContent = percents[i];
        }
        // Enable continue button only after a slider has been moved
        if (btnCont) btnCont.disabled = !hasSliderMoved;
      };

      // Wire sliders
      for (let i = 0; i < n; i++) {
        const s = display_element.querySelector(`#ps-slider-${i}`);
        s.addEventListener("input", (e) => {
          // Allow free dragging - just update the specific slider value
          percents[i] = Number(e.target.value);
          hasSliderMoved = true;
          updateUI();
        });
        s.addEventListener("change", (e) => {
          // On release, normalize all values to sum to 100
          percents[i] = Number(e.target.value);
          percents = normalize(percents);
          hasSliderMoved = true;
          updateUI();
        });
      }

      if (btnReset) {
        btnReset.addEventListener("click", () => {
          percents = equalPercents(n);
          hasSliderMoved = false;
          updateUI();
        });
      }

      const createTrialData = () => {
        const probs = percents.map((v) => +(v / 100).toFixed(trial.precision));
        return {
          labels: trial.labels.slice(),
          probs,
          percents: percents.slice(),
          sum_check: +probs.reduce((a, b) => a + b, 0).toFixed(trial.precision),
        };
      };

      const end_trial = (data) => {
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(data);
      };

      btnCont.addEventListener("click", () => {
        end_trial(createTrialData());
      });

      if (trial.trial_duration !== null) {
        this.jsPsych.pluginAPI.setTimeout(() => {
          end_trial(createTrialData());
        }, trial.trial_duration);
      }

      updateUI();
    }

    // --- Simulation support (basic) ---
    simulate(trial, simulation_mode, simulation_options, load_callback) {
      if (simulation_mode === "data-only") {
        load_callback();
        this.simulate_data_only(trial, simulation_options);
      } else if (simulation_mode === "visual") {
        this.simulate_visual(trial, simulation_options, load_callback);
      }
    }

    create_simulation_data(trial, simulation_options) {
      const n = trial.labels.length;
      const default_probs = Array(n).fill(1 / n);
      const data = {
        labels: trial.labels.slice(),
        probs: default_probs,
        percents: default_probs.map((p) => Math.round(p * 100)),
        sum_check: 1,
      };
      return this.jsPsych.pluginAPI.mergeSimulationData(
        data,
        simulation_options
      );
    }

    simulate_data_only(trial, simulation_options) {
      const data = this.create_simulation_data(trial, simulation_options);
      this.jsPsych.finishTrial(data);
    }

    simulate_visual(trial, simulation_options, load_callback) {
      const data = this.create_simulation_data(trial, simulation_options);
      const display_element = this.jsPsych.getDisplayElement();
      this.trial(display_element, trial);
      load_callback();
      // Click continue after 800ms
      this.jsPsych.pluginAPI.setTimeout(() => {
        const btn = display_element.querySelector("#ps-continue");
        if (btn) btn.click();
      }, 800);
    }
  }

  ProbabilitySlidersPlugin.info = info;
  return ProbabilitySlidersPlugin;
})(jsPsychModule);
