// Trial preview for collaborator feedback.
//
// This page does NOT run jsPsych. Instead it imports each study's real
// makeStimulusTrials(...) builder (the same code the live experiments use) and
// calls it for a single chosen scenario + condition, then renders the resulting
// jsPsych trial objects as static "screen" cards. Because every participant-
// facing string (vignette composition, lead-ins, slider labels, observed-action
// block, intimacy preamble) comes straight from _lib/ and the studies'
// trials.js, the preview cannot drift from what participants actually see.
//
// The jsPsych plugin globals (jsPsychHtmlSliderResponse etc.) are provided by
// the <script> tags in index.html; the trial objects reference them as their
// `type`, so they must exist, but jsPsych is never initialized or run.

import { makeStimulusTrials as make_food_inv_desire } from "../food_inv_desire/trials.js";
import { makeStimulusTrials as make_food_inv_joint_de } from "../food_inv_joint_de/trials.js";
import { makeStimulusTrials as make_food_inv_intimacy } from "../food_inv_intimacy/trials.js";
import { makeStimulusTrials as make_food_inv_joint_ie } from "../food_inv_joint_ie/trials.js";

// ----- study registry --------------------------------------------------------
// `given` = the latent variables this study reveals to the participant (these
// become selectable condition dropdowns). `inferred` = the latent variable(s)
// the participant rates — i.e. the dependent variable(s). `action` is always
// selectable (the observed behavior) and isn't listed here.
const STUDIES = {
  food_inv_desire: {
    paper: "1a",
    name: "Desire inference",
    make: make_food_inv_desire,
    given: ["intimacy", "effort"],
    inferred: ["desire"],
  },
  food_inv_joint_de: {
    paper: "1b",
    name: "Joint inference: desire + effort",
    make: make_food_inv_joint_de,
    given: ["intimacy"],
    inferred: ["desire", "effort"],
  },
  food_inv_intimacy: {
    paper: "2a",
    name: "Intimacy inference",
    make: make_food_inv_intimacy,
    given: ["desire", "effort"],
    inferred: ["intimacy"],
  },
  food_inv_joint_ie: {
    paper: "2b",
    name: "Joint inference: intimacy + effort",
    make: make_food_inv_joint_ie,
    given: ["desire"],
    inferred: ["intimacy", "effort"],
  },
};

// ----- condition factors -----------------------------------------------------
const INTIMACY_TEXTS = {
  max_formal: "maximally formal",
  neither: "neither formal nor intimate",
  somewhat_intimate: "somewhat intimate",
  max_intimate: "maximally intimate",
};

const FACTORS = {
  action: {
    label: "Observed action",
    field: "action_condition",
    options: [
      ["no_share", "no share"],
      ["low_risk_share", "low-risk share"],
      ["high_risk_share", "high-risk share"],
    ],
    format: (v) =>
      ({
        no_share: "no share",
        low_risk_share: "low-risk share",
        high_risk_share: "high-risk share",
      })[v],
  },
  intimacy: {
    label: "Intimacy",
    field: "intimacy_condition",
    options: [
      ["max_formal", "maximally formal"],
      ["neither", "neither formal nor intimate"],
      ["somewhat_intimate", "somewhat intimate"],
      ["max_intimate", "maximally intimate"],
    ],
    format: (v) => INTIMACY_TEXTS[v],
  },
  effort: {
    label: "Effort",
    field: "effort_condition",
    options: [
      ["low", "low"],
      ["high", "high"],
    ],
    format: (v) => v,
  },
  desire: {
    label: "Desire",
    field: "desire_condition",
    options: [
      ["low", "low"],
      ["high", "high"],
    ],
    format: (v) => v,
  },
};

const DV_LABELS = {
  desire: "desire",
  effort: "effort",
  intimacy: "intimacy",
};

// ----- state -----------------------------------------------------------------
// All four condition values are always carried so switching studies preserves
// the reviewer's choices; only the dropdowns relevant to the current study are
// shown. Defaults land on a diagnostic cell (a high-risk/saliva share between
// formal acquaintances), which is where the manipulations are most visible.
const state = {
  study: "food_inv_desire",
  scenario_label: null,
  action: "high_risk_share",
  intimacy: "max_formal",
  effort: "high",
  desire: "high",
};

// A minimal jsPsych stand-in. makeStimulusTrials only touches jsPsych inside
// makeInterTrialBlank's trial_duration callback, which we never invoke, so these
// methods exist only for safety.
const stubJsPsych = {
  randomization: {
    shuffle: (arr) => arr.slice(),
    sampleWithoutReplacement: (arr, n) => arr.slice(0, n),
  },
};

// jsPsych plugin globals, defined by the <script> tags in index.html. Each
// trial object carries one of these as its `type`. Reference them by bare name
// (as the studies' trials.js do): a jsPsych v8 plugin registers as a lexical
// `class`/`const` global, which is reachable by bare name but is NOT a property
// of `window`, so `window.jsPsychHtmlSliderResponse` would be undefined.
const TYPE_SLIDER = jsPsychHtmlSliderResponse;
const TYPE_FORM = jsPsychSurveyHtmlForm;
const TYPE_MULTI = jsPsychSurveyMultiChoice;
const TYPE_KEY = jsPsychHtmlKeyboardResponse;

let scenarios = [];
let scenariosByLabel = {};

// ----- helpers ---------------------------------------------------------------
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  for (const child of children) {
    if (child == null) continue;
    node.append(child.nodeType ? child : document.createTextNode(child));
  }
  return node;
}

// Inject experiment HTML, then drop the per-trial "Scenario N of M" heading:
// with a single-scenario preview it would read "Scenario 1 of 1", which is
// misleading. The card's own stage label conveys where we are instead.
function injectScreen(html) {
  const wrap = el("div", { className: "preview-screen" });
  wrap.innerHTML = html;
  wrap.querySelectorAll("h2").forEach((h) => {
    if (/^\s*Scenario\s+\d+\s+of\s+\d+\s*$/.test(h.textContent)) h.remove();
  });
  return wrap;
}

function fauxButton(label = "Continue") {
  return el("button", { className: "preview-button", disabled: true }, label);
}

// Static single-slider widget, mirroring the markup in _lib/two-slider.js so the
// shared .two-slider-* styles apply. The question text already sits in the
// trial's stimulus, so this renders only the track + endpoint labels.
function staticSlider(trial) {
  const labels = (trial.labels || [])
    .map((l) => `<div class="two-slider-label">${l}</div>`)
    .join("");
  const min = trial.slider_min ?? 0;
  const max = trial.slider_max ?? 100;
  const step = trial.step ?? 1;
  const start = Math.round((min + max) / 2);
  // Track width is governed by the `.card .two-slider-track` rule in index.html
  // (a fraction of the card, so the endpoint labels never clip).
  const wrap = el("div", { className: "two-slider-block" });
  wrap.innerHTML = `
    <div class="two-slider-track" style="margin: 0 auto;">
      <input type="range" class="jspsych-slider" min="${min}" max="${max}" step="${step}" value="${start}" />
      <div class="two-slider-labels">${labels}</div>
    </div>`;
  return wrap;
}

// Static multiple-choice rendering for the hike/wedding memory checks.
function staticMultiChoice(trial) {
  const wrap = el("div", { className: "preview-screen" });
  let html = trial.preamble || "";
  for (const q of trial.questions || []) {
    const opts = (q.options || [])
      .map(
        (opt) =>
          `<label class="question-option" style="display:block"><input type="radio" name="${q.name}" disabled /> ${opt}</label>`,
      )
      .join("");
    html += `
      <div class="jspsych-survey-multi-choice-question" style="margin: 16px auto; max-width: 720px; text-align: left;">
        <p class="jspsych-survey-multi-choice-text survey-multi-choice">${q.prompt}</p>
        ${opts}
      </div>`;
  }
  wrap.innerHTML = html;
  return wrap;
}

function stageOf(trial) {
  return trial.data && trial.data.stage ? trial.data.stage : null;
}

// ----- rendering -------------------------------------------------------------
function renderControls() {
  const host = document.getElementById("controls");
  host.replaceChildren();

  // Study
  const studySel = el("select");
  for (const [slug, s] of Object.entries(STUDIES)) {
    studySel.append(new Option(`${s.paper} — ${s.name}`, slug));
  }
  studySel.value = state.study;
  studySel.addEventListener("change", () => {
    state.study = studySel.value;
    renderControls();
    render();
  });
  host.append(controlGroup("Study", studySel));

  // Scenario
  const scenSel = el("select");
  for (const row of scenarios) {
    scenSel.append(
      new Option(
        `${row.scenario_label} — ${row.name_0} & ${row.name_1}`,
        row.scenario_label,
      ),
    );
  }
  scenSel.value = state.scenario_label;
  scenSel.addEventListener("change", () => {
    state.scenario_label = scenSel.value;
    render();
  });
  host.append(controlGroup("Scenario", scenSel));

  // Observed action (always) then the study's given-latent dropdowns.
  for (const key of ["action", ...STUDIES[state.study].given]) {
    const f = FACTORS[key];
    const sel = el("select");
    for (const [val, lab] of f.options) sel.append(new Option(lab, val));
    sel.value = state[key];
    sel.addEventListener("change", () => {
      state[key] = sel.value;
      render();
    });
    host.append(controlGroup(f.label, sel));
  }
}

function controlGroup(labelText, selectNode) {
  return el(
    "div",
    { className: "control" },
    el("label", {}, labelText),
    selectNode,
  );
}

function renderLegend() {
  const host = document.getElementById("legend");
  host.replaceChildren();
  const study = STUDIES[state.study];

  host.append(chip("given", "Observed action: ", FACTORS.action.format(state.action)));
  for (const key of study.given) {
    host.append(
      chip("given", `${FACTORS[key].label} (given): `, FACTORS[key].format(state[key])),
    );
  }
  const dvs = study.inferred.map((k) => DV_LABELS[k]).join(" + ");
  host.append(chip("inferred", "Rated by participant (DV): ", dvs));
}

// A legend pill: `mainText` is shown in the chip's bold weight, `valueText`
// (the current selection) in a lighter <span> (see .legend .chip span in
// index.html).
function chip(kind, mainText, valueText = "") {
  const c = el("span", { className: `chip ${kind}` });
  c.append(document.createTextNode(mainText));
  if (valueText) c.append(el("span", {}, valueText));
  return c;
}

function renderCards(trials) {
  const host = document.getElementById("cards");
  host.replaceChildren();

  for (const trial of trials) {
    const type = trial.type;

    // Single-slider rating (Studies 1a, 2a).
    if (type === TYPE_SLIDER) {
      const stage = stageOf(trial) || "rating";
      const body = injectScreen(trial.stimulus);
      body.append(staticSlider(trial));
      body.append(fauxButton(trial.button_label || "Continue"));
      host.append(card(stageLabel(stage), `stage-${stage}`, body));
      continue;
    }

    // Two-slider form (Studies 1b, 2b) and the memory checks both use survey
    // plugins; tell them apart by data.response_type.
    if (type === TYPE_FORM) {
      const stage = stageOf(trial) || "rating";
      const body = injectScreen((trial.preamble || "") + (trial.html || ""));
      body.append(fauxButton(trial.button_label || "Continue"));
      host.append(card(stageLabel(stage), `stage-${stage}`, body));
      continue;
    }

    if (type === TYPE_MULTI) {
      const body = staticMultiChoice(trial);
      body.append(fauxButton(trial.button_label || "Continue"));
      host.append(card("Memory check", "stage-memory", body));
      continue;
    }

    // Keyboard-response screens: the intimacy "press any key" intro (1a/1b),
    // the blank pause (empty stimulus), and the inter-trial blank ("Next
    // scenario"). Render only the intro; skip the content-free blanks.
    if (type === TYPE_KEY) {
      const s = (trial.stimulus || "").trim();
      if (s === "" || s === "Next scenario") continue;
      host.append(card("Intro screen", "stage-intro", injectScreen(trial.stimulus)));
      continue;
    }
  }
}

function stageLabel(stage) {
  if (stage === "prior") return "Prior rating (before observing the action)";
  if (stage === "posterior") return "Posterior rating (after observing the action)";
  return "Rating";
}

function card(labelText, labelClass, bodyNode) {
  return el(
    "div",
    { className: "card" },
    el("div", { className: `card-label ${labelClass}` }, labelText),
    el("div", { className: "card-body" }, bodyNode),
  );
}

function renderScenarioPanel() {
  const host = document.getElementById("scenario-panel");
  host.replaceChildren();
  const row = scenariosByLabel[state.scenario_label];
  if (!row) return;

  const rows = [
    ["group", "People & object"],
    ["Names", `${row.name_0} & ${row.name_1}`],
    ["Desire object", row.desire_object],
    ["group", "Vignette"],
    ["Vignette", row.vignette],
    ["group", "Desire paragraphs (given in 2a/2b)"],
    ["Desire — low", row.desire_low],
    ["Desire — high", row.desire_high],
    ["group", "Effort paragraphs (given in 1a/2a; slider endpoints in 1b/2b)"],
    ["Effort — low", row.low_risk_share_effort_low],
    ["Effort — high", row.low_risk_share_effort_high],
    ["group", "Observed actions"],
    ["No share", row.no_share],
    ["Low-risk share", row.low_risk_share],
    ["High-risk share", row.high_risk_share],
  ];

  const table = el("table");
  for (const [k, v] of rows) {
    if (k === "group") {
      const tr = el("tr", { className: "group-head" });
      tr.append(el("td", { colSpan: 2 }, v));
      table.append(tr);
    } else {
      const tr = el("tr");
      tr.append(el("th", {}, k), el("td", {}, v));
      table.append(tr);
    }
  }

  const details = el("details");
  details.append(
    el("summary", {}, `Full scenario data — ${row.scenario_label}`),
    table,
  );
  host.append(details);
}

function render() {
  try {
    const study = STUDIES[state.study];
    const row = scenariosByLabel[state.scenario_label];
    const stim = {
      ...row,
      action_condition: state.action,
      intimacy_condition: state.intimacy,
      effort_condition: state.effort,
      desire_condition: state.desire,
    };
    const trials = study.make(stubJsPsych, [stim]);
    renderLegend();
    renderCards(trials);
    renderScenarioPanel();
  } catch (err) {
    showError(err);
  }
}

function showError(err) {
  const host = document.getElementById("cards");
  host.replaceChildren(
    el(
      "div",
      { className: "preview-error" },
      `Failed to render the trial: ${err && err.message ? err.message : err}`,
    ),
  );
  console.error(err);
}

// ----- init ------------------------------------------------------------------
async function init() {
  try {
    // All four studies are generated from the same scenarios.csv, so one
    // study's stimuli.json carries every scenario row the preview needs.
    const res = await fetch("../food_inv_desire/json/stimuli.json");
    if (!res.ok) throw new Error(`stimuli.json: HTTP ${res.status}`);
    scenarios = await res.json();
    scenariosByLabel = Object.fromEntries(
      scenarios.map((r) => [r.scenario_label, r]),
    );
    state.scenario_label = scenarios[0].scenario_label;

    renderControls();
    render();
  } catch (err) {
    showError(err);
  }
}

init();
