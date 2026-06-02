1) #### **Data collection. Have any data been collected for this study already?**

   1) #### Yes, we already collected the data.

   2) #### No, no data have been collected for this study yet.

   3) #### **It's complicated. We have already collected some data but explain in Question 8 why readers may consider this a valid pre-registration nevertheless.**

      

2) #### **Hypothesis. What's the main question being asked or hypothesis being tested in this study?**

Inverse planning models assume that agents choose actions rationally to achieve more reward and pay less cost, and that observers can infer agents' internal states (like knowledge and desire) from their actions. Yet standard inverse planning examples leave out a large class of rewards and costs that exist not in the physical world, but in the sociological world of relationships. For example, when people plan whether and how to share food, some of the key rewards and costs depend on their relationship — sharing food from the same plate or biting from the same part of the food feels comfortable only in relatively intimate relationships. The cost of sharing can also depend on physical features of the situation that make sharing easier or harder. Here, we develop a social inverse planning model that recognizes that agents select actions jointly given their beliefs about relationships and the physical situation, and test whether this model can capture how human observers infer unobserved properties of the actor's desire state (e.g. how much they want the food) from observed actions.

In this experiment, participants read vignettes describing food-sharing scenarios and rate how much they think the characters want the food, both before and after observing the characters take a specific food-sharing action. We manipulate the physical effort required to share (low vs. high), the characters' relationship intimacy (from maximally formal to maximally intimate), and the action they take (ranging from not sharing the food at all to sharing in ways that involve saliva-transfer risk). We predict that (1) observing sharing actions will lead to higher inferred desire, because sharing — and especially risky sharing — incurs costs that are only worth paying when desire is high, (2) this effect will be stronger at low intimacy and under high effort, because both make sharing more costly and therefore more diagnostic of high desire, and (3) the full social inverse planning model will better capture human desire inferences than alternative models that lack the full reward-cost structure.

3) **Dependent variable. Describe the key dependent variable(s) specifying how they will be measured.**

The primary dependent variable is the belief update about desire state, computed as the difference between participants' prior and posterior ratings. Desire state is measured using a continuous slider where participants rate the relative probability of two contrasting desire states: the "low desire" state on one end and the "high desire" state on the other end, with the midpoint indicating that both states are equally likely. Participants provide this rating twice: once before observing the action (prior desire) and once after observing the action (posterior desire).

4) **Conditions. How many and which conditions will participants be assigned to?**

We use the same 16 vignettes from the forward planning experiment, describing different situations in US cultural contexts where two people might plausibly share food (e.g., eating cake at a birthday party, eating a hot dog at a basketball game, getting drinks at a house party). The names of the characters cover a variety of gender pairings (same gender, opposite gender, gender neutral). Each vignette includes a paragraph specifying the physical effort required to share the food (low vs. high effort) and a descriptor of the characters' relationship.

We manipulate three observer-visible variables. First, the physical effort required to share the food, with a "low effort" paragraph (sharing is physically easy in this situation) and a "high effort" paragraph (sharing requires more physical effort, e.g., the food is divided across containers that are far apart, the food is awkward to pass). Second, the description of how the characters would describe their relationship, from 0 "maximally formal" to 100 "maximally intimate" (sampled at 0, 50, 75, and 100). Third, which action the characters are observed to take: Action 0 (not sharing the food), Action 1 (sharing without saliva-transfer risk), or Action 2 (sharing with saliva-transfer risk).

Therefore, the design is 2 (Effort: low vs. high) × 4 (Intimacy: 0, 50, 75, 100) × 3 (Action: 0, 1, 2). Each participant sees all 16 scenarios, each scenario in one of the 24 conditions, and the assignment of condition to scenario is balanced within and across participants.

5) **Analyses. Specify exactly which analyses you will conduct to examine the main question/hypothesis.**

*Computational framework*

In the full model, an actor chooses an action proportional to its total utility, following a softmax choice rule with inverse temperature α (fixed to 1 for identifiability). The total utility of an action *a* given desire state *s*, intimacy *I*, and effort condition *e* is:

U(a | s, I, e) = w_v · V(a | s) − w_d · risk(a) · (1 − I)^γ − w_e · effort(a | e)

The signed valence V(a | s) ∈ [-1, +1] captures how well an action serves the actor's desire state (positive = serves, negative = actively counterproductive). The risk term risk(a) captures the bodily, spatial, and informational exposure required by an action (e.g., sharing the same utensil involves more risk than each person eating separately), modulated by intimacy through a power-law factor (1 − I)^γ: at high intimacy the risk penalty shrinks toward zero, so high-risk actions become relatively more attractive. The effort term effort(a | e) captures the physical effort required to carry out the action under the scene's effort condition. The exponent γ is a free parameter; γ = 1 reproduces the linear-intimacy special case.

The two alternative models are lesioned versions of the full model. The "discomfort only" model includes only the risk-discomfort term (− w_d · risk(a) · (1 − I)^γ) and drops V and effort. The "base" model includes V and effort but drops the risk-discomfort term, removing all relational structure (it has no γ since there is no intimacy modulator).

V, risk, and effort values are not stipulated by the experimenter; they are elicited per scenario from a large language model (Llama-3.3-70B via Together AI, 10 runs averaged per cell) using prompts that mirror what a human participant would read. Risk is elicited once per (scenario, action) and is effort-marginal by design (risk is formally an action property modulated by intimacy via the (1 − I)^γ term in the utility). Effort is elicited per (scenario, effort_condition, action). V is elicited per (scenario, action, desire_query).

The choice set the actor selects from is not the fixed 3-action canonical set. Instead, for each (scenario, observed_action, effort_condition, intimacy_condition) cell, the language model first generates plausible counterfactual actions the characters might have taken given what is salient in the situation, and then risk/effort/V are scored on the union {observed_action} ∪ generated_alternatives. The observer's actor softmaxes over this padded choice set (padded to 12 slots, with the observed canonical action in slot 0 by construction). This relaxes the experimenter's stipulation of "what the actor was choosing between" and lets the model reflect the actor's actual decision space.

*Inference*

Given an observed action *a*, intimacy *I*, and effort condition *e*, the model infers desire state *s* by inverting the forward model:

P(s | a, I, e) ∝ P(a | s, I, e) · P(s | I, e)

We use a uniform prior over desire state so that posteriors are directly comparable to participants' belief updates.

*Fitting*

For each model, we jointly fit the actor utility weights (w_v, w_d, w_e, γ — only the subset each variant uses) and the observer softmax temperature α_obs to the posterior data using maximum likelihood. We do not freeze forward planning parameters from the forward planning experiment; each inverse experiment fits its own actor weights, because the choice-set structure differs (the forward experiment uses a fixed 4-action set; this experiment uses LM-generated padded alternatives over a 3-action canonical core) and parameter values that minimize loss on one choice-set structure do not necessarily transfer.

*Model comparison*

The best-fitting models will be compared using AIC. We will also report bootstrapped 95% CIs on NLL differences. Following Burnham and Anderson (2002), ΔAIC values of 0–2 indicate essentially equivalent models, 4–7 indicate considerably less support for the higher-AIC model, and values greater than 10 indicate essentially no support for the higher-AIC model.

All model-vs-human correlations will be reported on held-out predictions from leave-one-scenario-out (LOSO) cross-validation: for each held-out scenario, we re-fit the actor utility weights and α_obs on the remaining 15 scenarios and predict the held-out scenario. We will report Pearson correlation between mean human belief updates and model-predicted belief updates at the condition × action level (8 effort × intimacy combinations × 3 actions = 24 conditions), with 95% bootstrapped confidence intervals.

*Qualitative predictions*

We will also visualize qualitative patterns to assess whether statistically preferred models capture theoretically meaningful features of the data. The base model can capture the basic pattern that observing sharing leads to higher inferred desire (since sharing carries the effort cost but yields V). The discomfort-only model can capture how intimacy modulates inferences (at low intimacy, the risk penalty is steep, so risky actions are diagnostic of high desire; at high intimacy, the risk penalty is mild so risky actions are less diagnostic). Only the full model should capture both effects jointly, and additionally capture how effort modulates inferences: under high effort, any sharing action is more diagnostic of high desire, because the actor accepted both the effort cost and (for risky sharing) the risk cost.

6) **Outliers and Exclusions. Describe exactly how outliers will be defined and handled, and your precise rule(s) for excluding observations.**

We include an attention check in the task and will exclude participants who fail this check. Second, we include two memory checks involving recalling details about the previous vignette (the names of the characters and what food they ate), and will exclude participants who fail both memory checks.

7) **Sample Size. How many observations will be collected or what will determine sample size? (No need to justify the decision, but be precise about exactly how the number will be determined.)**

We will collect 120 participants.

**Other. Anything else you would like to pre-register? (e.g., secondary analyses, variables collected for exploratory purposes, unusual analyses planned?)**

We previously ran a pilot version of this experiment with a smaller sample. The pilot was used to refine the vignettes and the effort manipulation paragraphs, and to verify that the experiment runs end-to-end. The pilot sample was too small to fit the model reliably, so all model-fitting analyses described above will be carried out for the first time on the registered full sample. We will report the pilot results alongside the registered analyses but treat them as exploratory.

The LM-elicited tables (V, risk, effort, alternative-action sets) are frozen prior to data collection and will not be re-elicited based on participant data. The prompts and the choice-set generation procedure are documented in the project repository.

**Name. Give a title for this AsPredicted pre-registration (Suggestion: use the name of the project, followed by study description.)**

Inferring desire state from food-sharing actions based on intimacy and physical effort

 **Finally. For record keeping purposes, please tell us the type of study you are pre-registering.**

1)  Class project or assignment  
   2)  **Experiment**  
   3)  Survey  
   4)  Observational/archival study  
   5)  Other:
