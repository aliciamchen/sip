1) #### **Data collection. Have any data been collected for this study already?**

   1) #### Yes, we already collected the data.

   2) #### No, no data have been collected for this study yet.

   3) #### **It's complicated. We have already collected some data but explain in Question 8 why readers may consider this a valid pre-registration nevertheless.**

   *Our answer is (3). Exploratory data (an instrument pilot and a calibration sample) has been collected and used to finalize the procedure; this registration governs a fresh confirmatory sample collected afterward. See "Other" for the calibration-then-confirmation design.*

      

2) #### **Hypothesis. What's the main question being asked or hypothesis being tested in this study?**

Inverse planning models assume that agents choose actions rationally to achieve more reward and pay less cost, and that observers can infer agents' internal states (like knowledge and desire) from their actions. Yet standard inverse planning examples leave out a large class of rewards and costs that exist not in the physical world, but in the sociological world of relationships. For example, when people plan whether and how to share food, some of the key rewards and costs depend on their relationship — sharing food from the same plate or biting from the same part of the food feels comfortable only in relatively intimate relationships. The cost of sharing can also depend on physical features of the situation that make sharing easier or harder. Here, we develop a social inverse planning model that recognizes that agents select actions jointly given their beliefs about relationships and the physical situation, and test whether this model can capture how human observers infer the actor's unobserved desire (e.g. how much they want the food) from observed actions.

In this experiment, participants read vignettes describing food-sharing scenarios and rate how much they think the characters like the food, both before and after observing the characters take a specific food-sharing action. We manipulate the physical effort required to share (low vs. high), the characters' relationship intimacy (from maximally formal to maximally intimate), and the action they take (ranging from not sharing the food at all to sharing in ways that involve saliva-transfer risk). We predict that (1) observing sharing actions will lead to higher inferred desire, because sharing — and especially risky sharing — incurs costs that are only worth paying when desire is high, (2) this effect will be stronger at low intimacy and under high effort, because both make sharing more costly and therefore more diagnostic of high desire, and (3) the full social inverse planning model will better capture human desire inferences than alternative models that lack the full reward-cost structure. Of these three predictions, (3) — the model comparison — is our primary confirmatory hypothesis, and (1) and (2) are directional qualitative predictions that we also assess.

3) **Dependent variable. Describe the key dependent variable(s) specifying how they will be measured.**

The primary dependent variable is the belief update about how much the characters desire the food, computed as the posterior rating minus the prior rating. Desire is measured with a continuous 0–100 slider on which participants rate how much they think the two characters would like the food, anchored from "Not at all" (0) to "Extremely" (100) with "Moderately" at the midpoint. The slider question names the specific food in the scenario (e.g., "How much do you think Carissa and Josh would like the hot dog?"). Participants provide this rating twice for each scenario: once before observing the action (prior) and once after observing the action (posterior). The belief update is the posterior minus the prior rating.

4) **Conditions. How many and which conditions will participants be assigned to?**

We use 16 vignettes describing different situations in US cultural contexts where two people might plausibly share food (e.g., eating cake at a birthday party, eating a hot dog at a basketball game, getting drinks at a house party). The names of the characters cover a variety of gender pairings (same gender, opposite gender, gender neutral). Each vignette includes a paragraph specifying the physical effort required to share the food (low vs. high effort) and a description of the characters' relationship. Desire for the food is not stated in this study: it is the latent variable participants infer, so the scenario does not say how much the characters want the food.

We manipulate three observer-visible variables. First, the physical effort required to share the food, with a "low effort" paragraph (sharing is physically easy in this situation) and a "high effort" paragraph (sharing requires more physical effort, e.g., the food is divided across containers that are far apart, the food is awkward to pass). Second, how the characters would describe their relationship, presented as one of four verbal relationship descriptions: maximally formal, somewhat formal, somewhat intimate, or maximally intimate. Intimacy is conveyed only through these verbal descriptions; participants are never shown a numeric intimacy value. Third, which action the characters are observed to take: Action 0 (not sharing the food), Action 1 (low-risk sharing that avoids saliva transfer), or Action 2 (high-risk sharing that involves saliva transfer).

Therefore, the design is 2 (Effort: low vs. high) × 4 (Intimacy: maximally formal, somewhat formal, somewhat intimate, maximally intimate) × 3 (Action: no sharing, low-risk sharing, high-risk sharing). Each participant sees all 16 scenarios, each scenario in one of the 24 conditions, and the assignment of condition to scenario is balanced within and across participants.

5) **Analyses. Specify exactly which analyses you will conduct to examine the main question/hypothesis.**

*Cognitive hypotheses*

We test a social inverse planning model against two simpler variants. In the full model, an actor chooses an action proportional to its total utility via a softmax choice rule. The total utility of an action *a* given desire *d*, intimacy *I*, and effort condition *e* is:

U(a | d, I, e) = w_v · d · g(a) − w_d · risk(a) · (1 − I)^γ − w_e · effort(a | e)

The reward term w_v · d · g(a) is the product of the continuous desire d ∈ [0, 1] (how much the characters want the food, the latent variable inferred in this study) and the desire-free goal-satisfaction g(a) ∈ [0, 1] (the degree to which the action achieves the goal of eating or sharing the food, independent of how much the food is wanted). The risk term risk(a) captures the bodily, spatial, and informational exposure required by an action (e.g., sharing the same utensil involves more risk than each person eating separately), modulated by intimacy through a power-law factor (1 − I)^γ: at high intimacy the risk penalty shrinks toward zero, so high-risk actions become relatively more attractive. The effort term effort(a | e) captures the physical effort required to carry out the action under the scene's effort condition. The exponent γ controls how strongly intimacy attenuates discomfort.

The two alternative models are lesioned versions of the full model, each motivated by a different prior account. The "discomfort only" model includes only the risk-discomfort term (− w_d · risk(a) · (1 − I)^γ) and drops the reward and effort terms. The "base" model includes the reward (w_v · d · g(a)) and effort terms but drops the risk-discomfort term, removing all relational structure (it has no γ). Our central cognitive hypothesis is that the full model — which integrates desire, physical effort, and relationship-modulated discomfort within a single generative model of action selection — will best capture human desire inferences.

The goal-satisfaction, risk, and effort of each action, and the set of alternative actions the characters could plausibly have taken, are not stipulated by the experimenter. They are derived using a large language model that reads the same scenario information available to a participant, so that the model evaluates the observed action against context-appropriate alternatives and can be applied to new scenarios.

*Planned model fitting and comparison*

Our primary confirmatory analysis is the model comparison: we test whether the full model predicts participants' desire inferences better out of sample than each of the two lesioned variants (discomfort-only and base).

The three models are fit and evaluated with a single, identical procedure. Each model jointly estimates its own free parameters from the observed belief updates — the full model estimates the actor utility weights w_v, w_d, w_e, the intimacy exponent γ, and the observer softmax temperature α; the discomfort-only model estimates w_d, γ, and α; the base model estimates w_v, w_e, and α — so that no model is handicapped relative to the others. Because the fitting procedure is identical across the three models, it cannot by itself favor any one of them, and we fix the entire procedure before collecting the confirmatory sample (see "Sample Size" and "Other"), so it cannot be tuned to the confirmatory data.

We evaluate predictive performance out of sample using leave-one-scenario-out cross-validation: parameters are estimated on 15 of the 16 scenarios and used to predict the held-out scenario, rotating through all 16 folds and pooling the held-out predictions. Out-of-sample evaluation is essential to a fair comparison here, because the full model has an additional parameter (γ) and a richer cost structure and would tend to win on in-sample fit by construction; held-out prediction credits that added flexibility only if it captures real structure.

Our primary metric is the out-of-sample log-likelihood of held-out participants' belief updates under each fitted model, summed across the 16 folds. The central confirmatory prediction is that the full model attains a higher pooled out-of-sample log-likelihood than each lesioned model. For each pairwise comparison we compute, per held-out observation, the difference in log-likelihood between the full model and the lesion, and assess whether the mean difference is reliably positive by bootstrapping over participants. As a secondary, interpretable descriptive metric we will also report the out-of-sample correlation between each model's predictions and participants' mean belief updates.

*What we are and are not preregistering.* We preregister the elements that define the model comparison and that an analyst could otherwise use to influence its outcome: the exact set of models and their functional forms (above), the requirement that the fitting procedure be identical across models, the leave-one-scenario-out out-of-sample evaluation scheme, the primary metric, and the decision rule. We do not preregister in detail, and instead treat as exploratory, the components of the fitting procedure that are applied identically to all three models and therefore cannot bias the comparison among them: the optimizer and its settings, parameter initialization, the resolution of the inference grid, the parameter priors, and the choice of posterior summary statistic. These are fixed during the calibration phase, before the confirmatory sample is collected, and then held constant.

This study also serves to establish whether the overall approach works as a foundation for the planned follow-up experiments, including the engineering effort of developing and validating a language-model pipeline that turns natural-language scenario descriptions into the action sets and utility features the cognitive model requires. That development is exploratory and is completed and frozen before the confirmatory sample (see "Other").

*Qualitative predictions*

As secondary, directional predictions, we will visualize qualitative patterns to assess whether the models capture theoretically meaningful features of the data. The base model can capture the basic pattern that observing sharing leads to higher inferred desire (since sharing carries the effort cost but yields the goal-satisfaction reward). The discomfort-only model can capture how intimacy modulates inferences (at low intimacy, the risk penalty is steep, so risky actions are diagnostic of high desire; at high intimacy, the risk penalty is mild so risky actions are less diagnostic). Only the full model should capture both effects jointly, and additionally capture how effort modulates inferences: under high effort, any sharing action is more diagnostic of high desire, because the actor accepted both the effort cost and (for risky sharing) the risk cost.

6) **Outliers and Exclusions. Describe exactly how outliers will be defined and handled, and your precise rule(s) for excluding observations.**

We include an attention check in the task and will exclude participants who fail this check. Second, we include two memory checks involving recalling details about the previous vignette (the names of the characters and what food they ate), and will exclude participants who fail both memory checks.

7) **Sample Size. How many observations will be collected or what will determine sample size? (No need to justify the decision, but be precise about exactly how the number will be determined.)**

This registration governs a confirmatory sample of approximately 240 participants (before exclusions), recruited fresh after the fitting and comparison procedure has been finalized and frozen. We will recruit this fixed number and run the preregistered analyses on those who remain after the exclusions in Question 6. This sample size is adequate because the models' few global parameters are fit by pooling across all conditions rather than estimated per cell, and because the primary comparison is a paired out-of-sample contrast of the same held-out observations under each model, which is well powered even though individual scenario × condition cells are sampled sparsely at this sample size.

This confirmatory sample follows an earlier, exploratory calibration sample of approximately 240 participants (and, before that, a 15-participant instrument pilot) used to develop and freeze the procedure; the calibration data is not part of the confirmatory test (see "Other"). As a secondary, higher-precision analysis we will additionally report the preregistered analyses on the pooled calibration + confirmation sample (approximately 480 participants), treated as descriptive rather than as a clean confirmation because the calibration data informed our analysis choices.

**Other. Anything else you would like to pre-register? (e.g., secondary analyses, variables collected for exploratory purposes, unusual analyses planned?)**

This study uses a calibration-then-confirmation design. We first developed the experiment and the full analysis pipeline using exploratory data: a 15-participant instrument pilot (used to refine the vignettes and the effort-manipulation paragraphs and to verify that the experiment and the fitting code run end to end), followed by a calibration sample of approximately 240 participants used to finalize the language-model pipeline, the model-fitting procedure, and the model-comparison procedure. These exploratory samples were used too freely — inspecting the data while making analysis choices — to serve as a confirmatory test, and the pilot in particular was too small to fit the model reliably.

Before collecting the confirmatory sample, we freeze both (a) the language-model-derived components (goal-satisfaction, risk, effort, and the alternative-action sets), which are fixed prior to confirmatory data collection and are not re-elicited based on participant data, and (b) the full model-fitting and model-comparison procedure described under "Analyses." The prompts and the choice-set generation procedure are documented in the project repository.

The primary confirmatory analyses described under "Analyses" are conducted on the fresh confirmatory sample alone. We will additionally report those analyses on the pooled calibration + confirmation sample as a secondary, higher-precision estimate, clearly labeled as descriptive rather than confirmatory because the calibration data informed our analysis choices. Pilot and calibration results will be reported alongside and treated as exploratory.

**Name. Give a title for this AsPredicted pre-registration (Suggestion: use the name of the project, followed by study description.)**

Inferring desire from food-sharing actions based on intimacy and physical effort

 **Finally. For record keeping purposes, please tell us the type of study you are pre-registering.**

1)  Class project or assignment  
   2)  **Experiment**  
   3)  Survey  
   4)  Observational/archival study  
   5)  Other:
