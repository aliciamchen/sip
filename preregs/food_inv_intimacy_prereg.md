# Preregistration for Study 2a

## 1) Data collection. Have any data been collected for this study already?

   1) Yes, we already collected the data.

   2) No, no data have been collected for this study yet.

   3) **It's complicated. We have already collected some data but explain in Question 8 why readers may consider this a valid pre-registration nevertheless.**

## 2) Hypothesis. What's the main question being asked or hypothesis being tested in this study?

Inverse planning models assume that agents choose actions rationally to achieve more reward and pay less cost, and that observers can infer agents' internal states (like knowledge and desire) from their actions. Yet standard inverse planning examples leave out a large class of rewards and costs that exist not in the physical world, but in the sociological world of relationships. For example, when people plan whether and how to share food, some of the key rewards and costs depend on their relationship — sharing food from the same plate or biting from the same part of the food feels comfortable only in relatively intimate relationships. Here, we develop a social inverse planning model that recognizes that agents select actions jointly given their beliefs about relationships and the physical environment, and test whether this model can capture how human observers infer the intimacy of the actors' relationship.

In this experiment, participants read vignettes describing scenarios where two people would plausibly share food, and rate how intimate they think the characters' relationship is, both before and after observing the characters take a food-sharing action. We manipulate how much the characters want the food, the physical effort required to share in a low-effort way, and the action they take. The characters' relationship is not described, so participants must infer it from the action.

We predict that (1) observing higher-risk sharing actions will lead to higher inferred intimacy, because the interpersonal risk these actions incur is only comfortable to bear in relatively intimate relationships; (2) this effect will be weaker when the characters' desire for the food is high (because high desire can itself explain the choice to share) and weaker when the low-risk sharing option is more physically effortful (because the effort can explain choosing the riskier option instead); and (3) the full social inverse planning model will better capture human intimacy inferences than alternative models that lack the full reward-cost structure. Of these three predictions, (3) — the model comparison — is our primary confirmatory hypothesis, and (1) and (2) are directional qualitative predictions that we also assess.

## 3) **Dependent variable. Describe the key dependent variable(s) specifying how they will be measured.**

Intimacy is measured with a continuous slider (0-100) on which participants rate the characters' relationship, from "Maximally formal" (0) to "Maximally intimate" (100). Participants provide this rating twice for each scenario: once before observing the action (prior) and once after observing the action (posterior). The primary dependent variable is belief update (posterior rating minus prior rating).

## 4) **Conditions. How many and which conditions will participants be assigned to?**

Participants read 16 vignettes describing different situations in US cultural contexts where two people might plausibly share food (e.g., eating cake at a birthday party, eating a hot dog at a basketball game, getting drinks at a house party). The names of the characters cover a variety of gender pairings (same gender, opposite gender, gender neutral). Each vignette includes a paragraph specifying how much the characters want the food and a paragraph specifying the physical effort required to share the food in a low-risk way, such as dividing the food into separate portions (e.g., knives for splitting the food are near them vs. they need to walk to the other side of the restaurant to get knives). Unlike Study 1a, no description of the characters' relationship is shown — the relationship is what participants infer.

We manipulate three variables. First, how much the characters in the scenario want the food ('low' vs. 'high'), manipulated by scenario-specific paragraphs. Second, the physical effort required to share the food in a low-risk way ('low' vs. 'high'). Third, which action the characters are observed to take: 'no share', 'low-risk share', or 'high-risk share', where 'risk' here corresponds to the likelihood of saliva transfer associated with the action.

Therefore, the design is 2 (Desire: low vs. high) × 2 (Effort: low vs. high) × 3 (Action: no sharing, low-risk sharing, high-risk sharing), for 12 conditions. Each participant sees all 16 scenarios, each scenario in one of the 12 conditions pseudo-randomly assigned, and the assignment of condition to scenario is balanced across participants.

## 5) **Analyses. Specify exactly which analyses you will conduct to examine the main question/hypothesis.**

We test a social inverse planning model against two simpler variants. In the full model, a joint actor chooses an action proportional to its total utility via a softmax choice rule. The total utility of an action *a* given desire *d*, intimacy *I*, and effort condition *e* is:

U(a | d, I, e) = w_v · d · g(a) − w_d · risk(a) · (1 − I)^γ − w_e · effort(a | e)

The reward term w_v · d · g(a) is the product of the continuous desire d ∈ [0, 1] (how much the characters want the food, which in this study is given by the desire paragraph) and the desire-free goal-satisfaction g(a) ∈ [0, 1] (the degree to which the action achieves the goal of eating or sharing the food, independent of how much the food is wanted). The risk term risk(a) captures the interpersonal vulnerability incurred by the action, which here is based on the likelihood and quantity of saliva transfer between two people (e.g., sharing the same utensil involves more risk than each person eating separately), and its effect on the utility is modulated by intimacy through (1 − I)^γ: at high intimacy the risk penalty shrinks toward zero. The intimacy I ∈ [0, 1] is the latent variable inferred in this study. The effort term effort(a | e) captures the physical effort required to carry out the action, given the features of the physical environment set by the effort condition. The exponent γ controls the shape of how intimacy attenuates discomfort.

To infer intimacy, an observer inverts this model of an actor, using Bayesian inference:

P(I | a, d, e) \propto P(a | d, I, e) · P(I | d, e)

The two alternative models are lesioned versions of the full model. The "discomfort only" model includes only the risk-discomfort term. The "base" model includes the reward and effort terms but drops the risk-discomfort term, removing the relational structure. Our main hypothesis is that the full model — which integrates desire, physical effort, and relationship-modulated discomfort within a single generative model of action selection — will best capture human intimacy inferences.

The set of alternative actions the characters could plausibly have taken, and the features of those actions (goal-satisfaction, risk, effort) are estimated using queries to a language model that capture how these concepts are described in the prior literature. The desire d ∈ [0, 1] implied by each desire paragraph is also estimated by the language model (one scalar per scenario × desire condition, elicited on a 0-100 scale and rescaled to [0, 1]).

*Planned model fitting and comparison*

We will test whether the full model predicts participants' intimacy inferences better than each of the two lesioned variants (discomfort-only and base). Note that because the base model omits the risk-discomfort term, its utility does not depend on intimacy, so it cannot infer intimacy from the observed action and predicts no belief update. The discomfort-only model retains the intimacy-modulated risk term, so it can produce intimacy updates and should reproduce their overall direction; but because it omits the reward and effort terms, it cannot capture how the desire and effort manipulations modulate those updates. Our primary confirmatory comparisons are therefore full versus discomfort-only and full versus base, and we predict that the full model will outperform both.

The Study 1a preregistration left some details of the language-model pipeline (e.g., number of LM runs, exact details of LM prompts, mixture model details) unspecified because that study supported the development of the pipeline. The pipeline is now finalized, so we specify those settings here. The language-model pipeline is run K = 20 times for each scenario × condition cell, and we treat each run as a simulated observer that supplies its own set of counterfactual actions and feature values that enter into a cognitive model and generate a resulting predicted belief update δ_k. Alternative actions are generated at sampling temperature 0.7 and features are scored at temperature 0.2, using meta-llama/Llama-3.3-70B-Instruct-Turbo via the Together AI API; each feature is elicited on a 0-6 rating scale and linearly rescaled to [0, 1]. We set uniform priors over intimacy. We model a participant's belief update u as drawn from a mixture over these simulated observers, with each mixture component a Gaussian centered on that run's predicted update δ_k. Across models, we fix the actor's softmax to 1 for identifiability, and additionally fit an observer softmax temperature α_obs that captures how sharply observers weigh higher-likelihood states, along with a response-noise scale σ. We will evaluate performance out of sample using leave-one-scenario-out cross-validation: parameters are estimated on 15 of the 16 scenarios and used to predict the held-out scenario. Our primary model-comparison metric is the per-trial held-out log-likelihood, with the differences between the 'full' model and each lesioned model reported with 95% confidence intervals obtained by bootstrap resampling of participants (1,000 resamples). As a secondary descriptive metric we will also report the out-of-sample Pearson correlation between condition-averaged model predictions and participants' belief updates.

## 6) **Outliers and Exclusions. Describe exactly how outliers will be defined and handled, and your precise rule(s) for excluding observations.**

Participants who do not pass the comprehension check on the instructions in 3 tries will be told to return the study, and no data will be saved for them. Additionally, we include an attention check and two memory checks (the memory checks involve recalling details about the previous vignette, and comprise three questions in total: one check asks two questions and the other asks one). Participants will be retained for analysis only if they pass the attention check and answer at least one of the three memory-check questions correctly. This rule is more stringent than the one preregistered for Study 1a, where participants were excluded only if they failed the attention check and both memory checks; that rule excluded zero participants, so we adopted a stricter rule for this study. As a robustness check, we will additionally verify that the results hold when retaining only the participants who answered all memory-check questions correctly.

## 7) **Sample Size. How many observations will be collected or what will determine sample size? (No need to justify the decision, but be precise about exactly how the number will be determined.)**

We will recruit 240 participants (pre-exclusions), for approximately 20 observations per scenario x condition combination. Participants will be recruited on Prolific and pre-screened to be adult fluent English speakers living in the United States; participants who completed previous studies in this line of research will be excluded from participating. Participants will be paid $5 for the study, which takes approximately 20 minutes. All procedures are approved by the MIT Committee on the Use of Humans as Experimental Subjects (COUHES).

## **Other. Anything else you would like to pre-register? (e.g., secondary analyses, variables collected for exploratory purposes, unusual analyses planned?)**

Regarding our answer to Question 1: a small pilot sample (approximately 17-20 participants) was collected for this study earlier, solely for the purpose of developing the modeling pipeline (the language-model elicitation settings, the mixture likelihood, and the fitting and cross-validation code). These pilot data will not be part of the confirmatory sample. The confirmatory sample will be collected fresh after this preregistration, and all confirmatory analyses will be run on the new data only.

## **Name. Give a title for this AsPredicted pre-registration (Suggestion: use the name of the project, followed by study description.)**

Inferring relationship intimacy from food-sharing actions based on desire and physical effort

## **Finally. For record keeping purposes, please tell us the type of study you are pre-registering.**

1)  Class project or assignment  
   2)  **Experiment**  
   3)  Survey  
   4)  Observational/archival study  
   5)  Other:
