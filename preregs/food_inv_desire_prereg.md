# Preregistration for Study 1a

## 1) Data collection. Have any data been collected for this study already?

   1) Yes, we already collected the data.

   2) No, no data have been collected for this study yet.

   3) It's complicated. We have already collected some data but explain in Question 8 why readers may consider this a valid pre-registration nevertheless.

## 2) Hypothesis. What's the main question being asked or hypothesis being tested in this study?

Inverse planning models assume that agents choose actions rationally to achieve more reward and pay less cost, and that observers can infer agents' internal states (like knowledge and desire) from their actions. Yet standard inverse planning examples leave out a large class of rewards and costs that exist not in the physical world, but in the sociological world of relationships. For example, when people plan whether and how to share food, some of the key rewards and costs depend on their relationship — sharing food from the same plate or biting from the same part of the food feels comfortable only in relatively intimate relationships. Here, we develop a social inverse planning model that recognizes that agents select actions jointly given their beliefs about relationships and the physical environment, and test whether this model can capture how human observers infer the actor's desire for the food. 

In this experiment, participants read vignettes describing scenarios where two people would plausibly share food, and rate how much they think the characters would like the food, both before and after observing the characters take a food-sharing action. We manipulate the intimacy of the two characters' relationship, the physical effort required to share in a low-effort way, and the action they take. 

We predict that (1) observing sharing actions will lead to higher inferred desire, because sharing incurs costs that are only worth paying when desire is high; (2) this effect will be stronger at low intimacy and when the sharing action is more physically effortful, because both make sharing more costly and therefore more diagnostic of high desire, and (3) the full social inverse planning model will better capture human desire inferences than alternative models that lack the full reward-cost structure. Of these three predictions, (3) — the model comparison — is our primary confirmatory hypothesis, and (1) and (2) are directional qualitative predictions that we also assess.

## 3) **Dependent variable. Describe the key dependent variable(s) specifying how they will be measured.**

Desire is measured with a continuous slider (0-100) on which participants rate how much they think the two characters would like the relevant food in the scenario, from "Not at all" (0) to "Extremely" (100) with "Moderately" at the midpoint. Participants provide this rating twice for each scenario: once before observing the action (prior) and once after observing the action (posterior). The primary dependent variable is belief update (posterior rating minus prior rating).

## 4) **Conditions. How many and which conditions will participants be assigned to?**

Participants read 16 vignettes describing different situations in US cultural contexts where two people might plausibly share food (e.g., eating cake at a birthday party, eating a hot dog at a basketball game, getting drinks at a house party). The names of the characters cover a variety of gender pairings (same gender, opposite gender, gender neutral). Each vignette includes a paragraph specifying the physical effort required to share the food in a low-risk way, such as dividing the food into separate portions (e.g., knives for splitting the food are near them vs. they need to walk to the other side of the restaurant to get knives) and a description of the characters' relationship. 

We manipulate three variables. First, the physical effort required to share the food in a low-risk way ('low' vs. 'high'). Second, how the characters in the scenario would describe their relationship (maximally formal, somewhat formal, somewhat intimate, or maximally intimate). Third, which action the characters are observed to take: 'no share', 'low-risk share', or 'high-risk share', where 'risk' here corresponds to the likelihood of saliva transfer associated with the action. 

Therefore, the design is 2 (Effort: low vs. high) × 4 (Intimacy: maximally formal, somewhat formal, somewhat intimate, maximally intimate) × 3 (Action: no sharing, low-risk sharing, high-risk sharing). Each participant sees all 16 scenarios, each scenario in one of the 24 conditions pseudo-randomly assigned, and the assignment of condition to scenario is balanced across participants.

## 5) **Analyses. Specify exactly which analyses you will conduct to examine the main question/hypothesis.**

We test a social inverse planning model against two simpler variants. In the full model, a joint actor chooses an action proportional to its total utility via a softmax choice rule. The total utility of an action *a* given desire *d*, intimacy *I*, and effort condition *e* is:

U(a | d, I, e) = w_v · d · g(a) − w_d · risk(a) · (1 − I)^γ − w_e · effort(a | e)

The reward term w_v · d · g(a) is the product of the continuous desire d ∈ [0, 1] (how much the characters want the food, the latent variable inferred in this study) and the desire-free goal-satisfaction g(a) ∈ [0, 1] (the degree to which the action achieves the goal of eating or sharing the food, independent of how much the food is wanted). The risk term risk(a) captures the interpersonal vulnerability incurred by the action, which here is based on the likelihood and quantity of saliva transfer between two people (e.g., sharing the same utensil involves more risk than each person eating separately), and its effect on the utility is modulated by intimacy through (1 − I)^γ: at high intimacy the risk penalty shrinks toward zero. The effort term effort(a | e) captures the physical effort required to carry out the action, given the features of the physical environment set by the effort condition. The exponent γ controls the shape of how intimacy attenuates discomfort. 

To infer desire, an observer inverts this model of an actor, using Bayesian inference: 

P(d | a, I, e) \propto P(a | d, I, e) · P(d | I, e)

The two alternative models are lesioned versions of the full model. The "discomfort only" model includes only the risk-discomfort term. The "base" model includes the reward and effort terms but drops the risk-discomfort term, removing the relational structure. Our main hypothesis is that the full model — which integrates desire, physical effort, and relationship-modulated discomfort within a single generative model of action selection — will best capture human desire inferences.

The set of alternative actions the characters could plausibly have taken, and the features of those actions (goal-satisfaction, risk, effort) are estimated using queries to a language model that capture how these concepts are described in the prior literature. The intimacy I ∈ [0, 1] is also estimated given the verbal description of the relationship. 

*Planned model fitting and comparison*

We will test whether the full model predicts participants' desire inferences better than each of the two lesioned variants (discomfort-only and base). Note that because the discomfort-only model omits the reward term, it has no desire-dependent utility and so cannot infer desire from action, so it predicts no belief update. Therefore the main comparison here is full versus base, which isolates the contribution of the relationship-modulated discomfort term beyond the reward and effort terms.

Note that one purpose of this study is to support the development of the model pipeline, for the purposes of detailed preregistration and tests of generalization in future studies. Therefore, some of these details (e.g., number of LM runs, exact details of LM prompts, mixture model details and assumptions) are left unspecified in this preregistration. However, these settings are applied equally to all models and should not bias the comparison between them. 

The language-model pipeline is run multiple times for each scenario × condition cell, and we treat each run as a simulated observer that supplies its own set of counterfactual actions, feature values that enter into a cognitive model and generate a resulting predicted belief update δ_k. We set uniform priors over desire. We model a participant's belief update u as a drawn from a mixture over these simulated observers. We fit the free parameters for each model by scoring the model performance under this likelihood. Across models, we fix the actor's softmax to 1 for identifiability, and additionally fit an observer softmax temperature α_obs that captures how sharply observers weigh higher-likelihood states. We will evaluate performance out of sample using leave-one-scenario-out cross-validation: parameters are estimated on 15 of the 16 scenarios and used to predict the held-out scenario. Our primary model-comparison metric is the per-trial held-out log-likelihood, with the difference between the 'full' and 'base' models reported with 95% bootstrap CIs (resampling participants). As a secondary descriptive metric we will also report the out-of-sample Pearson correlation between each model's predictions and participants' belief updates. 

## 6) **Outliers and Exclusions. Describe exactly how outliers will be defined and handled, and your precise rule(s) for excluding observations.**

Participants who do not pass the comprehension check on the instructions in 3 tries will be told to return the study. Additionally, we include an attention check and two memory checks (the memory checks involve recalling details about the previous vignette: the names of the characters and what food they ate). We will exclude participants if they fail the attention check and both memory checks. We will check the robustness of this decision and also rerun the analyses retaining only the participants who passed both memory checks. 

## 7) **Sample Size. How many observations will be collected or what will determine sample size? (No need to justify the decision, but be precise about exactly how the number will be determined.)**

We will recruit 480 participants (pre-exclusions), for approximately 20 participants per scenario x condition combination. 

## **Other. Anything else you would like to pre-register? (e.g., secondary analyses, variables collected for exploratory purposes, unusual analyses planned?)**

## **Name. Give a title for this AsPredicted pre-registration (Suggestion: use the name of the project, followed by study description.)**

Inferring desire from food-sharing actions based on intimacy and physical effort

## **Finally. For record keeping purposes, please tell us the type of study you are pre-registering.**

1)  Class project or assignment  
   2)  **Experiment**  
   3)  Survey  
   4)  Observational/archival study  
   5)  Other:
