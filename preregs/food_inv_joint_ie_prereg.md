# Preregistration for Study 2b

## 1) Data collection. Have any data been collected for this study already?

   1) Yes, we already collected the data.

   2) **No, no data have been collected for this study yet.**

   3) It's complicated. We have already collected some data but explain in Question 8 why readers may consider this a valid pre-registration nevertheless.

## 2) Hypothesis. What's the main question being asked or hypothesis being tested in this study?

Inverse planning models assume that agents choose actions rationally to achieve more reward and pay less cost, and that observers can infer agents' internal states (like knowledge and desire) from their actions. Yet standard inverse planning examples leave out a large class of rewards and costs that exist not in the physical world, but in the sociological world of relationships. For example, when people plan whether and how to share food, some of the key rewards and costs depend on their relationship — sharing food from the same plate or biting from the same part of the food feels comfortable only in relatively intimate relationships. Here, we develop a social inverse planning model that recognizes that agents select actions jointly given their beliefs about relationships and the physical environment, and test whether this model can capture how human observers jointly infer the intimacy of the actors' relationship and the state of the physical environment.

In this experiment, participants read vignettes describing scenarios where two people would plausibly share food, and rate both how intimate they think the characters' relationship is and which of two physical world states (one making low-risk sharing easy, one making it effortful) they think is more likely, both before and after observing the characters take a food-sharing action. We manipulate how much the characters want the food and the action they take. Neither the characters' relationship nor the physical effort required to share in a low-risk way is described, so participants must infer both from the action.

We predict that (1) high-risk sharing will lead observers to infer a more intimate relationship and not sharing a more formal one, with low-risk sharing in between; (2) high desire will weaken the intimacy signal of sharing but strengthen the formality signal of not sharing; (3) because relationship and physical world state are inferred jointly, a single action updates both: low-risk sharing will lower the inferred effort of that option while high-risk sharing or no sharing will raise it, and high-risk sharing will raise inferred intimacy and inferred effort together (an intimate relationship and a high-effort environment being competing explanations for the riskier choice); and (4) the full social inverse planning model will better capture these inferences than models lacking the full reward-cost structure.

The model comparison is our primary confirmatory hypothesis.

## 3) **Dependent variable. Describe the key dependent variable(s) specifying how they will be measured.**

Participants provide two ratings at each stage of a trial. Intimacy is measured with a continuous slider (0-100) on which participants rate the characters' relationship, from "Maximally formal" (0) to "Maximally intimate" (100), with "Neither formal nor intimate" at the midpoint. The beliefs about the physical world state are measured with a slider whose two endpoints are the two physical-world-state descriptions written for that scenario (one under which sharing the food in a low-risk way requires little physical effort, and one under which it requires substantial effort), with the middle of the scale labeled "Equally likely"; participants rate which of the two world states they believe is more likely. Both sliders appear on the same page. Participants provide both ratings twice for each scenario: once before observing the action (prior) and once after observing the action (posterior). The primary dependent variables are the belief updates (posterior rating minus prior rating) on each of the two sliders.

## 4) **Conditions. How many and which conditions will participants be assigned to?**

Participants read 16 vignettes describing different situations in US cultural contexts where two people might plausibly share food (e.g., eating cake at a birthday party, eating a hot dog at a basketball game, getting drinks at a party). The names of the characters cover a variety of gender pairings (same gender, opposite gender, gender neutral). Each vignette includes a paragraph specifying how much the characters want the food. Neither a description of the characters' relationship nor a paragraph specifying the physical effort required to share the food in a low-risk way is shown — both are variables participants infer.

We manipulate two variables. First, how much the characters in the scenario want the food ('low' vs. 'high'), manipulated by scenario-specific paragraphs. Second, which action the characters are observed to take: 'no share', 'low-risk share', or 'high-risk share', where 'risk' here corresponds to the likelihood of saliva transfer associated with the action.

Therefore, the design is 2 (Desire: low vs. high) × 3 (Action: no sharing, low-risk sharing, high-risk sharing), for 6 conditions. Each participant sees all 16 scenarios, each scenario in one of the 6 conditions pseudo-randomly assigned, and the assignment of condition to scenario is balanced across participants.

## 5) **Analyses. Specify exactly which analyses you will conduct to examine the main question/hypothesis.**

We test a social inverse planning model against two simpler ablated variants. In the full model, a joint actor chooses among the available actions via a softmax on total utility. The total utility of an action *a* given desire *d*, intimacy *I*, and effort condition *e* is:

U(a | d, I, e) = w_v · d · g(a) − w_d · risk(a) · (1 − I)^γ − w_e · effort(a | e)

The reward term w_v · d · g(a) is the product of the continuous desire d ∈ [0, 1] (how much the characters want the food, which in this study is given by the desire paragraph) and the desire-free goal-satisfaction g(a) ∈ [0, 1] (the degree to which the action achieves the goal of eating or sharing the food, independent of how much the food is wanted). The risk term risk(a) captures the interpersonal vulnerability incurred by the action, which here is based on the likelihood and quantity of saliva transfer between two people (e.g., sharing the same utensil involves more risk than each person eating separately), and its effect on the utility is modulated by intimacy through (1 − I)^γ: at high intimacy the risk penalty shrinks toward zero. The effort term effort(a | e) captures the physical effort required to carry out the action, given the features of the physical environment set by the effort condition. The intimacy I ∈ [0, 1] and the effort condition e are the two latent variables inferred in this study. The exponent γ controls the shape of how intimacy attenuates discomfort.

To jointly infer intimacy and the physical world state, an observer inverts this model of an actor, using Bayesian inference:

P(I, e | a, d) \propto P(a | d, I, e) · P(I, e | d)

The two alternative models are lesioned versions of the full model. The "discomfort only" model includes only the risk-discomfort term. The "base" model includes the reward and effort terms but drops the risk-discomfort term, removing the relational structure. Our main hypothesis is that the full model — which integrates desire, physical effort, and relationship-modulated discomfort within a single generative model of action selection — will best capture human inferences about intimacy and the physical world state.

The set of alternative actions the characters could plausibly have taken, and the features of those actions (goal-satisfaction, risk, effort) are estimated using queries to a language model that capture how these concepts are described in the prior literature. The desire d ∈ [0, 1] implied by each desire paragraph is also estimated by the language model (one scalar per scenario × desire condition, elicited on a 0-100 scale and rescaled to [0, 1]).

*Planned model fitting and comparison*

We will test whether the full model predicts participants' joint intimacy and world-state inferences better than each of the two lesioned variants (discomfort-only and base). Note that because the base model omits the risk-discomfort term, its utility does not depend on intimacy, so it cannot infer intimacy from the observed action and predicts no intimacy update (though its effort term still lets it update beliefs about the physical world state). Conversely, the discomfort-only model omits the effort term, so it cannot update beliefs about the physical world state; it can produce intimacy updates and should reproduce their overall direction, but because it omits the reward and effort terms, it cannot capture how the desire manipulation and the inferred world state modulate those updates. Our primary confirmatory comparisons are therefore full versus discomfort-only and full versus base, and we predict that the full model will outperform both.

The Study 1a preregistration left some details of the language-model pipeline (e.g., number of LM runs, etc.) unspecified because that study supported the development of the pipeline. We specify those settings here. The language-model pipeline is run K = 20 times for each scenario × condition cell, and we treat each run as a simulated observer that supplies its own set of counterfactual actions and feature values that enter into a cognitive model and generate a resulting predicted belief update for each inferred variable. Alternative actions are generated at sampling temperature 0.7 and features are scored at temperature 0.2, using meta-llama/Llama-3.3-70B-Instruct-Turbo via the Together AI API; each feature is elicited on a 0-6 rating scale and linearly rescaled to [0, 1]. We set uniform priors over the inferred latent variables. We model a participant's pair of belief updates as drawn from a mixture over these simulated observers; because each run predicts a two-dimensional update (intimacy and world state), each mixture component is an isotropic bivariate Gaussian (a single shared σ, no covariance term) centered on that run's pair of predicted updates, and any predicted correlation between the two inferences enters through the run-to-run spread of these joint predictions. Across models, the actor's softmax inverse temperature is fixed at α = 1 for identifiability, and we additionally fit an observer inverse temperature α_obs that captures how sharply observers weigh higher-likelihood states, along with a response-noise scale σ; all of a model's free parameters are fit jointly by maximum likelihood. We will evaluate performance out of sample using leave-one-scenario-out cross-validation: parameters are estimated on 15 of the 16 scenarios and used to predict the held-out scenario. Our primary model-comparison metric is the per-trial held-out log-likelihood, with the differences between the 'full' model and each lesioned model reported with 95% confidence intervals obtained by bootstrap resampling of participants (1,000 resamples). As a secondary descriptive metric we will also report the out-of-sample Pearson correlation between condition-averaged model predictions and participants' belief updates.

*Assessment of the directional predictions*

We will assess the directional predictions (1)–(3) descriptively: for each of the two belief updates, we will report the mean update in each action × desire condition, with 95% confidence intervals from bootstrap resampling of participants, and check whether each predicted pattern holds in the direction of these condition means. We will also examine descriptively whether the world-state update within high-risk sharing is smaller when desire is high (high desire provides an alternative explanation for the risky choice). These qualitative patterns complement the model comparison above, which is the study's confirmatory hypothesis.

As a manipulation-independence check, we will report the difference in mean prior intimacy ratings between the two desire conditions (with a 95% confidence interval from bootstrap resampling of participants), to check that the desire manipulation does not itself shift prior beliefs about the relationship. The primary dependent variable subtracts the prior rating, but we will report this check regardless of its outcome.

## 6) **Outliers and Exclusions. Describe exactly how outliers will be defined and handled, and your precise rule(s) for excluding observations.**

Participants who do not pass the comprehension check on the instructions in 3 tries will be told to return the study, and no data will be saved for them. Additionally, we include an attention check and two memory checks (the memory checks involve recalling details about the previous vignette, and comprise three questions in total: one check asks two questions and the other asks one). Participants will be retained for analysis only if they pass the attention check and answer at least one of the three memory-check questions correctly. As a robustness check, we will additionally verify that the results hold when retaining only the participants who answered all memory-check questions correctly.

## 7) **Sample Size. How many observations will be collected or what will determine sample size? (No need to justify the decision, but be precise about exactly how the number will be determined.)**

We will recruit 120 participants (pre-exclusions), for approximately 20 observations per scenario x condition combination. Participants will be recruited on Prolific and pre-screened to be adult fluent English speakers living in the United States; participants who completed previous studies will be excluded from participating.

## **Other. Anything else you would like to pre-register? (e.g., secondary analyses, variables collected for exploratory purposes, unusual analyses planned?)**

The above analyses and predictions primarily concern the marginals. We will also investigate how the two DVs relate to each other on the trial and scenario level.

## **Name. Give a title for this AsPredicted pre-registration (Suggestion: use the name of the project, followed by study description.)**

Jointly inferring relationship intimacy and the physical world state from food-sharing actions based on desire

## **Finally. For record keeping purposes, please tell us the type of study you are pre-registering.**

1)  Class project or assignment  
   2)  **Experiment**  
   3)  Survey  
   4)  Observational/archival study  
   5)  Other:
