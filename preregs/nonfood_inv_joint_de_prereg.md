# Preregistration for Study 3a

## 1) Data collection. Have any data been collected for this study already?

   1) Yes, we already collected the data.

   2) **No, no data have been collected for this study yet.**

   3) It's complicated. We have already collected some data but explain in Question 8 why readers may consider this a valid pre-registration nevertheless.

## 2) Hypothesis. What's the main question being asked or hypothesis being tested in this study?

Inverse planning models assume that agents choose actions rationally to achieve more reward and pay less cost, and that observers can infer agents' internal states (like knowledge and desire) from their actions. Yet standard inverse planning examples leave out a large class of rewards and costs that exist not in the physical world, but in the sociological world of relationships. The earlier studies in this project tested a social inverse planning model in food sharing, where the relational cost comes from the likelihood of saliva transfer. Here we test whether the same model, with the same relational cost structure, generalizes beyond food to other ways two people can be interpersonally vulnerable to each other: bodily access via shared objects and substances (e.g., sharing earbuds or a lip ointment), shared physical exposure (e.g., sharing one blanket or sleeping bag, changing in the same locker room), and private access (e.g., disclosing exact salaries, confiding about a breakup).

In this experiment, participants read vignettes describing scenarios where two people face a situation they could handle separately or by sharing an object, space, or piece of private information, and rate both how much they think the characters would like the relevant outcome and which of two physical world states (one making the low-risk way of handling the situation easy, one making it effortful) they think is more likely, both before and after observing the characters taking an action. We manipulate the intimacy of the two characters' relationship and the action they take.

We predict that (1) observing actions that attain the outcome will lead to higher inferred desire, and this effect will be stronger in more formal relationships, because vulnerable actions are more costly at low intimacy and therefore more diagnostic of high desire; (2) the observed action will also inform beliefs about the physical world state: observing the low-risk action will suggest that the low-effort world state is more likely, whereas observing the high-risk action or no sharing will suggest that the high-effort world state is more likely, because if the low-risk action had been easy it would have been a more attractive option; (3) the relationship will modulate these joint desire–effort inferences; and (4) the full social inverse planning model will better capture human inferences than alternative models that lack the full reward-cost structure. Of these predictions, (4) — the model comparison — is our primary confirmatory hypothesis, and (1)–(3) are directional qualitative predictions that we also assess. We also expect that these signatures should hold within each of the three scenario domains (bodily access, shared physical exposure, private access).

## 3) **Dependent variable. Describe the key dependent variable(s) specifying how they will be measured.**

Participants provide two ratings at each stage of a trial. Desire is measured with a continuous slider (0-100) on which participants rate how much they think the two characters would like the relevant outcome in the scenario (e.g., to listen to a new podcast episode, to warm up under a blanket, to know how their salaries compare), from "Not at all" (0) to "Extremely" (100) with "Moderately" at the midpoint. The beliefs about the physical world state are measured with a slider whose two endpoints are the two physical-world-state descriptions written for that scenario (one under which the low-risk way of handling the situation requires little effort, and one under which it requires substantial effort), with the middle of the scale labeled "Equally likely"; participants rate which of the two world states they believe is more likely. Both sliders appear on the same page. Participants provide both ratings twice for each scenario: once before observing the action (prior) and once after observing the action (posterior). The primary dependent variables are the belief updates (posterior rating minus prior rating) on each of the two sliders.

## 4) **Conditions. How many and which conditions will participants be assigned to?**

Participants read 16 vignettes describing different situations in US cultural contexts where two people could plausibly share an object, a space, or private information. The scenarios span three domains of interpersonal vulnerability: bodily access via shared objects and substances (6 scenarios), shared physical exposure (5 scenarios), and private access (5 scenarios). The names of the characters cover a variety of gender pairings (same gender, opposite gender, gender neutral). 

We manipulate two variables. First, how the characters in the scenario would describe their relationship (maximally formal, somewhat formal, somewhat intimate, or maximally intimate). Second, which action the characters are observed to take: 'no share', 'low-risk share', or 'high-risk share', where 'risk' here corresponds to the degree of interpersonal exposure the action creates — bodily substance transfer, physical exposure in close quarters, or access to private information.

Therefore, the design is 4 (Intimacy: maximally formal, somewhat formal, somewhat intimate, maximally intimate) × 3 (Action: no sharing, low-risk sharing, high-risk sharing), for 12 conditions. Each participant sees all 16 scenarios, each scenario in one of the 12 conditions pseudo-randomly assigned, and the assignment of condition to scenario is balanced across participants.

## 5) **Analyses. Specify exactly which analyses you will conduct to examine the main question/hypothesis.**

We test a social inverse planning model against two simpler ablated variants. In the full model, a joint actor chooses among the available actions via a softmax on total utility. The total utility of an action *a* given desire *d*, intimacy *I*, and effort condition *e* is:

U(a | d, I, e) = w_v · d · g(a) − w_d · risk(a) · (1 − I)^γ − w_e · effort(a | e)

The reward term w_v · d · g(a) is the product of the continuous desire d ∈ [0, 1] (how much the characters want the relevant outcome, one of the two latent variables inferred in this study) and the desire-free goal-satisfaction g(a) ∈ [0, 1] (the degree to which the action attains that outcome, independent of how much it is wanted). The risk term risk(a) captures the interpersonal vulnerability incurred by the action — bodily substance transfer, physical exposure, or private disclosure — and its effect on the utility is modulated by intimacy through (1 − I)^γ: at high intimacy the risk penalty shrinks toward zero. The effort term effort(a | e) captures the effort required to carry out the action, given the features of the environment set by the effort condition e (the other latent variable inferred in this study). For the disclosure actions in the private-access scenarios, effort is the executional cost of producing the account (how long the telling takes and how much explaining it requires), not the emotional difficulty of revealing the content, which the risk term captures. The exponent γ controls the shape of how intimacy attenuates discomfort.

To jointly infer desire and the physical world state, an observer inverts this model of an actor, using Bayesian inference:

P(d, e | a, I) \propto P(a | d, I, e) · P(d, e | I)

In this study, intimacy I is given by the description of the relationship; desire d and the effort condition e are the latent variables the observer infers.

The two alternative models are lesioned versions of the full model. The "discomfort only" model includes only the risk-discomfort term. The "base" model includes the reward and effort terms but drops the risk-discomfort term, removing the relational structure. Our main hypothesis is that the full model — which integrates desire, effort, and relationship-modulated discomfort within a single generative model of action selection — will best capture human inferences about desire and the physical world state.

The set of alternative actions the characters could plausibly have taken, and the features of those actions (goal-satisfaction, risk, effort) are estimated using queries to a language model that capture how these concepts are described in the prior literature. The intimacy I ∈ [0, 1] is also estimated by the language model given the verbal description of the relationship (one scalar per relationship level, elicited on a 0-100 scale and rescaled to [0, 1]). Because the base model's utility contains no relational term, its alternative action sets are elicited without the relationship description in the prompt; the full and discomfort-only models use alternative sets elicited with the relationship description included.

*Planned model fitting and comparison*

We will test whether the full model predicts participants' joint desire and world-state inferences better than each of the two lesioned variants (discomfort-only and base). Note that because the discomfort-only model omits both the reward term and the effort term, its utility depends on neither desire nor the physical world state, so it cannot infer either variable from the observed action and predicts no belief update on either slider. Therefore the main comparison here is full versus base, which isolates the contribution of the relationship-modulated discomfort term beyond the reward and effort terms.

The language-model pipeline and fitting procedure are the same as in the food studies (Studies 1b–2b), where they were developed and finalized. The language-model pipeline is run K = 20 times for each scenario × condition cell, and we treat each run as a simulated observer that supplies its own set of counterfactual actions and feature values that enter into a cognitive model and generate a resulting predicted belief update for each inferred variable. Alternative actions are generated at sampling temperature 0.7 and features are scored at temperature 0.2, using meta-llama/Llama-3.3-70B-Instruct-Turbo via the Together AI API; each feature is elicited on a 0-6 rating scale and linearly rescaled to [0, 1]. We set uniform priors over the inferred latent variables. We model a participant's pair of belief updates as drawn from a mixture over these simulated observers; because each run predicts a two-dimensional update (desire and world state), each mixture component is an isotropic bivariate Gaussian (a single shared σ, no covariance term) centered on that run's pair of predicted updates, and any predicted correlation between the two inferences enters through the run-to-run spread of these joint predictions. Across models, the actor's softmax inverse temperature is fixed at α = 1 for identifiability, and we additionally fit an observer inverse temperature α_obs that captures how sharply observers weigh higher-likelihood states, along with a response-noise scale σ; all of a model's free parameters are fit jointly by maximum likelihood. We will evaluate performance out of sample using leave-one-scenario-out cross-validation: parameters are estimated on 15 of the 16 scenarios and used to predict the held-out scenario. Our primary model-comparison metric is the per-trial held-out log-likelihood, with the difference between the 'full' and 'base' models reported with 95% confidence intervals obtained by bootstrap resampling of participants (1,000 resamples). As a secondary descriptive metric we will also report the out-of-sample Pearson correlation between condition-averaged model predictions and participants' belief updates.

*Assessment of the directional predictions*

We will assess the directional predictions (1)–(3) descriptively: for each of the two belief updates, we will check whether each predicted pattern holds in the direction of these condition means. These qualitative patterns complement the model comparison above, which is the study's confirmatory hypothesis.

*Per-domain analyses*

As secondary analyses, we will assess whether the results hold separately within each of the three scenario domains (bodily access, shared physical exposure, private access). We will similarly report the model-comparison metrics computed within each domain descriptively.

## 6) **Outliers and Exclusions. Describe exactly how outliers will be defined and handled, and your precise rule(s) for excluding observations.**

Participants who do not pass the comprehension check on the instructions in 3 tries will be told to return the study, and no data will be saved for them. Additionally, we include an attention check and two memory checks (the memory checks involve recalling details about the previous vignette, and comprise three questions in total: one check asks two questions and the other asks one). Participants will be retained for analysis only if they pass the attention check and answer at least one of the three memory-check questions correctly. As a robustness check, we will additionally verify that the results hold when retaining only the participants who answered all memory-check questions correctly.

## 7) **Sample Size. How many observations will be collected or what will determine sample size? (No need to justify the decision, but be precise about exactly how the number will be determined.)**

We will recruit 240 participants (pre-exclusions), for approximately 20 observations per scenario x condition combination. Participants will be recruited on Prolific and pre-screened to be adult fluent English speakers living in the United States; participants who completed previous studies will be excluded from participating.

## **Other. Anything else you would like to pre-register? (e.g., secondary analyses, variables collected for exploratory purposes, unusual analyses planned?)**

The above analyses and predictions primarily concern the marginals. We will also investigate how the two DVs relate to each other on the trial and scenario level. 

## **Name. Give a title for this AsPredicted pre-registration (Suggestion: use the name of the project, followed by study description.)**

Jointly inferring desire and the physical world state from interpersonally vulnerable non-food actions based on intimacy

## **Finally. For record keeping purposes, please tell us the type of study you are pre-registering.**

1)  Class project or assignment  
   2)  **Experiment**  
   3)  Survey  
   4)  Observational/archival study  
   5)  Other:
