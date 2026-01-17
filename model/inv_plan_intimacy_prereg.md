1) #### **Data collection. Have any data been collected for this study already?**

   1) #### Yes, we already collected the data.

   2) #### **No, no data have been collected for this study yet.**

   3) #### It's complicated. We have already collected some data but explain in Question 8 why readers may consider this a valid pre-registration nevertheless.

      

2) #### **Hypothesis. What's the main question being asked or hypothesis being tested in this study?**

Inverse planning models assume that agents choose actions rationally to achieve more reward and pay less cost, and that observers can infer agents' internal states (like knowledge and desire) from their actions. Yet standard inverse planning examples leave out a large class of rewards and costs that exist not in the physical world, but in the sociological world of relationships. For example, when people plan whether and how to share food, some of the key rewards and costs depend on their relationship — sharing food from the same plate or biting from the same part of the food feels comfortable only in relatively intimate relationships. Here, we develop a social inverse planning model that recognizes that agents select actions given their beliefs about relationships, and test whether this model can capture how human observers infer unobserved properties of the relationship (e.g. intimacy) from observed actions. 

In this experiment, participants read vignettes describing food-sharing scenarios and rate how intimate they think the relationship between two characters is, both before and after observing the characters take a specific food-sharing action. We manipulate the characters' motivational state (how much they want the food) and the action they take (ranging from not sharing the food at all to sharing in ways that involve increasing saliva-transfer risk). We predict that (1) observing riskier actions will lead to higher inferred intimacy, (2) this effect will be stronger in the low-motivation condition, because high motivation provides an alternative explanation for accepting discomfort, and (3) the full social inverse planning model will better capture human intimacy inferences than alternative models that lack the full reward-cost structure.

3) **Dependent variable. Describe the key dependent variable(s) specifying how they will be measured.**

The primary dependent variable is the belief update about intimacy, computed as the difference between participants’ prior and posterior ratings. Intimacy is measured using a continuous slider on a scale from 0 ("maximally formal") to 100 ("maximally intimate"). Participants provide this rating twice: once before observing the action (prior intimacy) and once after observing the action (posterior intimacy). 

4) **Conditions. How many and which conditions will participants be assigned to?**

We use the same 16 vignettes from the forward planning experiment, describing different situations in US cultural contexts where two people might plausibly share food (e.g., eating cake at a birthday party, eating a hot dog at a basketball game, getting drinks at a house party). The names of the characters cover a variety of gender pairings (same gender, opposite gender, gender neutral).

We manipulate the motivational state of the two people, corresponding to how much they want to eat the food. In the "low motivation" state, the two people don't want to or are indifferent to eating the food together (e.g., one person prefers another food, the people are not hungry). In the "high motivation" state, the two people want to eat the food (e.g., they both like the food, they are hungry). We also manipulate which action the characters are observed to take: Action 0 (not sharing the food), Action 1 (sharing without saliva-transfer risk), Action 2 (sharing with some saliva-transfer risk), or Action 3 (sharing with higher saliva-transfer risk).

Therefore, the design is 2 (Motivational state: low vs. high) × 4 (Action: 0, 1, 2, 3). Each participant sees all 16 scenarios, each scenario in one of the 8 conditions, and the assignment of condition to scenario is balanced within and across participants.

	

5) **Analyses. Specify exactly which analyses you will conduct to examine the main question/hypothesis.**

In the full model, a joint actor chooses an action proportional to its total utility, following a softmax choice rule with inverse temperature α. The total utility of an action a given motivational state s and intimacy I is:

U\_tot(a|s, I) \= w\_r · r(a|s, I) − w\_d · d(a|I) − w\_c · c(a)

The reward r(a|s, I) of taking a food-sharing action (compared to not eating the food) is the base reward r\_0, scaled by the motivational state s and the intimacy I — the more the actors want to eat the food, and the more intimate the relationship, the higher the reward of eating the food together (for Action 0, the reward is 0). The discomfort d(a|I) of taking the action is the saliva-transfer risk ρ(a), scaled down by intimacy: d(a|I) \= (1 − I) · ρ(a) — the more intimate the relationship, the less uncomfortable a risky action is. The sharing cost c(a) captures the baseline cost of sharing food versus not sharing. This may reflect a coordination cost: sharing requires negotiating who takes what and coordinating with another person, whereas not sharing allows each person to eat independently and is a simpler strategy. 

The two alternative models are lesioned versions of the full model. The "vanilla inverse planning" model includes all utility terms but does not consider how intimacy scales either reward or discomfort. The "discomfort only" model only considers how intimacy mitigates the discomfort of taking a risky action, without a representation of total reward or total cost.

Given an observed action *a* and motivational state *s*, the model infers intimacy *I* by inverting the forward model:

P(I | a, s) ∝ P(a | s, I) · P(I | s)

We will test whether the qualitative patterns in human data match the basic predictions of the full model. We will generate predictions from all three models using stipulated values based on the design: saliva-transfer risk ρ \= \[0, 0, 1, 2\] for Actions 0–3 and sharing cost c \= \[0, 1, 1, 1\], reward slope r\_0 \= 1\. To generate model predictions that we can compare directly to the observed belief updates, we use a uniform prior over intimacy. 

For each model, we freeze the forward planning parameters obtained from Experiment 1 (α=1 fixed for identifiability; w\_r, w\_d, w\_c fitted via maximum likelihood). We fit a single additional parameter, α\_obs, using maximum likelihood. This parameter captures how sharply the observer weighs the likelihood of different intimacy levels (or motivational states) given the observed action.

Full model frozen parameters: w\_r=1.132, w\_d=1.379, w\_c=0.874  
“Vanilla inv planning” frozen parameters: w\_r=1.648, w\_d=0.496, w\_c=0.882  
“Discomfort only” frozen parameters: w\_d=1.488

The best-fitting models will be compared using AIC. Since all models have one fitted parameter, ΔAIC \= 2 × ΔNLL. Following Burnham and Anderson (2002), ΔAIC values of 0–2 indicate essentially equivalent models, 4–7 indicate considerably less support for the higher-AIC model, and values greater than 10 indicate essentially no support for the higher-AIC model. We will also report bootstrapped 95% CIs on NLL differences.

This experiment (where participants infer intimacy given motivational state and action) is complemented by a separate experiment where participants infer motivational state given intimacy and action. To assess overall model fit across both inverse planning experiments, we will compute the Pearson correlation between mean human belief updates and model-predicted belief updates, aggregating across experiments (24 total conditions: 8 from the current experiment, 16 from the inverse planning experiment inferring motivational state). We will compute 95 percent bootstrapped confidence intervals over the correlation values. This aggregated correlation provides a test of whether the model generalizes across both types of inverse inference (inferring intimacy and inferring motivational state).

We predict that the full model will be a better fit than the alternatives. We will also visualize qualitative patterns to assess whether statistically preferred models capture theoretically meaningful features of the data. The discomfort-only model captures the basic pattern of how observing risky actions leads to higher inferred intimacy, while observing actions without saliva-sharing risk (Actions 0 and 1\) leads to lower inferred intimacy. However, only the full model should capture how motivational state modulates these inferences. When motivation is low, risky actions are more informative about intimacy — accepting discomfort despite not really wanting the food is diagnostic of an intimate relationship. When motivation is high, less-risky actions are more informative — choosing not to share despite really wanting the food is diagnostic of a formal relationship.

6) **Outliers and Exclusions. Describe exactly how outliers will be defined and handled, and your precise rule(s) for excluding observations.**

We include an attention check in the task and will exclude participants who fail this check. Second, we include two memory checks involving recalling details about the previous vignette (the names of the characters and what food they ate), and will exclude participants who fail both memory checks. 

7) **Sample Size. How many observations will be collected or what will determine sample size? (No need to justify the decision, but be precise about exactly how the number will be determined.)**

We will collect 120 participants. 

**Other. Anything else you would like to pre-register? (e.g., secondary analyses, variables collected for exploratory purposes, unusual analyses planned?)**

**Name. Give a title for this AsPredicted pre-registration (Suggestion: use the name of the project, followed by study description.)**

Inferring relationship intimacy from food-sharing actions based on motivational state

 **Finally. For record keeping purposes, please tell us the type of study you are pre-registering.**

1)  Class project or assignment  
   2)  **Experiment**  
   3)  Survey  
   4)  Observational/archival study  
   5)  Other:

