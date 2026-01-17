1) #### **Data collection. Have any data been collected for this study already?**

   1) #### Yes, we already collected the data.

   2) #### **No, no data have been collected for this study yet.**

   3) #### It's complicated. We have already collected some data but explain in Question 8 why readers may consider this a valid pre-registration nevertheless.

      

2) #### **Hypothesis. What's the main question being asked or hypothesis being tested in this study?**

Inverse planning models assume that agents choose actions rationally to achieve more reward and pay less cost, and that observers can infer agents' internal states (like knowledge and desire) from their actions. Yet standard inverse planning examples leave out a large class of rewards and costs that exist not in the physical world, but in the sociological world of relationships. For example, when people plan whether and how to share food, some of the key rewards and costs depend on their relationship — sharing food from the same plate or biting from the same part of the food feels comfortable only in relatively intimate relationships. Here, we develop a social inverse planning model that recognizes that agents select actions given their beliefs about relationships, and test whether this model captures how human observers predict food-sharing actions.

In this experiment, participants read vignettes describing food-sharing scenarios and rate the probability that two characters will take each of four possible actions, ranging from not sharing the food at all to sharing in ways that involve increasing saliva-transfer risk. We manipulate the characters' motivational state (how much they want the food) and relationship intimacy (from maximally formal to maximally intimate). We predict that the full social inverse planning model will better fit human predictions than two alternative models: a "discomfort only" model that only considers how intimacy mitigates discomfort from saliva-sharing risk, and a "vanilla inverse planning" model that does not consider how intimacy scales reward and cost.

3) **Dependent variable. Describe the key dependent variable(s) specifying how they will be measured.**

The primary dependent variable is probability ratings for each of four actions, measured using four sliders that are constrained to sum to 1 (100%). Action 0 is where the two characters do not share the food together (e.g., one person eats the food while the other eats something else). Action 1 is where they share the food in a way that does not risk saliva transfer (e.g., pouring separate portions). Action 2 is where they share the food with some risk of saliva transfer (e.g., double-dipping). Action 3 is where they share the food with higher risk of saliva transfer (e.g., passing the same food item back and forth, biting from the same spot).

4) **Conditions. How many and which conditions will participants be assigned to?**

We wrote 16 vignettes describing different situations in US cultural contexts where two people might plausibly share food (e.g., eating cake at a birthday party, eating a hot dog at a basketball game, getting drinks at a house party). The names of the characters cover a variety of gender pairings (same gender, opposite gender, gender neutral).

We manipulate the utility of the food itself for the two people, corresponding to how much they want to eat the food. In the 'low motivation' state, the two people don't want to or are indifferent to eating the food together (e.g., one person prefers another food, the people are not hungry). In the 'high motivation' state, the two people want to eat the food (e.g., they both like the food, they are hungry). We also manipulate the description of how the characters would describe their relationship, from 0 "maximally formal" to 100 "maximally intimate" (sampled at 0, 50, 75, and 100).

Therefore, the design is 2 (Motivational state: low vs. high) x 4 (Intimacy: 0, 50, 75, 100). Each participant sees all 16 scenarios, each scenario in one of the 8 conditions, and the assignment of condition to scenario is balanced within and across participants.   
	

5) **Analyses. Specify exactly which analyses you will conduct to examine the main question/hypothesis.**

*Computational framework*

In the full model, a joint actor chooses an action proportional to its total utility, following a softmax choice rule with inverse temperature α. The total utility of an action a given motivational state s and intimacy I is:

U\_tot(a|s, I) \= w\_r · r(a|s, I) − w\_d · d(a|I) − w\_c · c(a)

The reward r(a|s, I) of taking a food-sharing action (compared to not eating the food) is the base reward r\_0, scaled by the motivational state s and the intimacy I — the more the actors want to eat the food, and the more intimate the relationship, the higher the reward of eating the food together (for Action 0, the reward is 0). The discomfort d(a|I) of taking the action is the saliva-transfer risk ρ(a), scaled down by intimacy: d(a|I) \= (1 − I) · ρ(a) — the more intimate the relationship, the less uncomfortable a risky action is. The sharing cost c(a) captures the baseline cost of sharing food versus not sharing. This may reflect a coordination cost: sharing requires negotiating who takes what and coordinating with another person, whereas not sharing allows each person to eat independently and is a simpler strategy. 

The two alternative models are lesioned versions of the full model. The "vanilla inverse planning" model includes all utility terms but does not consider how intimacy scales either reward or cost. The "discomfort only" model only considers how intimacy mitigates the discomfort of taking a risky action, without a representation of total reward or cost.

We will test whether the qualitative patterns in human data match the basic predictions of the full model. We will generate predictions from all three models using stipulated values based on the design: saliva-transfer risk ρ \= \[0, 0, 1, 2\] for Actions 0–3 and sharing cost c \= \[0, 1, 1, 1\], reward slope r\_0 \= 1\. 

For each model, we will fit the weights (w\_r, w\_d, w\_c), and inverse temperature α to the data using maximum likelihood. The best-fitting models will be compared using AIC and Pearson correlation. Following Burnham and Anderson (2002), ΔAIC values of 0–2 indicate essentially equivalent models, 4–7 indicate considerably less support for the higher-AIC model, and values greater than 10 indicate essentially no support for the higher-AIC model. Pearson correlation for each model will be computed at the condition x action level; we will compute 95 percent bootstrapped confidence intervals over the correlation values. 

We predict that the full model will be a better fit than the alternatives. We will also visualize qualitative patterns to assess whether statistically preferred models capture theoretically meaningful features of the data that alternative models miss. Specifically, only the full model should capture two features: (1) the preference for Action 0 in the low-motivation condition and the preference for Action 1 in the high-motivation condition, and (2) the separation of action probabilities based on intimacy, where higher intimacy makes less-risky actions less likely and more-risky actions more likely.

6) **Outliers and Exclusions. Describe exactly how outliers will be defined and handled, and your precise rule(s) for excluding observations.**

We include an attention check in the task and will exclude participants who fail this check. Second, we include two memory checks involving recalling details about the previous vignette (the names of the characters and what food they ate), and will exclude participants who fail both memory checks. 

7) **Sample Size. How many observations will be collected or what will determine sample size? (No need to justify the decision, but be precise about exactly how the number will be determined.)**

We will collect 120 participants. 

**Other. Anything else you would like to pre-register? (e.g., secondary analyses, variables collected for exploratory purposes, unusual analyses planned?)**

The confirmatory analyses test the model at the level of experimental conditions by using stipulated parameter values that are constant across all 16 scenarios. If our confirmatory hypothesis is supported, we will explore methods for capturing scenario-specific variance. Specifically, we will explore the use of large language models to provide estimates of scenario-specific parameters (for risk (ρ), effort (c), and reward (r\_0)) directly from the stimulus materials, and whether this improves model fit compared to using the experimenter-assigned stipulated values. If this significantly improves model fit in the current experiment, then we will also preregister model fits using these LLM-derived scenario-specific values in future experiments. This approach allows us to potentially make a model with some range of stimulus-computable generalization to new scenarios. 

**Name. Give a title for this AsPredicted pre-registration (Suggestion: use the name of the project, followed by study description.)**

Planning how to share food based on motivational state and relationship 

 **Finally. For record keeping purposes, please tell us the type of study you are pre-registering.**

1)  Class project or assignment  
   2)  **Experiment**  
   3)  Survey  
   4)  Observational/archival study  
   5)  Other:

