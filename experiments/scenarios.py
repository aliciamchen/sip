# ruff: noqa
"""Source of truth for the 16 food-sharing scenarios — the 3-action set.

This is the stimulus set for the active inverse-planning experiments (Studies 1a,
1b, 2a, 2b). Each scenario has 3 actions (no sharing / low-risk sharing / high-risk
sharing) plus separable `desire_low`/`desire_high` and `low_risk_share_effort_low`/`low_risk_share_effort_high`
paragraphs and intimacy framing, so desire, effort, and intimacy can each be
manipulated alongside the observed action.

`desire_object` is the scenario-specific object of the desire-elicitation
question "How much do you think {name_0} and {name_1} would like ___?" (e.g. "the
hot dog", "the coffee", "each other's soups"), so the question names the actual
food instead of the generic "the food". It's used in Studies 1a and 1b, where
desire is inferred.

The 3 actions per scenario are:
- `no_share` = the no-share action.
- `low_risk_share` = the low-risk (non-saliva) sharing action, written so the effort cost
  is carried by the `low_risk_share_effort_low` / `low_risk_share_effort_high` paragraph rather than baked into the
  action text.
- `high_risk_share` = the high-risk sharing action.

Edit this file to change scenarios; the CSV is a generated artifact.
Regenerate with: `uv run python experiments/scenarios.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "build"))
from _scenario_io import write_scenarios_csv

OUT = Path(__file__).resolve().parent / "scenarios.csv"

COLUMNS = [
    "scenario_label",
    "name_0",
    "name_1",
    "vignette",
    "desire_object",
    "desire_low",
    "desire_high",
    "low_risk_share_effort_low",
    "low_risk_share_effort_high",
    "no_share",
    "low_risk_share",
    "high_risk_share",
]

rows = [
    dict(
        scenario_label="basketball",
        name_0="Carissa",
        name_1="Josh",
        vignette="Carissa and Josh are attending a basketball game together. During halftime, they go to the hot-dog stand in the arena. When they get to the stand, they realize that it's cash only, and that between them they only have enough cash to get one hot dog.",
        desire_object="the hot dog",
        desire_low="Neither of them is particularly hungry.",
        desire_high="They are both very hungry.",
        low_risk_share_effort_low="Right next to the hot-dog stand is a condiment station with a stack of clean plastic knives set out for customers to use.",
        low_risk_share_effort_high="There are no knives at the hot-dog stand or the nearby condiment area — the nearest knives are at a sit-down restaurant on the far side of the arena, which would take several minutes to walk to and back.",
        no_share="They leave the stand without a hot dog and go back to the game.",
        low_risk_share="They order a hot dog. They get a knife, cut the hot dog in half, and each eats from their own portion.",
        high_risk_share="They order a hot dog and trade off taking bites directly from it, until it is finished.",
    ),
    dict(
        scenario_label="birthday",
        name_0="Lio",
        name_1="Mitchell",
        vignette="Lio and Mitchell are at a birthday party. The cake served is a dark chocolate cake with a raspberry sauce. After everyone sings happy birthday, the birthday person cuts the cake into slices and puts the slices on plates with forks. The slices are really large, and Lio and Mitchell both don't think they can finish a full slice.",
        desire_object="the cake",
        desire_low="Lio and Mitchell both feel neutral about dark chocolate raspberry cake.",
        desire_high="Lio and Mitchell both really like dark chocolate raspberry cake.",
        low_risk_share_effort_low="Extra forks and empty plates have been laid out on the cake table for anyone who wants to share a slice.",
        low_risk_share_effort_high="All the extra forks and plates have already been cleared back to the kitchen, and getting them would require walking to the kitchen and looking through the cabinets.",
        no_share="Neither Lio nor Mitchell takes a slice of cake, since neither wants to take a whole large slice they can't finish.",
        low_risk_share="They decide to share one slice. They get an extra fork and plate, cut the slice in half, put each half on its own plate, and each eat their own half.",
        high_risk_share="They decide to share one slice, and share the single fork provided to eat the slice together from the same plate.",
    ),
    dict(
        scenario_label="brunch",
        name_0="Allison",
        name_1="Dana",
        vignette="Allison and Dana meet at a brunch restaurant together on Saturday morning. They each select their own dishes and then discuss what else to order. One intriguing choice is to also order a stack of the restaurant's famous pancakes, to eat for dessert after they eat their main dishes. They decide to wait until they have finished their main dishes, to decide whether to order the pancakes.",
        desire_object="the pancakes",
        desire_low="After eating their main dishes, they are both pretty full.",
        desire_high="After eating their main dishes, they are both still hungry.",
        low_risk_share_effort_low="There is a box of extra utensils on their table.",
        low_risk_share_effort_high="To get extra utensils, they would need to walk to the utensils station on the opposite side of the restaurant.",
        no_share="They decide not to order the pancakes.",
        low_risk_share="They order the pancakes. They get extra utensils and use them to cut portions of the stack onto their own plates and eat from their own plates.",
        high_risk_share="They order the pancakes. They use their own utensils, which they have already eaten with, to eat directly from the shared stack.",
    ),
    dict(
        scenario_label="takeout",
        name_0="Emily",
        name_1="Elizabeth",
        vignette="Emily and Elizabeth are at a conference for work. After a long day of events, they get back to the conference hotel. It is late and all the nearby places are closed, so they decide to order chicken tenders delivered to the hotel. When they open the bag, they see that the restaurant has only included one container of honey mustard dipping sauce. They look in the bag and see if the restaurant has included anything else, and see that the restaurant has included several ketchup packets.",
        desire_object="the honey mustard sauce",
        desire_low="Neither Emily nor Elizabeth minds whether they use ketchup or honey mustard.",
        desire_high="Both Emily and Elizabeth prefer honey mustard sauce to ketchup.",
        low_risk_share_effort_low="The takeout came with extra sauce containers that they can use to pour out and divide the sauce.",
        low_risk_share_effort_high="The takeout didn't come with extra sauce containers or a surface they can pour the sauce onto, so to get them they need to go to the hotel lobby downstairs.",
        no_share="Emily and Elizabeth each use the ketchup packets and leave the single honey mustard container unopened.",
        low_risk_share="They get a sauce container and pour half of the honey mustard into it so that each person has their own dipping portion. Each person only dips from their own portion.",
        high_risk_share="They both dip their own chicken tenders into the single honey mustard container, double-dipping the same tenders back into the sauce after each bite.",
    ),
    dict(
        scenario_label="cooking",
        name_0="Liana",
        name_1="Serina",
        vignette="Liana and Serina are both major foodies who love cooking. They spend the evening cooking a big batch of a new pasta recipe to bring to an event the next day. While the pasta is still warm, they consider eating some for their own dinner tonight. There is a plate on the counter that they have used for placing their cooking utensils.",
        desire_object="the pasta",
        desire_low="Neither of them especially feels like having any of the pasta tonight.",
        desire_high="The pasta smells amazing and they're both tempted to have some tonight.",
        low_risk_share_effort_low="There are more clean plates and utensils in the cabinets, ready to use.",
        low_risk_share_effort_high="All the other plates and utensils are mid-cycle in the dishwasher — getting them out clean would require waiting for the cycle to finish.",
        no_share="Liana and Serina don't have any pasta themselves tonight; they keep the whole batch for the party.",
        low_risk_share="They get clean plates and extra utensils, serve some pasta from the pot onto their individual plates, and eat from their own plates using their own utensils.",
        high_risk_share="They spoon a shared portion from the pot onto the single plate on the counter, and both eat from that plate together with the same fork.",
    ),
    dict(
        scenario_label="apples",
        name_0="Camille",
        name_1="Haoyu",
        vignette="Camille and Haoyu are at an orchard for a group apple-picking outing in the fall. There are many different apple varieties at the orchard, including Red Delicious, Yellow Delicious, Honeycrisp, and Jonagold. At the entrance, the employees tell them that visitors are welcome to taste the apples. When they enter the orchard, the apple variety in the first aisle is the Jonagold variety. Camille picks an apple from a tree.",
        desire_object="the Jonagold variety",
        desire_low="Both Camille and Haoyu have tasted the Jonagold variety at this orchard before, so they are indifferent about tasting it again.",
        desire_high="Neither Camille nor Haoyu has tasted the Jonagold variety at this orchard before, so they would both like to taste it.",
        low_risk_share_effort_low="The Jonagold aisle is dense with ripe apples at eye level — Haoyu could easily reach up and pick her own separate apple from the same tree.",
        low_risk_share_effort_high="This particular tree has only the one apple Camille picked within reach; all the other Jonagolds are on trees at the far end of the orchard, a long walk away.",
        no_share="Camille takes bites out of the apple. Haoyu does not taste the apple.",
        low_risk_share="Haoyu picks a separate Jonagold apple for herself, and each of them eats their own apple.",
        high_risk_share="They share the one apple, passing it back and forth and each taking bites until it is finished.",
    ),
    dict(
        scenario_label="dip",
        name_0="Declan",
        name_1="Eric",
        vignette="Declan and Eric are preparing snacks for a house party. They get two kinds of chips, tortilla chips and pita chips, and prepare two kinds of dips: a buffalo chicken dip and a spinach and artichoke dip. Before putting the dips out for guests, they spoon a small tasting amount of each dip into a couple of small bowls to sample themselves.",
        desire_object="the dips",
        desire_low="Neither of them thinks it's particularly important to taste the dips before the party.",
        desire_high="They both really want to taste the dips before the party.",
        low_risk_share_effort_low="There is a stack of clean small appetizer plates already unpacked and sitting on the snack table, ready to use.",
        low_risk_share_effort_high="The appetizer plates are still packed away in a box of party supplies that hasn't been unboxed yet, buried in the garage under other party boxes — digging them out would take a while.",
        no_share="Neither of them tastes the dips before the party; they put everything out for guests as-is.",
        low_risk_share="They get two small appetizer plates, divide the tasting portions between the two plates, and each tries the dips with chips from their own plate.",
        high_risk_share="They leave the tasting portions in the shared bowls and dip the same chips into them, each taking bites and double-dipping them back in.",
    ),
    dict(
        scenario_label="drinks",
        name_0="Noah",
        name_1="Ria",
        vignette="Noah and Ria are at a party. At the bar there is a one-off special cocktail the bartender invented — tequila mixed with ice cream — and only one serving is left.",
        desire_object="the cocktail",
        desire_low="Neither of them is especially interested in the cocktail.",
        desire_high="They are both very interested in the cocktail.",
        low_risk_share_effort_low="There is a stack of clean empty cups right at the bar within arm's reach.",
        low_risk_share_effort_high="The bartender has stepped away to the back, and the extra clean cups are on a shelf behind the bar that's inaccessible without flagging someone down — which would take a long time given how packed the party is.",
        no_share="Neither of them drinks the special cocktail; they each get a regular drink from the bar instead.",
        low_risk_share="They get a second clean cup, pour half of the cocktail into it, and each sip from their own cup.",
        high_risk_share="They share the single cup of cocktail, passing it back and forth and sipping from the same rim.",
    ),
    dict(
        scenario_label="driving",
        name_0="Danielle",
        name_1="Katherine",
        vignette="Danielle and Katherine are driving together to an event a few hours away, leaving early in the morning. On their way out of town, they stop at a gas station to get coffee. The coffee machine is out of order partway through pouring, so only one cup gets filled.",
        desire_object="the coffee",
        desire_low="Both of them got enough sleep last night, so they aren't particularly tired.",
        desire_high="Both of them are extremely tired, and need the coffee to be able to stay awake.",
        low_risk_share_effort_low="Right next to the coffee machine at this gas station is a self-serve stack of clean empty cups that customers can grab from.",
        low_risk_share_effort_high="There are no cups out at the coffee machine; getting a second cup would require asking at the counter, and the line to pay is long.",
        no_share="They do not share the coffee. Danielle, who is driving the car, drinks the coffee.",
        low_risk_share="They get another cup, pour half of the coffee into the other cup, and each drink from their own cup.",
        high_risk_share="They share the coffee from the single cup, passing it back and forth.",
    ),
    dict(
        scenario_label="fair",
        name_0="Marianne",
        name_1="Lisa",
        vignette="Marianne and Lisa are at the county fair. They've spent a day walking around. Around the time the fair is closing, the food trucks are giving out their leftover food. They both go to a food truck that sells fresh corn on the cob. The food truck gives them a large skewer of corn. They then go to some other food trucks, and get some funnel cake and also some fried cheese. They put all the food on a plate, and need to decide how to share it.",
        desire_object="the corn",
        desire_low="The corn looks okay, and they're more excited about the other food that they got.",
        desire_high="The corn looks especially appealing to both of them, even more so than the other food.",
        low_risk_share_effort_low="There is a cutlery station nearby with extra knives, plates, and napkins for visitors to use.",
        low_risk_share_effort_high="The food truck has packed up its cutlery station, and if they want cutlery, they need to go to the cutlery station on the other side of the fair.",
        no_share="They eat the food on their sides of the plate, and don't share the individual items of food. The corn is on Marianne's side of the plate, so she eats the corn.",
        low_risk_share="They get cutlery and a plate, and use the knife to cut the corn off the cob onto the plate. They each eat the corn from the plate with their own forks.",
        high_risk_share="They switch off taking bites from the cob.",
    ),
    dict(
        scenario_label="gala",
        name_0="Elena",
        name_1="Todd",
        vignette="Elena and Todd are at a fancy gala. Elena is deciding what to order, and asks Todd what he ordered. Todd said that he ordered a pumpkin spice martini and that the bar offers many different types of interesting seasonal drinks. Intrigued, Elena decides to order an apple pie espresso martini.",
        desire_object="each other's drinks",
        desire_low="When the drinks come, Elena and Todd do not particularly want to try each other's drinks.",
        desire_high="When the drinks come, Elena and Todd want to try each other's drinks.",
        low_risk_share_effort_low="There is a dispenser with extra clean straws sitting right on the bar that anyone can grab from.",
        low_risk_share_effort_high="The straw dispenser near them is out of straws, and the servers are deep in the banquet service — flagging one down to bring extra straws would take a long time.",
        no_share="They only drink from their own drinks, and do not try each other's drinks.",
        low_risk_share="They get two extra clean straws and use them to try each other's drinks, before drinking their own drinks directly from the rim of the glass.",
        high_risk_share="They try each other's drinks by drinking directly from the rims of the glass.",
    ),
    dict(
        scenario_label="hike",
        name_0="Tony",
        name_1="Alvin",
        vignette="Tony and Alvin go on a day hike in New Hampshire. Alvin packs snacks and energy bars, while Tony brings peanut butter and jelly sandwiches. Halfway down the mountain, they take a snack break. Alvin realizes that he has run out of his food. Tony pulls out a sandwich from his pack.",
        desire_object="the sandwich",
        desire_low="Neither of them is very hungry, and the hike is almost over.",
        desire_high="They are both tired and hungry.",
        low_risk_share_effort_low="Tony's backpacking knife is in the top lid of his pack and is easy to grab.",
        low_risk_share_effort_high="Tony's backpacking knife is buried at the very bottom of his pack, and grabbing it would require taking out all of his gear, looking for the knife, and re-packing everything.",
        no_share="Tony and Alvin do not share the sandwich; Tony eats the sandwich himself.",
        low_risk_share="Tony gets his knife from the pack, uses it to cut the sandwich in half, and hands one half to Alvin.",
        high_risk_share="Tony and Alvin alternate taking bites directly from the sandwich.",
    ),
    dict(
        scenario_label="oysters",
        name_0="Will",
        name_1="Jay",
        vignette="Will and Jay are at a seafood restaurant. They are interested in ordering the oyster platter. The server mentions that the restaurant has different types of oysters in stock, each type with unique flavors and notes. They decide to order one of each type of oyster.",
        desire_object="all the oyster types",
        desire_low="They are both indifferent to tasting all the different oyster types.",
        desire_high="They both would like to taste as many different oyster types as possible.",
        low_risk_share_effort_low="The restaurant is quiet tonight and the server is attentive — they expect that extra small plates and forks, for splitting the oysters, will come immediately whenever they ask.",
        low_risk_share_effort_high="The restaurant is packed and the server is slammed — flagging him down for extra small plates and forks, for splitting the oysters, would take a long time.",
        no_share="They only eat the oysters (the meat and the brine) on their own sides of the platter. They do not share the individual oysters.",
        low_risk_share="They ask the server for two extra small plates and cocktail forks. They use the forks to split each oyster's meat between the plates, and each eats half of each oyster from their own plate.",
        high_risk_share="For each oyster, they split it directly from the shell — each biting off half of the meat with their teeth and drinking some of the brine from the same shell, before passing it to the other person.",
    ),
    dict(
        scenario_label="social",
        name_0="Sonia",
        name_1="Alan",
        vignette="Sonia and Alan arrive late to an ice cream social hosted by their religious organization. Unfortunately, there is only one ice cream cone left. There are other kinds of desserts available, however, including many flavors of cookies.",
        desire_object="the ice cream",
        desire_low="Neither of them particularly wants ice cream.",
        desire_high="They both really want ice cream.",
        low_risk_share_effort_low="There is a stack of clean spoons on the dessert table.",
        low_risk_share_effort_high="The dessert table has been mostly cleared and the spoons have been put away in the kitchen, so getting clean spoons would mean interrupting the hosts who are busy with other cleanup.",
        no_share="Neither of them takes the ice cream cone. They instead go and eat cookies.",
        low_risk_share="They get two spoons and each uses their own spoon to eat the ice cream out of the cone.",
        high_risk_share="They pass the cone back and forth, taking turns licking the ice cream and biting into the cone.",
    ),
    dict(
        scenario_label="soup",
        name_0="Christina",
        name_1="Tanya",
        vignette="Christina and Tanya are having dinner at a restaurant. They each order a soup as a starter. Christina orders the chicken noodle soup, and Tanya orders the lentil soup.",
        desire_object="each other's soups",
        desire_low="Christina and Tanya are satisfied with their own soups.",
        desire_high="Christina and Tanya want to try each other's soups.",
        low_risk_share_effort_low="The restaurant is quiet and the server is attentive — extra bowls would come to the table quickly if they asked.",
        low_risk_share_effort_high="The restaurant is packed and the server is slammed — asking for extra bowls would mean a long wait.",
        no_share="They eat only their own soups.",
        low_risk_share="They ask for two extra bowls, spoon some of their own soup into an extra bowl, and swap the extra bowls.",
        high_risk_share="They put both bowls in the middle of the table and eat directly from each other's bowls, going back and forth between the two bowls.",
    ),
    dict(
        scenario_label="wedding",
        name_0="Maxwell",
        name_1="Ralph",
        vignette="Maxwell and Ralph are seated together at a wedding. After the appetizer, there are two dishes that guests can choose from for the main course. One dish is a mushroom risotto and the other dish is a coconut curry salmon. Both dishes look really good, so they both have a hard time selecting what to order. Maxwell ends up getting the mushroom risotto, and Ralph ends up getting the coconut curry salmon.",
        desire_object="both dishes",
        desire_low="When the food comes, they are actually really happy with what they ended up selecting. Maxwell and Ralph each feel that their own dish looks better than the other one.",
        desire_high="When the food comes, they realize how good both dishes look, and that they really want to taste both dishes.",
        low_risk_share_effort_low="The catering staff has brought a tray of extra utensils to the table for guests.",
        low_risk_share_effort_high="The catering staff is deep in the next course of service, and there are no extra utensils on the table — flagging a server down for new utensils would mean a long wait.",
        no_share="Maxwell eats his mushroom risotto, and Ralph eats his coconut curry salmon. They do not share their food.",
        low_risk_share="They get an extra set of utensils, and use them to place some of each dish onto the other's plate. They then eat from their own plates using their original utensils.",
        high_risk_share="They eat from each other's plates throughout the meal, going back and forth between their own plate and the other's plate with their own forks.",
    ),
]

write_scenarios_csv(rows, COLUMNS, OUT, expected_rows=16)
