# ruff: noqa
"""Source of truth for the 16 effort-manipulation food-sharing scenarios.

Parallels `scenarios.py` (same 16 scenario labels and character names) but collapses
the action space to two actions (`action_1` non-saliva-sharing, `action_2` saliva-sharing)
and manipulates effort (how easy it is to avoid saliva sharing) via a trailing
paragraph. Each scenario has one shared `vignette` followed by one of two
effort-manipulation paragraphs, `effort_low` or `effort_high`. Reward is held fixed
at high and integrated into the shared vignette.

Edit this file to change scenarios; the CSV is a generated artifact.
Regenerate with: `uv run python experiments/scenarios_effort.py`
"""

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent / "scenarios_effort.csv"

COLUMNS = [
    "scenario_label",
    "name_0",
    "name_1",
    "vignette",
    "effort_low",
    "effort_high",
    "action_1",
    "action_2",
]

rows = [
    dict(
        scenario_label="basketball",
        name_0="Carissa",
        name_1="Josh",
        vignette="Carissa and Josh are attending a basketball game together. During halftime, they go to the hot-dog stand in the arena. They are both very hungry. When they get to the stand, they realize that it's cash only, and that between them they only have enough cash to get one hot dog.",
        effort_low="Right next to the hot-dog stand is a condiment station with a stack of clean plastic knives set out for customers to use.",
        effort_high="There are no knives at the hot-dog stand or the nearby condiment area — the nearest knives are at a sit-down restaurant on the far side of the arena, which would take several minutes to walk to and back.",
        action_1="They order a hot dog. They get a knife, cut the hot dog in half, and each eats from their own portion.",
        action_2="They order a hot dog and trade off taking bites from it until it is finished.",
    ),
    dict(
        scenario_label="birthday",
        name_0="Lio",
        name_1="Mitchell",
        vignette="Lio and Mitchell are at a birthday party. The cake served is a dark chocolate cake with a raspberry sauce, which they both really like. After everyone sings happy birthday, the birthday person cuts the cake into slices and puts the slices on plates with forks. The slices are really large, and Lio and Mitchell both don't think they can finish a full slice on their own.",
        effort_low="Extra forks and empty plates have been laid out on the cake table for anyone who wants to share a slice.",
        effort_high="All the extra forks and plates have already been cleared back to the kitchen.",
        action_1="They decide to share one slice. They get an extra fork and plate, cut the slice in half, put each half on its own plate, and each eat their own half.",
        action_2="They decide to share one slice, and share the single fork provided to eat the slice together from the same plate.",
    ),
    dict(
        scenario_label="brunch",
        name_0="Allison",
        name_1="Dana",
        vignette="Allison and Dana meet at a brunch restaurant together on Saturday morning. They each select their own dishes and then discuss what else to order. One intriguing choice is to also order a stack of the restaurant's famous pancakes to share, to eat for dessert after they eat their main dishes. After eating their main dishes, they are both still hungry and order the pancakes.",
        effort_low="There is a box of extra utensils on each person's table.",
        effort_high="There is a station of extra utensils on the other side of the restaurant.",
        action_1="They get extra utensils and use them to cut portions of the stack onto their own plates and eat from their own plates.",
        action_2="They eat directly from the shared stack using their own forks.",
    ),
    dict(
        scenario_label="takeout",
        name_0="Emily",
        name_1="Elizabeth",
        vignette="Emily and Elizabeth are at a conference for work. After a long day of events, they get back to the conference hotel. It is late and all the nearby places are closed, so they decide to order chicken tenders delivered to the hotel. Both of them prefer honey mustard sauce to ketchup, but when they open the bag they see that the restaurant has only included one container of honey mustard.",
        effort_low="The takeout came with extra sauce containers that they can use to pour out the sauce.",
        effort_high="The takeout didn't come with extra sauce containers or a surface they can pour the sauce onto, so to get them they need to go to the hotel restaurant downstairs.",
        action_1="They get a sauce container and pour half of the honey mustard into it so that each person has their own dipping portion. Each person only dips from their own portion.",
        action_2="They both dip their own chicken tenders into the single honey mustard container, double-dipping the same tenders back into the sauce after each bite.",
    ),
    dict(
        scenario_label="cooking",
        name_0="Liana",
        name_1="Serina",
        vignette="Liana and Serina are both major foodies who love cooking. They discover a new pasta recipe they're both excited to try, so they go to Serina's house in the evening to cook it together. They plan to enjoy their dinner while watching an episode of a TV show in the living room. By the time the pasta is done, it looks so good that they both want to dig in immediately.",
        effort_low="There are clean plates and a clean serving spoon in the cabinets, ready to use.",
        effort_high="All the plates and the serving spoon are mid-cycle in the dishwasher — getting them out clean would require waiting for the cycle to finish.",
        action_1="They get clean plates and a serving spoon, use the spoon to transfer pasta from the pot onto their individual plates, and carry their plates to the living room to eat with their own forks.",
        action_2="They bring the pot to the living room and eat the pasta together directly from the shared pot.",
    ),
    dict(
        scenario_label="apples",
        name_0="Camille",
        name_1="Haoyu",
        vignette="Camille and Haoyu go apple picking together in the fall. At the orchard, there are many different apple varieties, including Red Delicious, Yellow Delicious, Honeycrisp, and Jonagold. At the entrance, the employees tell them that visitors are welcome to taste the apples. When they enter the orchard, the apple variety in the first aisle is the Jonagold variety. Camille picks an apple from a tree to taste. Haoyu has not tasted the Jonagold variety at this orchard before and really wants to taste it.",
        effort_low="The Jonagold aisle is dense with ripe apples at eye level — Haoyu could easily reach up and pick her own separate apple from the same tree.",
        effort_high="This particular tree has only the one apple Camille picked within reach; all the other Jonagolds are on trees at the far end of the orchard, a long walk away.",
        action_1="Haoyu picks a separate Jonagold apple for herself, and each of them eats their own apple.",
        action_2="Camille eats some of the apple and then passes it to Haoyu, who eats the remaining portion.",
    ),
    dict(
        scenario_label="dip",
        name_0="Declan",
        name_1="Eric",
        vignette="Declan and Eric are preparing snacks for a house party. They get two kinds of chips — tortilla chips and pita chips — and prepare two kinds of dips: a buffalo chicken dip and a spinach and artichoke dip. They both want to try the chips with the different kinds of dips before serving them to their guests.",
        effort_low="There is a stack of clean small appetizer plates already unpacked and sitting on the snack table, ready to use.",
        effort_high="The appetizer plates are still packed away in a box of party supplies that hasn't been unboxed yet, buried in the garage under other party boxes — digging them out would take a while.",
        action_1="They get two small appetizer plates, put some of each dip on each plate, and try the dips with chips from their own plates.",
        action_2="Declan dips the chips into the dips, takes a bite from each to test the dip, then passes the rest of the chip with dip to Eric to finish.",
    ),
    dict(
        scenario_label="drinks",
        name_0="Noah",
        name_1="Ria",
        vignette="Noah and Ria are at a party. Noah is holding a drink that he tells Ria is a special cocktail that he invented, that he made himself at the bar. (It's tequila mixed with ice cream.) He tells Ria how good it is and suggests she try it. The drink sounds good to Ria, and she really wants to try it.",
        effort_low="There is a stack of clean empty cups right at the bar within arm's reach.",
        effort_high="The bartender has stepped away to the back, and the clean cups are on a shelf behind the bar that's inaccessible without flagging someone down — which would take a long time given how packed the party is.",
        action_1="Ria gets another cup and Noah pours some of his drink into that cup. They each sip from their own cups.",
        action_2="Ria takes Noah's cup and sips directly from his cup.",
    ),
    dict(
        scenario_label="driving",
        name_0="Danielle",
        name_1="Sonia",
        vignette="Danielle and Sonia are driving together from San Francisco to Sacramento. They leave at 6am. Right when they leave, they stop at a gas station to get an iced coffee. The coffee machine is out of order partway through pouring, so only one cup gets filled. Both of them are extremely tired and need the coffee to be able to stay awake.",
        effort_low="Next to the coffee machine at this gas station is a self-serve stack of clean empty cups that customers can grab from.",
        effort_high="There are no cups out at the coffee machine; getting a second cup would require asking at the counter, and the line to pay is long.",
        action_1="They get another cup, pour half of the coffee into the other cup, and each drink from their own cup.",
        action_2="They share the coffee from the same cup, passing it back and forth.",
    ),
    dict(
        scenario_label="fair",
        name_0="Marianne",
        name_1="Lisa",
        vignette="Marianne and Lisa are at the county fair. They've spent a day walking around. Around the time the fair is closing, the food trucks are giving out their leftover food. They both go to a food truck that sells fresh corn on the cob, and the food truck gives them a large skewer of corn. The corn looks especially appealing to both of them.",
        effort_low="There is a cutlery station nearby with knives, plates, and napkins for guests to use.",
        effort_high="The food truck has packed up its cutlery station, and they need to search for cutlery at the other food trucks.",
        action_1="They find cutlery and a plate, and use the knife to cut the corn off the cob onto the plate. They each eat the corn from the plate with their own forks.",
        action_2="They switch off taking bites from the cob.",
    ),
    dict(
        scenario_label="gala",
        name_0="Elena",
        name_1="Todd",
        vignette="Elena and Todd are at a fancy gala. Elena is deciding what to order, and asks Todd what he ordered. Todd said that he ordered a pumpkin spice martini, and that the bar offers many different types of interesting seasonal drinks. Intrigued, Elena decides to order an apple pie espresso martini. When the drinks come, Elena and Todd really want to try each other's drinks.",
        effort_low="There is a dispenser with clean straws sitting right on the bar that anyone can grab from.",
        effort_high="The bar is out of straws, and the servers are deep in the banquet service — flagging one down to bring straws would take a long time.",
        action_1="They get two clean straws and use them to try each other's drinks, before drinking their own drinks directly from the rim of the glass.",
        action_2="They try each other's drinks by drinking directly from the rims of the glass.",
    ),
    dict(
        scenario_label="hike",
        name_0="Tony",
        name_1="Alvin",
        vignette="Tony and Alvin go on a day hike in New Hampshire. Alvin packs snacks and energy bars, while Tony brings peanut butter and jelly sandwiches. Halfway down the mountain, they take a snack break. Tony pulls out a sandwich and offers to share some of the sandwich with Alvin. At this time, they are both tired and hungry.",
        effort_low="Tony's backpacking knife is in the top lid of his pack and is easy to grab.",
        effort_high="Tony's backpacking knife is buried at the very bottom of his pack, under all of his gear.",
        action_1="Tony gets his knife from the pack, uses it to cut the sandwich in half, and hands one half to Alvin.",
        action_2="Tony eats half of the sandwich and then hands the other half to Alvin, who eats the remaining portion.",
    ),
    dict(
        scenario_label="oysters",
        name_0="Will",
        name_1="Jay",
        vignette="Will and Jay are at a seafood restaurant. They are interested in ordering the oyster platter. The server mentions that the restaurant has different types of oysters in stock, each type with unique flavors and notes. They decide to order one of each type of oyster, because they both really want to taste as many different oyster types as possible.",
        effort_low="The restaurant is quiet tonight and the server is attentive — they expect that extra small plates will come immediately whenever they ask.",
        effort_high="The restaurant is packed and the server is slammed — flagging him down for extra small plates and forks would take a long time.",
        action_1="They ask the server for two extra small plates and cocktail forks. They use the forks to split each oyster's meat between the plates, and each eats half of each oyster from their own plate.",
        action_2="For each oyster, they split it directly from the shell — each biting off half of the meat with their teeth and drinking some of the brine from the same shell.",
    ),
    dict(
        scenario_label="social",
        name_0="Sonia",
        name_1="Alan",
        vignette="Sonia and Alan arrive late to an ice cream social hosted by their religious organization. Unfortunately, there is only one ice cream cone left. There are other kinds of desserts available, however, including many flavors of cookies. However, they both really want ice cream.",
        effort_low="There is a stack of clean spoons on the dessert table right next to the ice cream bowls.",
        effort_high="The dessert table has been mostly cleared and the spoons have been put away in the kitchen, so getting a spoon would mean interrupting the hosts who are busy with other cleanup.",
        action_1="They get two spoons and each uses their own spoon to eat the ice cream out of the cone.",
        action_2="They pass the cone back and forth, taking turns licking the ice cream and biting into the cone.",
    ),
    dict(
        scenario_label="soup",
        name_0="Christina",
        name_1="Tanya",
        vignette="Christina and Tanya are having dinner at a restaurant. They each order a soup as a starter. Christina orders the chicken noodle soup, and Tanya orders the lentil soup. After they have tasted their own soup, each remarks that the other's choice looks especially good, and they both really want to taste the other's soup.",
        effort_low="The restaurant is quiet and the server is attentive — extra bowls would come to the table quickly if they asked.",
        effort_high="The restaurant is packed and the server is slammed — asking for extra bowls would mean a long wait.",
        action_1="They ask for two extra bowls, spoon some of their own soup into an extra bowl, and swap the extra bowls.",
        action_2="They put both bowls in the middle of the table and use their own spoons to eat from each other's bowls, going back and forth between the two bowls.",
    ),
    dict(
        scenario_label="wedding",
        name_0="Maxwell",
        name_1="Ralph",
        vignette="Maxwell and Ralph are seated together at a wedding. After the appetizer, there are two dishes that guests can choose from for the main course. One dish is a mushroom risotto and the other dish is a coconut curry salmon. Both dishes look really good, so they both have a hard time selecting what to order. Maxwell ends up getting the mushroom risotto, and Ralph ends up getting the coconut curry salmon. When the food comes, they realize how good both dishes look, and they really want to taste both dishes.",
        effort_low="The catering staff has brought a tray of extra utensils to the table for guests.",
        effort_high="The catering staff is deep in the next course of service, and there are no extra utensils on the table — flagging a server down for new utensils would mean a long wait.",
        action_1="They request an extra set of utensils, and use them to place some of each dish onto the other's plate. They then eat from their own plates using their original utensils.",
        action_2="They eat from each other's plates throughout the meal, going back and forth between their own plate and the other's plate with their own forks.",
    ),
]

assert len(rows) == 16, f"expected 16 rows, got {len(rows)}"
labels = [r["scenario_label"] for r in rows]
assert len(set(labels)) == 16, f"duplicate labels: {labels}"

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")
