# Substrate and scenario details

## Details of specific substrates

## Allelopathic Harvest.

[![Substrate video](https://img.youtube.com/vi/ESugMMdKLxI/0.jpg)](https://youtu.be/ESugMMdKLxI)

This substrate contains three different varieties of berry (red, green, & blue)
and a fixed number of berry patches, which could be replanted to grow any color
variety of berry. The growth rate of each berry variety depends linearly on the
fraction that that color comprises of the total. Players have three planting
actions with which they can replant berries in their chosen color. All players
prefer to eat red berries (reward of 2 per red berry they eat versus a reward of
1 per other colored berry). Players can achieve higher return by selecting just
one single color of berry to plant, but which one to pick is, in principle,
difficult to coordinate (start-up problem) -- though in this case all prefer red
berries, suggesting a globally rational chioce. They also always prefer to eat
berries over spending time planting (free-rider problem).

Allelopathic Harvest was first described in Koster et al. (2020).

Köster, R., McKee, K.R., Everett, R., Weidinger, L., Isaac, W.S., Hughes, E.,
Duenez-Guzman, E.A., Graepel, T., Botvinick, M. and Leibo, J.Z., 2020.
Model-free conventions in multi-agent reinforcement learning with heterogeneous
preferences. arXiv preprint arXiv:2010.09054.

## Arena Running with Scissors in the Matrix.

[![Substrate video](https://img.youtube.com/vi/esXPyGBIf2Y/0.jpg)](https://youtu.be/esXPyGBIf2Y)

This substrate is the same as _Running with Scissors in the Matrix_ except in
this case there are eight players and the map layout is different. Even though
there are eight players, they still interact in dyadic pairs via the usual
rock-paper-scissors payoff matrix.

Players have the default `11 x 11` (off center) observation window.

## Bach or Stravinsky in the Matrix.

[![Substrate video](https://img.youtube.com/vi/SiFjSyCp2Ss/0.jpg)](https://youtu.be/SiFjSyCp2Ss)

See _Running with Scissors in the Matrix_ for a general description of the game
dynamics. Here the payoff matrix represents the Bach or Stravinsky (battle of
the sexes) game. `K = 2` resources represent "Bach" and "Stravinsky" pure
strategies.

Bach or Stravinsky is an asymmetric game. Players are assigned by their slot id
to be either row players (blue) or column players (orange). Interactions are
only resolved when they are between a row player and a column player. Otherwise,
e.g. when a row player tries to interact with another row player, then nothing
happens.

Players have the default `11 x 11` (off center) observation window.

## Capture the Flag.

[![Substrate video](https://img.youtube.com/vi/VRNt55-0IqE/0.jpg)](https://youtu.be/VRNt55-0IqE)

This substrate a team based zero sum game. There are four players on each team.

There is a red team and blue team. Players can paint the ground anywhere by
using their zapping beam. If they stand on their own color then they gain health
up to a maximum of 3 (so they are more likely to win shootouts). They lose
health down to 1 from their default of 2 when standing on the opposing team's
color (so they are more likely to lose shootouts in that case). Health recovers
stochastically, at a fixed rate of 0.05 per frame. It cannot exceed its maximum,
determined by the current color of the ground the agent is standing on.

Players also cannot move over their opposing team's color. If the opposing team
paints the square underneath their feet then they get stuck in place until they
use their own zapping beam to re-paint the square underneath and in front of
themselves to break free. In practice this slows them down by one frame (which
may be critical if they are being chased).

Friendly fire is impossible; agents cannot zap their teammates.

In the _Capture the Flag_ substrate the final goal is capturing the opposing
team's flag. Payoffs are common to the entire winning team. Indicator tiles
around the edge of the map and in its very center display which teams have their
own flag on their base, allowing them the possibility of capturing their
opponent's flag by bringing it to their own base/flag. When indicator tiles are
red then only the red team can score. When indicator tiles are blue then only
the blue team can score. When the indicator tiles are purple then both teams
have the possibility of scoring (though neither is close to doing so) since both
flags are in their respective home bases.

## Chemistry: Branched Chain Reaction.

[![Substrate video](https://img.youtube.com/vi/ZhRB-_ruoH8/0.jpg)](https://youtu.be/ZhRB-_ruoH8)

Individuals are rewarded by driving chemical reactions involving specific
molecules. They need to suitably coordinate the alternation of branches while
keeping certain elements apart that would otherwise react unfavourably, so as
not to run out of the molecules required for continuing the chain. Combining
molecules efficiently requires coordination but can also lead to exclusion of
players.

Reactions are defined by a directed graph. Reactant nodes project into reaction
nodes, which project out to product nodes. Reactions occur stochastically when
all reactants are brought near one another. Agents can carry a single molecule
around the map with them at a time. Agents are rewarded when a specific reaction
occurs that involves the molecule they are currently carrying (as either a
reactant or a product).

## Chemistry: Metabolic Cycles.

[![Substrate video](https://img.youtube.com/vi/oFK9VujhpeI/0.jpg)](https://youtu.be/oFK9VujhpeI)

Individuals benefit from two different food generating reaction cycles. Both
will run on their own (autocatalytically), but require energy to continue.
Bringing together side products from both cycles generates new energy such that
the cycles can continue. The population needs to keep both cycles running to get
high rewards.

Reactions are defined by a directed graph. Reactant nodes project into reaction
nodes, which project out to product nodes. Reactions occur stochastically when
all reactants are brought near one another. Agents can carry a single molecule
around the map with them at a time. Agents are rewarded when a specific reaction
occurs that involves the molecule they are currently carrying (as either a
reactant or a product).

## Chicken (hawk - dove) in the Matrix.

[![Substrate video](https://img.youtube.com/vi/uhAb2busSDY/0.jpg)](https://youtu.be/uhAb2busSDY)

See _Running with Scissors in the Matrix_ for a general description of the game
dynamics. Here the payoff matrix represents the Chicken game. `K = 2` resources
represent "hawk" and "dove" pure strategies.

Players have the default `11 x 11` (off center) observation window.

## Clean Up.

[![Substrate video](https://img.youtube.com/vi/jOeIunFtTS0/0.jpg)](https://youtu.be/jOeIunFtTS0)

Clean Up is a seven player game. Players are rewarded for collecting apples. In
Clean Up, apples grow in an orchard at a rate inversely related to the
cleanliness of a nearby river. The river accumulates pollution at a constant
rate. The apple growth rate in the orchard drops to zero once the pollution
accumulates past a threshold value. Players have an additional action allowing
them to clean a small amount of pollution from the river in front of themselves.
They must physically leave the apple orchard to clean the river. Thus, players
must maintain a public good of high orchard regrowth rate through effortful
contributions. This is a public good provision problem because the benefit of a
healthy orchard is shared by all, but the costs incurred to ensure it exists are
born by individuals.

Players are also able to zap others with a beam that removes any player hit by
it from the game for 50 steps.

Clean Up was first described in Hughes et al. (2018).

Hughes, E., Leibo, J.Z., Phillips, M., Tuyls, K., Duenez-Guzman, E., Castaneda,
A.G., Dunning, I., Zhu, T., McKee, K., Koster, R. and Roff, H., 2018, Inequity
aversion improves cooperation in intertemporal social dilemmas. In Proceedings
of the 32nd International Conference on Neural Information Processing Systems
(pp. 3330-3340).

## Collaborative Cooking: Impassable.

[![Substrate video](https://img.youtube.com/vi/yn1uSNymQ_U/0.jpg)](https://youtu.be/yn1uSNymQ_U)

A pure common interest cooking game inspired by Carroll et al. (2019).

Players need to collaborate to follow recipes. They are separated by an
impassable kitchen counter, so no player can complete the objective alone.

The recipe they must follow is for tomato soup: 1. Add three tomatoes to the
cooking pot. 2. Wait for the soup to cook (status bar completion). 3. Bring a
bowl to the pot and pour the soup from the pot into the bowl. 4. Deliver the
bowl of soup at the goal location.

This substrate is a pure common interest game. All players share all rewards.

Players have a `5 x 5` observation window.

Carroll, M., Shah, R., Ho, M.K., Griffiths, T.L., Seshia, S.A., Abbeel, P. and
Dragan, A., 2019. On the utility of learning about humans for human-AI
coordination. arXiv preprint arXiv:1910.05789.

## Collaborative Cooking: Passable.

[![Substrate video](https://img.youtube.com/vi/R_TBitc3hto/0.jpg)](https://youtu.be/R_TBitc3hto)

Same as _Collaborative Cooking: Impassable_ except here players can pass each
other in the kitchen, allowing less coordinated yet inefficient strategies by
individual players.

See _Collaborative Cooking: Impassable_ for the recipe that players must follow.

This substrate is a pure common interest game. All players share all rewards.

Players have a `5 x 5` observation window.

## Commons Harvest: Closed.

[![Substrate video](https://img.youtube.com/vi/ZHjrlTft98M/0.jpg)](https://youtu.be/ZHjrlTft98M)

See _Commons Harvest: Open_ for the general description of the mechanics at play
in this substrate.

In the case of _Commons Harvest: Closed, agents can learn to defend naturally
enclosed regions. Once they have done that then they have an incentive to avoid
overharvesting the patches within their region. It is usually much easier to
learn sustainable strategies here than it is in _Commons Harvest: Open_.
However, they usually involve significant inequality since many agents are
excluded from any natural region.

## Commons Harvest: Open.

[![Substrate video](https://img.youtube.com/vi/ZwQaUj8GS6U/0.jpg)](https://youtu.be/ZwQaUj8GS6U)

Apples are spread around the map and can be consumed for a reward of 1. Apples
that have been consumed regrow with a per-step probability that depends on the
number of uneaten apples in a `L2` norm neighborhood of radius 2 (by default).
After an apple has been eaten and thus removed, its regrowth probability depends
on the number of uneaten apples still in its local neighborhood. With standard
parameters, it the grown rate decreases as the number of uneaten apples in the
neighborhood decreases and when there are zero uneaten apples in the
neighborhood then the regrowth rate is zero. As a consequence, a patch of apples
that collectively doesn't have any nearby apples, can be irrevocably lost if all
apples in the patch are consumed. Therefore, agents must exercise restraint when
consuming apples within a patch. Notice that in a single agent situation, there
is no incentive to collect the last apple in a patch (except near the end of the
episode). However, in a multi-agent situation, there is an incentive for any
agent to consume the last apple rather than risk another agent consuming it.
This creates a tragedy of the commons from which the substrate derives its name.

This mechanism was first described in Janssen et al (2010) and adapted for
multi-agent reinforcement learning in Perolat et al (2017).

Janssen, M.A., Holahan, R., Lee, A. and Ostrom, E., 2010. Lab experiments for
the study of social-ecological systems. Science, 328(5978), pp.613-617.

Perolat, J., Leibo, J.Z., Zambaldi, V., Beattie, C., Tuyls, K. and Graepel, T.,
2017. A multi-agent reinforcement learning model of common-pool resource
appropriation. In Proceedings of the 31st International Conference on Neural
Information Processing Systems (pp. 3646-3655).

## Commons Harvest: Partnership.

[![Substrate video](https://img.youtube.com/vi/ODgPnxC7yYA/0.jpg)](https://youtu.be/ODgPnxC7yYA)

See _Commons Harvest: Open_ for the general description of the mechanics at play
in this substrate.

This substrate is similar to _Commons Harvest: Closed_, except that it now
requires two players to work together to defend a room of apples (there are two
entrance corridors to defend). It requires effective cooperation both to defend
the doors and to avoid over-harvesting. It can be seen as a test of whether or
not agents can learn to trust their partners to (a) defend their shared
territory from invasion, and (b) act sustainably with regard to their shared
resources. This is the kind of trust born of mutual self interest. To be
successful, agents must recognize the alignment of their interests with those of
their partner and act accordingly.

## King of the Hill.

[![Substrate video](https://img.youtube.com/vi/DmO2uqGBPco/0.jpg)](https://youtu.be/DmO2uqGBPco)

See _Capture the Flag_ for the description of the painting, zapping, and
movement mechanics, which also operate in this substrate.

In the King of the Hill substrate the goal is to control the hill region in the
center of the map. The hill is considered to be controlled by a team if at least
80% of it has been colored in one team's color. The status of the hill is
indicated by indicator tiles around the map an in the center. Red indicator
tiles mean the red team is in control. Blue indicator tiles mean the blue team
is in control. Purple indicator tiles mean no team is in control.

## Prisoner's Dilemma in the Matrix.

[![Substrate video](https://img.youtube.com/vi/bQkEKc1zNuE/0.jpg)](https://youtu.be/bQkEKc1zNuE)

See _Running with Scissors in the Matrix_ for a general description of the game
dynamics. Here the payoff matrix represents the Prisoner's Dilemma game. `K = 2`
resources represent "cooperate" and "defect" pure strategies.

Players have the default `11 x 11` (off center) observation window.

## Pure Coordination in the Matrix.

[![Substrate video](https://img.youtube.com/vi/5G9M7rGI68I/0.jpg)](https://youtu.be/5G9M7rGI68I)

See _Running with Scissors in the Matrix_ for a general description of the game
dynamics. Here the payoff matrix represents a pure coordination game with `K =
3` different ways to coordinate, all equally beneficial.

Players have the default `11 x 11` (off center) observation window.

Both players are removed and their inventories are reset after each interaction.

## Rationalizable Coordination in the Matrix.

[![Substrate video](https://img.youtube.com/vi/BpHpoir06mY/0.jpg)](https://youtu.be/BpHpoir06mY)

See _Running with Scissors in the Matrix_ for a general description of the game
dynamics. Here the payoff matrix represents a coordination game with `K = 3`
different ways to coordinate. Coordinating on one of the three options yields a
reward of 1 for both players, another yields a reward of 2, and the third yields
a reward of 3.

Players have the default `11 x 11` (off center) observation window.

Both players are removed and their inventories are reset after each interaction.

## Running with Scissors in the Matrix.

[![Substrate video](https://img.youtube.com/vi/oqYd4Ib5g70/0.jpg)](https://youtu.be/oqYd4Ib5g70)

Players can move around the map and collect resources of `K` discrete types. In
addition to movement, the agents have an action to fire an "interaction" beam.
All players carry an inventory with the count of resources picked up since last
respawn.

Players can observe their own inventory but not the inventories of their
coplayers. When another agent is zapped with the interaction beam, an
interaction occurs. The resolution of the interactions is determined by a
traditional matrix game, where there is a `K x K` payoff matrix describing the
reward produced by the pure strategies available to the two players. The
resources map one-to-one to the pure strategies of the matrix game. Unless
stated otherwise, for the purposes of resolving the interaction, the zapping
agent is considered the row player, and the zapped agent the column player. The
actual strategy played depends on the resources picked up before the
interaction. The more resources of a given type an agent picks up, the more
committed the agent becomes to the pure strategy corresponding to that resource.

In the case of running with scissors, `K = 3`, corresponding to rock, paper, and
scissors pure strategies respectively.

The payoff matrix is the traditional rock-paper-scissors game matrix.

Running with scissors was first described in Vezhnevets et al. (2020). Two
players gather rock, paper or scissor resources in the environment and can
challenge one another to a 'rock, paper scissor' game, the outcome of which
depends on the resources they collected. It is possible to observe the policy
that one's partner is starting to implement, either by watching them pick up
resources or by noting which resources are missing, and then take
countermeasures. This induces a wealth of possible feinting strategies.

Players can also zap resources with their interaction beam to destroy them. This
creates additional scope for feinting strategies.

Players have a `5 x 5` observation window.

Vezhnevets, A., Wu, Y., Eckstein, M., Leblond, R. and Leibo, J.Z., 2020. OPtions
as REsponses: Grounding behavioural hierarchies in multi-agent reinforcement
learning. In International Conference on Machine Learning (pp. 9733-9742). PMLR.

## Stag Hunt in the Matrix.

[![Substrate video](https://img.youtube.com/vi/7fVHUH4siOQ/0.jpg)](https://youtu.be/7fVHUH4siOQ)

See _Running with Scissors in the Matrix_ for a general description of the game
dynamics. Here the payoff matrix represents the Stag Hunt game. `K = 2`
resources represent "stag" and "hare" pure strategies.

The map configuration is different from other "_in the Matrix_" games. In this
case there are more _hare_ resources than _stag_ resources.

Players have the default `11 x 11` (off center) observation window.

## Territory: Open.

[![Substrate video](https://img.youtube.com/vi/3hB8lABa6nI/0.jpg)](https://youtu.be/3hB8lABa6nI)

Players can claim a resource in two ways: (1) by touching it, and (2) by using a
"claiming beam", different from the zapping beam, which they also have. Claimed
resources are colored in the unique color of the player that claimed them.
Unclaimed resources are gray. Players cannot walk through resources, they are
like walls.

Once a resource has been claimed a countdown begins. After 100 timesteps, the
claimed resource becomes active. This is visualized by a white and gray plus
sign appearing on top. Active resources provide reward stochastically to the
player that claimed them at a rate of 0.01 per timestep. Thus the more resources
a player claims and can hold until they become active, the more reward they
obtain.

The claiming beam is of length 2. It can color through a resource to
simultaneously color a second resource on the other side. If two players stand
on opposite sides of a wall of resources of width 2 and one player claims all
the way across to the other side (closer to the other player than themselves)
then the player on the other side might reasonably perceive that as a somewhat
aggressive action. Less aggressive of course than the other option both players
have: using their zapping beam. If any resource is zapped twice then it gets
permanently destroyed. It no longer functions as a wall or a resource, allowing
players to pass through.

Like resources, when players are hit by a zapping beam they also get removed
from the game and never regenerate. Once a player has been zapped out it is
gone. All resources it claimed are immediately returned to the unclaimed state.

## Territory: Rooms.

[![Substrate video](https://img.youtube.com/vi/u0YOiShqzA4/0.jpg)](https://youtu.be/u0YOiShqzA4)

See _Territory: Open_ for the general description of the mechanics at play in
this substrate.

In this substrate, _Territory: Rooms_, individuals start in segregated rooms
that strongly suggest a partition individuals could adhere to. They can break
down the walls of these regions and invade each other's "natural territory", but
the destroyed resources are lost forever. A peaceful partition is possible at
the start of the episode, and the policy to achieve it is easy to implement. But
if any agent gets too greedy and invades, it buys itself a chance of large
rewards, but also chances inflicting significant chaos and deadweight loss on
everyone if its actions spark wider conflict. The reason it can spiral out of
control is that once an agent's neighbor has left their natural territory then
it becomes rational to invade the space, leaving one's own territory undefended,
creating more opportunity for mischief by others.

## Scenario details

### Allelopathic Harvest

1.  _Focals are resident and a visitor prefers green._ This is a resident-mode
    scenario with a background population of A3C bots. The focal population must
    recognize that a single visitor bot has joined the game and is persisting in
    replanting berry patches in a color the resident population does not prefer
    (green instead of their preferred red). The focal agents should zap the
    visitor to prevent them from planting too much.

1.  _Visiting a green preferring population._ This is a visitor-mode scenario
    with a background population of A3C bots. Four focal agents join twelve from
    the background population. In this case the background population strongly
    prefers green colored berries. They plant assiduously so it would be very
    difficult to stop them from making the whole map green. The four focal
    agents prefer red berries but need to recognize that in this case they need
    to give up on planting berries according to their preferred convention and
    instead join the resident population in following their dominant convention
    of green. Otherwise, if the focal agents persist in planting their preferred
    color then they can succeed only in making everyone worse off.

1.  _Universalization_. Agents face themselves in this setting. As a
    consequence, all players have congruent incentives alleviating the conflict
    between groups that desire different berry colors. This test also exposes
    free rider agents that rely on others to do the work of replanting.

### Arena Running with Scissors in the Matrix

1.  _Versus gullible bots._ This is a half-and-half-mode scenario with a
    background population of A3C bots. Here the four focal agents interact in
    transient dyadic pairs with four bots from the background population that
    were trained to best respond to bots that implement pure rock, paper, or
    scissors strategies. They try to watch which resource their potential
    partners are collecting in order to pick the best response. However they are
    vulnerable to feinting strategies that trick them into picking the wrong
    counter.

1.  _Versus mixture of pure bots._ This is a half-and-half-mode scenario with a
    background population of A3C bots. Here the four focal agents interact in
    transient dyadic pairs with four bots from the background population sampled
    from the full set of pure strategy bots. So some will be pure rock players,
    some pure paper, and some pure scissors. The task for the focal agents is to
    watch the other players, see what strategy they are implementing and act
    accordingly.

1.  _Versus pure rock bots._ This is a half-and-half-mode scenario with a
    background population of A3C bots. Here the four focal agents interact in
    transient dyadic pairs with four bots from the background population that
    were trained with pseudorewards so they would implement pure rock
    strategies. The task of the focal agents is to collect paper resources,
    avoid one another, and target the rock players from the population to get
    high scores per interaction.

1.  _Versus pure paper bots._ This is a half-and-half-mode scenario with a
    background population of A3C bots. Here the four focal agents interact in
    transient dyadic pairs with four bots from the background population that
    were trained with pseudorewards so they would implement pure paper
    strategies. The task of the focal agents is to collect scissors resources,
    avoid one another, and target the paper players from the population to get
    high scores per interaction.

1.  _Versus pure scissors bots._ This is a half-and-half-mode scenario with a
    background population of A3C bots. Here the four focal agents interact in
    transient dyadic pairs with four bots from the background population that
    were trained with pseudorewards so they would implement pure scissors
    strategies. The task of the focal agents is to collect rock resources, avoid
    one another, and target the scissors players from the population to get high
    scores per interaction.

### Bach or Stravinsky in the Matrix

1.  _Visiting pure Bach fans._ This is a visitor-mode scenario with a background
    population of A3C bots. Here the focal agent must work out that it is best
    to play Bach regardless of whether playing in a given episode as a row
    player or a column player. All potential interaction partners play Bach, so
    the the focal agent should follow that convention and thereby coordinate
    with them.

1.  _Visiting pure Stravinsky fans._ This is a visitor-mode scenario with a
    background population of A3C bots. Here the focal agent must work out that
    it is best to play Stravinsky regardless of whether playing in a given
    episode as a row player or a column player. All potential interaction
    partners play Stravinsky, so the the focal agent should follow that
    convention and thereby coordinate with them.

1.  _Universalization_. This test may expose agents that have learned to be too
    stubborn.

### Capture the Flag

All bots in the background population trained with the following pseudoreward scheme: reward = 1 for zapping an avatar on the opposing team, reward = 2 for zapping the opposing team's flag carrier, reward = 3 for returning a flag dropped on the ground back to base by touching it, reward = 5 for picking up the opposing team's flag, and reward = 25 for capturing the opposing team's flag (or -25 when the opposing team does the same).

1.  _Focal team versus shaped A3C bot team._ This is a half-and-half-mode
    scenario with a background population of A3C bots. Here a team of four focal
    agents square off against a team of four bots sampled from the background
    population. This is a purely competitive game so the goal is to defeat the
    opposing team. It requires teamwork and coordination to do so. Since focal
    players are always sampled from the same training population, they are
    familiar in this case with their teammates but the opposing team is
    unfamiliar to them (never encountered during training).

1.  _Focal team versus shaped V-MPO bot team._ This is a half-and-half-mode
    scenario with a background population of V-MPO bots. Here a team of four
    focal agents square off against a team of four bots sampled from the
    background population. This is a purely competitive game so the goal is to
    defeat the opposing team. It requires teamwork and coordination to do so.
    Since focal players are always sampled from the same training population,
    they are familiar in this case with their teammates but the opposing team is
    unfamiliar to them (never encountered during training).

1.  _Ad hoc teamwork with shaped A3C bots._ This is a visitor-mode scenario with
    a background population of A3C bots. It demands skills of ad-hoc teamwork.
    In this case the lone focal agent must coordinate with unfamiliar teammates
    in order to defeat a similarly unfamiliar opposing team.

1.  _Ad hoc teamwork with shaped V-MPO bots._ This is a visitor-mode scenario
    with a background population of VMPO bots. It demands skills of ad-hoc
    teamwork. In this case the lone focal agent must coordinate with unfamiliar
    teammates in order to defeat a similarly unfamiliar opposing team.

### Chemistry: Branched Chain Reaction

The task features a reaction chain with two branch where the products from one can be fed back into the other to continue in a sustainable manner. While either branch can be run as long as there is material, running either more than the other runs out of some critical molecules. We train A3C both who are only rewarded for one branch. The evaluation agents can achieve their task if they manage to group with the right selection of other agents carrying suitable molecules, setting up a site where all the reaction keeps running sustainably.

1.  _Focals meet X preferring bots._ This is a half-and-half-mode scenario with
    a background population of A3C bots specialized in branch $$X$$. The
    evaluated agents will have to find a way to combine with this particular
    kind of bots to set up desired chain reactions.

1.  _Focals meet Y preferring bots._ This is a half-and-half-mode scenario with
    a background population of A3C bots specialized in branch $$Y$$.

1.  _Focals are resident._ This is a resident-mode scenario where the evaluated
    agents are paired with a single A3C bot that will be a specialist in on of
    the two branches.

1.  _Visiting another population._ This is a visitor-mode scenario with a
    background population of A3C bots where each is specialized on one of the
    cycles.

1.  _Universalization_. If policies have become so specialized that they are no
    longer able to function as generalists when necessary then they won't
    perform well in this test. Partner choice is especially important for this
    substrate which seems to make universalization scenarios somewhat easier.

### Chemistry: Metabolic Cycles

The task has some sub-cycles of reaction and we train bots who are are only rewarded for the food produced in one of the two, besides also getting rewarded for generating new energy. This training results in bots specialized in either cycle $$A$$ or cycle $$B$$ and that will ignore the other. In the first two tests it will be tested if the evaluated agents can adapts to being paired with bots specialized in one of the cycles and adapt to run the other.

1.  _Focals meet A preferring bots._ This is a half-and-half-mode scenario with
    a background population of A3C bots specialized in cycle $$A$$. The
    evaluated agents will have to run the cycle $$B$$ side to be successful.

1.  _Focals meet B preferring bots._ This is a half-and-half-mode scenario with
    a background population of A3C bots specialized in cycle $$B$$. The
    evaluated agents will have to run the cycle $$A$$ side to be successful.

1.  _Focals are resident._ This is a resident-mode scenario where the evaluated
    agents are paired with a single A3C bot that will be a specialist in on of
    the two cycles. For optimal returns the residents needs to make sure a good
    balance across tasks is achieved.

1.  _Visiting another population._ This is a visitor-mode scenario with a
    background population of A3C bots where each is specialized on one of the
    cycles. For the most success the visitor needs to take on the most needed
    task that depends on the sampled bots.

1.  _Universalization_. This test exposes populations that learned
    specialization and division of labor strategies. If policies have become so
    specialized that they are no longer able to function as generalists when
    necessary then they won't perform well in this test.

### Chicken in the Matrix

1.  _Meeting a mixture of pure bots._ This is a half-and-half-mode scenario with
    a background population of A3C bots. In this case the background population
    samples four bots uniformly over a set where each was pretrained to mostly
    play either hawk or dove. The focal agents must watch their interaction
    partners carefully to see if they have collected hawk and if they did, then
    either avoid them or play dove. If they want to play hawk then they should
    try to interact with other players who they saw collecting dove.

1.  _Visiting a pure dove population._ This is a visitor-mode scenario where a
    single focal agent joins seven from the background population. All members
    of the background population were pretrained to specialize in playing dove.
    Thus the correct policy for the focal visitor is to play hawk.

1.  _Focals are resident and visitors are hawks._ This is resident-mode scenario
    where three members of the background population join a focal resident
    population of five agents. In this case the background population bots
    specialize in playing hawk. The task for the resident agents is to seek
    always to interact in (hawk, dove) or (dove, dove) pairings, importantly
    avoiding (hawk, hawk).

1.  _Visiting a gullible population._ This is a visitor-mode scenario where a
    single focal agent joins seven from the background population. In this case
    the background population bots trained alongside (mostly) pure hawk and dove
    agents, but were not themselves given any non-standard pseudoreward scheme.
    They learned to look for interaction partners who are collecting dove
    resources so they can defect on them by playing hawk. Thus they can be
    defeated by feinting to make them think you are playing dove when you
    actually will play hawk.

1.  _Visiting grim reciprocators._ This is a visitor-mode scenario where two
    focal agents join a group of six from the background population. The
    background population bots are conditional cooperators. They play dove
    unless and until they are defected on by a partner (i.e.~ their partner
    chooses hawk). After that they will attempt to play hawk in all future
    encounters until the end of the episode. Such a strategy is often called
    _Grim Trigger_ because it never forgives a defection. Note that they do not
    attempt to punish the specific player that defected on them in the first
    place, instead defecting indiscriminately on all future interaction
    partners, sparking a wave of recrimination that ends up causing all the
    background bots to defect after a short while. Since always last 1000 steps,
    the best strategy here for the focal agents would probably be to play dove
    up until near the end of the episode and then play hawk since no further
    retaliation will be possible at that point.

1.  _Universalization_. Dove-biased policies will perform relatively well but
    hawk-based policies will perform exceptionally badly here.

### Clean Up

1.  _Visiting an altruistic population._ This is a visitor-mode scenario where
    three focal agents join four from the background population. The background
    bots are all interested primarily in cleaning the river. They rarely consume
    apples themselves. The right choice for agents of the focal population is to
    consume apples, letting the background population do the work of maintaining
    the public good.

1.  _Focals are resident and visitors free ride._ This is a resident-mode
    scenario where four focal agents are joined by three from the background
    population. The background bots will only consume apples and never clean the
    river. It tests if the focal population is robust to having a substantial
    minority of the players (three out of seven) be _defectors_ who never
    contribute to the public good. Focal populations that learned too brittle of
    a division of labor strategy, depending on the presence of specific focal
    players who do all the cleaning, are unlikely to do well in this scenario.

1.  _Visiting a turn-taking population that cleans first._ This is a
    visitor-mode scenario where three focal agents join four agents from the
    background population. The background bots implement a strategy that
    alternates cleaning with eating every 250 steps. These agents clean in the
    first 250 steps. Focal agents that condition an unchangeable choice of
    whether to clean or eat based on what their coplayers do in the beginning of
    the episode (a reasonable strategy that self-play may learn) will perceive
    an opening to eat in this case. A better strategy would be to take turns
    with the background bots: clean when they eat and eat when they clean.

1.  _Visiting a turn-taking population that eats first._ This is a visitor-mode
    scenario where three focal agents join four agents from the background
    population. As in scenario `3`, the background bots implement a strategy
    that alternates cleaning with eating every 250 steps. However, in this case
    they start out by eating in the first 250 steps. If the focal population
    learned a policy resembling a _grim trigger_ (cooperate until defected upon,
    then defect forever after), then such agents will immediately think they
    have been defected upon and retaliate. A much better strategy, as in SC2, is
    to alternate cleaning and eating out of phase with the background
    population.

1.  _Focals are visited by one reciprocator._ This is a resident-mode scenario
    where six focal agents are joined by one from the background population. In
    this case the background bot is a conditional cooperator. That is, it will
    clean the river as long as at least one other agent also cleans. Otherwise
    it will (try to) eat. Focal populations with at least one agent that
    reliably cleans at all times will be successful. It need not be the same
    cleaner all the time, and indeed the solution is more equal if all take
    turns.

1.  _Focals are visited by two suspicious reciprocators._ This is resident-mode
    scenario where five focal agents are joined by two conditional cooperator
    agents from the background population. Unlike the conditional cooperator in
    scenario `5`, these conditional cooperators have a more stringent condition:
    they will only clean if at least two other agents are cleaning. Otherwise
    they will eat. Thus, two cleaning agents from the focal population are
    needed at the outset in order to get the background bots to start cleaning.
    Once they have started then one of the two focal agents can stop cleaning.
    This is because the two background bots will continue to clean as long as,
    from each agent's perspective, there are at least two other agents cleaning.
    If any turn taking occurs among the focal population then they must be
    careful not to leave a temporal space when no focal agents are cleaning lest
    that cause the background bots to stop cooperating, after which they could
    only be induced to clean again by sending two focal agents to the river to
    clean

1.  _Focals are visited by one suspicious reciprocator._ This is a resident-mode
    scenario where six focal agents are joined by one conditional cooperator
    agent from the background population. As in scenario `6`, this conditional
    cooperator requires at least two other agents to be cleaning the river
    before it will join in and clean itself. The result is that the focal
    population must spare two agents to clean at all times, otherwise the
    background bot will not help. In that situation, the background bot joins as
    a third cleaner. But the dynamics of the substrate are such that it is
    usually more efficient if only two agents clean at once. Focal populations
    where agents have learned to monitor the number of other agents cleaning at
    any given time and return to the apple patch if more than two are present
    will have trouble in this scenario because once one of those agents leaves
    the river then so too will the background bot, dropping the total number
    cleaning from three down to one, which is even more suboptimal since a
    single agent cannot clean the river effectively enough to keep up a high
    rate of apple growth. The solution is that the focal population must notice
    the behavior of the conditional cooperating background bot and accept that
    there is no way to achieve the optimal number of cleaners (two), and instead
    opt for the least bad possibility where three agents (including the
    background conditional cooperator) all clean together.

1.  _Universalization_. This test exposes both free riders and overly altruistic
    policies. Both will get low scores here.

### Collaborative Cooking: Impassable

1.  _Visiting a V-MPO population._ This is a visitor-mode scenario where one
    focal agent must join a group of three from the background population. The
    focal agent must observe the patterns of its coplayers' coordination and
    help out where ever it can. Above all, it must avoid getting in the way.

1.  _Focals are resident._ This is a resident-mode scenario where three focal
    agents are joined by one from the background population. It tests if the
    coordination strategy learned by the focal population is robust to replacing
    one familiar agent with an unfamiliar agent from the background population,
    who might not be used to their conventions.

1.  _Universalization_. This test penalizes overspecialization.

### Collaborative Cooking: Passable

1.  _Visiting uncoordinated generalists._ This is a visitor-mode scenario where
    one focal agent must join a group of three from the background population.
    The background population agents are all generalists who try to do all tasks
    themselves, and do not really coordinate much. The focal agent must
    recognize that the others are not being helpful and elect a similarly
    independent strategy.

1.  _Focals are resident and visited by an uncoordinated generalist._ This is a
    resident-mode scenario where three focal agents are joined by one from the
    background population. The background bot is a generalist that attempts to
    do all tasks itself, and does not try to coordinate. The test is for the
    agents sampled from the focal population to find ways to cooperate despite
    the presence of this agent, who might frequently get in the way.

1.  _Universalization_. This test penalizes overspecialization.

### Commons Harvest: Open

1.  _Focals are resident and visited by two zappers._ This is a resident-mode
    scenario where fourteen focal agents are joined by two from the background
    population. The two background bots are both especially interested in
    zapping, though not to the detriment of the rest of their policy. They do
    not act sustainably. The solution is to act sustainably while preventing the
    two background bots from depleting too many patches by harvesting them
    unsustainably. This can be done by zapping them whenever they approach a
    patch too closely.

1.  _Focals are resident and visited by six zappers._ This is also a
    resident-mode scenario, but this time 10 focal agents are joined by six from
    the background community. The task is the same as scenario `1` but this time
    it is much harder because there are six unsustainable and zapping prone
    background bots instead of just two.

1.  _Universalization_. It is sometimes not problematic for a single individual
    to act greedily but if everyone acted that same way then it would be a
    catastrophe. This test exposes agents that learned to take advantage of such
    threshold effects.

### Commons Harvest: Closed

1.  _Focals are resident and visited by two zappers._ This is a resident-mode
    scenario where fourteen focal agents are joined by two from the background
    population. Both background bots will act sustainably if given the chance by
    controlling the entrance to a room, though they are not perfectly consistent
    in this behavior. The solution is for as many individuals as possible to try
    to capture individual rooms by zapping any agent as they emerge from the
    corridor. This has the effect of making their harvesting behavior less
    interdependent. An agent that successfully controls the entrance to _its
    territory_ is thus incentivized not to overharvest it.

1.  _Focals are resident and visited by six zappers._ Same as scenario `1` but
    harder since this time there are six background bots instead of just two.

1.  _Visiting a population of zappers._ This is a visitor-mode scenario where
    four focal agents join twelve background bots. It is similar to scenario `1`
    and scenario `2`. Despite the change from resident to visitor mode, the task
    for each individual focal agent is still more or less the same. The reason
    this is the case is that the goal behavior here is for agents to take
    actions to secure their independence. Once secured, an agent can control its
    own territory indefinitely, without any need to depend om others.

1.  _Universalization_. This test is similar to its analog in Commons Harvest
    Open.

### Commons Harvest: Partnership

This substrate raised the question: how can we make scenarios to test for abilities that state-of-the-art agents don't yet have?

Before starting this project we already knew that selfish RL agents acting in a
group would learn to act unsustainably and cause a tragedy of the commons
outcome in open field _Commons Harvest_ substrates like our _Commons Harvest:
Open_. We also already knew that agents that manage to get inside walled off
regions could learn to zap invaders as they run down the corridor, and that
after learning effective defense they would be reincentivized to act sustainably
toward their resources since they could be sure to monopolize their future yield
until the end of the episode, as in our _Commons Harvest: Closed_ substrate. We
wanted to create a test scenario where the walled regions each had two entrances
so that pairs of agents would have to work together to defend _their_ territory.
Solving this test scenario should require agents to trust one another to (a) act
sustainably with regard to their shared resources, and (b) competently guard one
of the two entrances.

We knew from preliminary work that it is quite difficult to get current state-of-the-art agents to learn this kind of trust behavior. How then could we create bots to populate test scenarios that test for a behavior we don't know how to create in the first place? The solution we found was to create bots that would function as good partners for agents that truly got the concept of trust---even though they would not themselves truly _understand_ it in a general way. We gave bots negative rewards during training for crossing invisible tiles located along the vertical midline of each room. Once agents came to regard those tiles as akin to walls, the game then resembled again the situation of _Commons Harvest: Closed_ where they reliably learn sustainable policies. A focal agent playing alongside one of these _good partner_ bots would experience a coplayer that dutifully guards one or the other entrance and only ever collects apples from one side of the room. A focal agent capable of the limited form of trust demanded by this scenario should be able to cooperate with such a partner.

1.  _Meeting good partners._ This is a half-and-half-mode scenario where eight
    focal agents join a background population consisting of eight good partner
    agents. The objective, for the agents lucky enough to reach the closed off
    regions first, is to collaborate with their partner to (a) defend the
    corridor an (b) act sustainably in harvesting their shared apple patches.

1.  _Focals are resident and visitors are good partners._ This is a
    resident-mode scenario where twelve focal agents join four background good
    partner agents. It is similar to scenario `1`. However, because there are
    more familiar individuals around (it is resident mode), the need for ad hoc
    cooperation with unfamiliar individuals is decreased.

1.  _Visiting a population of good partners._ This is a visitor-mode scenario
    where four focal agents join twelve background good partner agents. Like
    scenario `2`, this scenario differs from scenario `1` in the extent to which
    success depends on ad hoc cooperation with unfamiliar individuals. In this
    case _more_ cooperation with unfamiliar individuals is required than in
    scenario `1`.

1.  _Focals are resident and visited by two zappers._ This is a resident-mode
    scenario where fourteen focal agents are joined by two background bots. Here
    the background bots are not good partners. They will frequently zap other
    players and act unsustainably whenever they get the chance to harvest. The
    solution is to partner with familiar agents, act sustainably, and cooperate
    to defend territories against invasion from the background bots.

1.  _Focals are resident and visited by six zappers._ This is a resident-mode
    scenario where ten focal agents are joined by six background bots who are
    not good partners, act unsustainably, and zap frequently. It is a harder
    version of scenario `4`. Since attempted invasions will occur more
    frequently, agents must be more skillful to prevent their territories from
    being overrun.

1.  _Visiting a population of zappers._ This is a visitor-mode scenario there
    four focal agents are joined by twelve background bots who are not good
    partners, act unsustainably, and zap frequently. In this case it will
    usually be impossible for the small minority of focal agents to find one
    another and form an effective partnership. The best option from a menu with
    no good options is just to harvest apples as quickly as possible and act
    more-or-less unsustainably once invaded.

1.  _Universalization_. This test is similar to its analog in Commons Harvest
    Open.

### King of the Hill

_King of the Hill_ is an eight player game pitting two teams of four players each against one another.

The true (default) reward scheme is as follows. All members of a team get a
reward of 1 on every frame that their team controls the hill region in the
center of the map. If no team controls the map on a given frame (because no team
controls more than 80% of the hill) then no team gets a reward on that frame. No
other events provide any reward. This reward scheme is always in operation at
test time.

The alternative _zap while in control_ reward scheme is as follows. Agents get a reward of 1 whenever both of the following conditions are satisfied simultaneously: (1) their team is in control of the hill, and (2) they just zapped a player of the opposing team, bringing their health to 0, and removing them from the game. This reward scheme was only used to train (some) background populations. Training with this scheme produces qualitatively different policies that still function effectively under the true reward scheme.

1.  _Focal team versus default V-MPO bot team._ In this scenario a team composed
    entirely of focal agents must defeat a team composed entirely from a
    background population. This tests learned teamwork since a familiar team
    plays against an unfamiliar opposing team. In this case the background
    population was trained using the V-MPO algorithm, and used the substrate's
    true (default) reward scheme (see above). This background population tends
    to spend most of its time near the hill.

1.  _Focal team versus shaped A3C bot team._ In this scenario a team composed
    entirely of focal agents must defeat a team composed entirely from a
    background population. This tests learned teamwork since a familiar team
    plays against an unfamiliar opposing team. In this case the background
    population was trained using the A3C algorithm, and used the _zap while in
    control_ reward scheme (see above).

1.  _Focal team versus shaped V-MPO bot team._ In this scenario a team composed
    entirely of focal agents must defeat a team composed entirely from a
    background population. This tests learned teamwork since a familiar team
    plays against an unfamiliar opposing team. In this case the background
    population was trained using the V-MPO algorithm, and used the _zap while in
    control_ reward scheme. This causes the agents in the background population
    to implement a _spawn camping_ policy. The counter-strategy is to evade
    opponents at the spawn location by running immediately toward the hill area
    to capture it, forcing the opposing team to abandon their position and
    return to defend the hill in a more chaotic and vulnerable fashion.

1.  _Ad hoc teamwork with default V-MPO bots._ This scenario tests ad-hoc
    teamwork. One agent sampled from the focal population joins a team with
    three other agents from a background population to compete against another
    team of four agents sampled the same background population. If the focal
    agent works well with its unfamiliar teammates then it can tip the balance.
    In this case the background population was trained using the V-MPO algorithm
    and used the substrate's true (default) reward scheme. This background
    population tends to spend most of its time near the hill so in order to
    coordinate with them the focal agent should also spend time there. In
    particular, it should guard whichever entrance to the room containing the
    hill is least well guarded by its allies.

1.  _Ad hoc teamwork with shaped A3C bots._ This scenario tests ad-hoc teamwork.
    One agent sampled from the focal population joins a team with three other
    agents from a background population to compete against another team of four
    agents sampled the same background population. If the focal agent works well
    with its unfamiliar teammates then it can tip the balance. In this case the
    background population was trained using the A3C algorithm and used the _zap
    while in control_ reward scheme.

1.  _Ad hoc teamwork with shaped V-MPO bots._ This scenario tests ad-hoc
    teamwork. One agent sampled from the focal population joins a team with
    three other agents from a background population to compete against another
    team of four agents sampled the same background population. If the focal
    agent works well with its unfamiliar teammates then it can tip the balance.
    In this case the background population was trained using the V-MPO algorithm
    and used the _zap while in control_ reward scheme. To work well with these
    background bots who implement a _spawn camping_ policy, the focal agent
    should follow them to the opposing team's base and help guard the escape
    routes by which spawning agents could get past them and take the hill. They
    must pick the side of the map to guard where less of their allies are
    stationed.

### Prisoner's Dilemma in the Matrix

Players are identifiable by color (randomized between episodes but maintained for each episode's duration).

1.  _Visiting unconditional cooperators._ This is a visitor-mode scenario where
    one agent sampled from the focal population joins seven agents from a
    background population. In this case all the background bots will play
    cooperative strategies (mostly collecting _cooperate_ resources and rarely
    collecting _defect_). The objective of a rational focal visitor is to
    exploit them by collecting _defect_ resources.

1.  _Focals are resident and visitors are unconditional cooperators._ This is a
    resident-mode scenario where six focal agents are joined by two agents from
    the background population. The background bots play cooperative strategies,
    as in scenario `1`. A focal population will do well by defecting against
    these unfamiliar cooperators, but it should take care to identify them
    specifically. If they defect against all players regardless of identity then
    they will end up defecting on the focal population as well, lowering the
    overall score achieved.

1.  _Focals are resident and visitors defect._ This is a resident-mode scenario
    where six focal agents are joined by two agents from the background
    population. The background bots play defecting strategies (mostly collecting
    _defect_ resources). A focal population will do well by avoiding interaction
    with them and instead preferentially interacting with one another, and
    cooperating.

1.  _Meeting gullible bots._ This is a half-and-half-mode scenario where four
    focal agents join four background bots. In this case the background
    population bots trained alongside (mostly) pure cooperator and defector
    bots, but were not themselves given any non-standard pseudoreward scheme.
    Thus they became bots that try to preferentially interact with partners who
    they have seen collecting cooperate resources. The best strategy against
    them is probably for the focal players to trick them into thinking they are
    defecting so they won't try to force interactions. This creates the
    necessary space to allow the focal population to preferentially interact
    (and cooperate) with one another.

1.  _Visiting a population of grim reciprocators._ This is a visitor mode
    scenario where one focal agent joins seven from the background population.
    The background population bots are conditional cooperators. They will
    collect cooperate resources and attempt to play cooperate in all
    interactions until they have twice been defected upon by their partners.
    After that, they try to retaliate by collecting defect resources and aiming
    to defect in all remaining encounters till the end of the episode. Once they
    start defecting they do so indiscriminately. They do not try to specifically
    punish the players who defected on them. This sparks a wave of recrimination
    that ends up causing all the background bots to defect after a short while.
    Since episodes always last 1000 steps, the best strategy for the focal agent
    is probably to cooperate up until near the end of the episode and then
    defect once there is no remaining time for retaliation.

1.  _Visiting a population of hair-trigger grim reciprocators._ Just like
    scenario `5` but with background bot conditional cooperators operating on
    more of a hair trigger. That is, unlike scenario `5` where the bots will
    forgive one defection, the bots here will not forgive any defection at all.
    They begin defecting indiscriminately after the first time any agent defects
    on them.

1.  _Universalization_. Cooperation biased agents will do well while defection
    biased agents will do poorly here. If an agent learned a tit-for-tat style
    policy then it would also do well here.

### Pure Coordination in the Matrix

Unlike most other substrates, players here are not identifiable by color. All players look exactly the same as one another. It would be very difficult to track an individual for any significant length of time.

1.  _Focals are resident and visitor is mixed._ This is a resident-mode scenario
    where seven focal agents are joined by one agent sampled from a background
    population. In this case the background population contains different kinds
    of specialist agents, each targeting one particular resource out of three.
    Focal agents need to watch other players to see what resource they are
    collecting and try to pick the same one. Most of the time they will be
    interacting with familiar others so whatever strategy they learned at test
    time will suffice. This scenario tests that their coordination is not
    disrupted by the presence of an unfamiliar other player who specializes in
    one particular resource.

1.  _Visiting resource A fans._ This is a visitor-mode scenario where one agent
    sampled from the focal population joins seven sampled from a background
    population. In this case the background population consists of specialists
    in resource A.

1.  _Visiting resource B fans._ Just like scenario `2` but with a background
    population consisting of specialists in resource B.

1.  _Visiting resource C fans._ Just like scenario `2` but with a background
    population consisting of specialists in resource C.

1.  _Meeting uncoordinated strangers._ This is a half-and-half-mode scenario
    where four focal agents join four background bots. As in scenario `1`, the
    background population contains players with all three different
    specializations. Thus they will normally be very uncoordinated. The
    objective for the focal population is to carefully watch which resources its
    potential partners will choose and select interaction partners on that
    basis, thus mainly interacting with familiar individuals who should all be
    playing the same strategy.

1.  _Universalization_. The main possible failure mode here is if too many
    agents try to collect too many of the same resources at one so they don't
    get distributed well and some agents miss the chance to collect one.

### Rationalizable Coordination in the Matrix

Unlike most other substrates, players here are not identifiable by color. All players look exactly the same as one another. It would be very difficult to track an individual for any significant length of time.

1.  _Focals are resident and visitor is mixed._ This is a resident-mode scenario
    where seven focal agents are joined by one agent sampled from a background
    population. In this case the background population contains different kinds
    of specialist agents, each targeting one particular resource out of three.
    This scenario is similar to Pure Coordination in the Matrix in that it tests
    that the focal population's coordination is not disrupted by the presence of
    an unfamiliar other player who specializes in one particular resource. The
    problem is more difficult here than in the pure coordination substrate
    though because all but one coordination choice is irrational. That is, while
    all choices are better than miscoordination, it is still clearly better to
    coordinate on some choices over others.

1.  _Visiting resource A fans._ This is a visitor-mode scenario where one agent
    sampled from the focal population joins seven sampled from a background
    population. In this case the background population consists of specialists
    in resource A. It would be irrational for the group to coordinate on
    resource A since both resource B and C are better for everyone.

1.  _Visiting resource B fans._ Just like scenario `2` but with a background
    population consisting of specialists in resource B. Even though it is better
    to coordinate on resource B than on resource A, it is still irrational for
    the group to coordinate on it since resource C is strictly better.

1.  _Visiting resource C fans._ Just like scenario `2` but with a background
    population consisting of specialists in resource C. This is the resource
    that it is rational to coordinate on.

1.  _Meeting uncoordinated strangers._ This is a half-and-half-mode scenario
    where four focal agents join four background bots. As in scenario `1`, the
    background population contains players with all three different
    specializations. Thus they will normally be very uncoordinated. The
    objective for the focal population is to carefully watch which resources its
    potential partners will choose and select interaction partners on that
    basis, thus mainly interacting with familiar individuals who should all be
    collecting resource C since it is rational to do so.

1.  _Universalization_. This test is similar to its analog for Pure Coordination
    in the Matrix.

### Running with Scissors in the Matrix

This is a two-player zero-sum game. It was first introduced in.

1.  _Versus gullible opponent._ Here the focal agent must defeat an opposing
    agent that was trained to best respond to agents playing pure strategies.
    The opponent should attempt to scout out what strategy the focal agent is
    playing so it can pick the appropriate counter. To defeat it, the focal
    agent should feint toward one resource and then collect the counter to its
    counter. So for example, if the focal agent successfully feinted that it
    would pick rock, inducing its opponent to pick paper, the focal agent should
    then collect and play scissors.

1.  _Versus mixed strategy opponent._ Here the focal agent must defeat an
    opponent that was trained to play a relatively pure strategy: either rock,
    paper, or scissors. However, the specific opponent is sampled at test time
    so it could be any of those. To defeat it, the focal agent should scout out
    which pure strategy its opponent is playing and then collect the resources
    to implement its counter strategy.

1.  _Versus pure rock opponent._ Same as scenario `2`, but the opponent will
    always play rock.

1.  _Versus pure paper opponent._ Same as scenario `2`, but the opponent will
    always play paper.

1.  _Versus pure scissors opponent._ Same as scenario `2`, but the opponent will
    always play scissors.

### Stag Hunt in the Matrix

1.  _Visiting a population of stags._ This is a visitor-mode scenario where one
    agent sampled from the focal population joins seven sampled from a
    background population. All background bots have been trained to primarily
    play stag. The right solution is for the focal visitor agent to also play
    stag.

1.  _Visiting a population of hares._ This is a visitor-mode scenario where one
    agent sampled from the focal population joins seven sampled from a
    background population. This test is just like scenario `1` except the
    background population was trained primarily to play hare. The right solution
    is for the focal visitor agent to also play hare.

1.  _Visiting a population of grim reciprocators._ This is a visitor-mode
    scenario where two focal agents join six bots from the background
    population. The background population bots are reciprocators. They start out
    by playing stag but if any of their interaction partners plays hare then
    they become triggered to play hare themselves in all their future
    interactions till the end of the episode.

1.  _Universalization_. Stag biased agents will do better than hare biased
    agents as long as they can avoid running out of stag resources due to
    individuals picking up more than they need to commit to the strategy.

### Territory: Open

1.  _Focals are resident and visited by a shaped bot._ This is a resident-mode
    scenario where eight agents from the focal population are joined by one
    agent sampled from a background population. The background bot is typically
    quite active, it runs around the map and claims as much territory as it can.
    It doesn't use its zapping beam very often. The right response to it is
    often to zap it early in the episode so it can't get in the way of the rest
    of the agents fairly dividing up the resources according to whatever
    convention they agree on (i.e. whatever convention emerged during training).

1.  _Visiting a population of shaped bots._ This is a visitor-mode scenario
    where one focal agent joins eight from a background population. The
    background bots all behave as described in scenario `1`. They color a lot of
    territory but frequently color over one another's territory. The focal
    visitor could follow their conventional behavior and run around the map with
    them claiming and reclaiming as they go. Or alternatively, it may be
    possible to defend an area of the map and zap them if they approach too
    close. The latter strategy would likely achieve higher rewards if
    successful.

1.  _Universalization_. This test exposes overly aggressive policies. They will
    perform very poorly.

### Territory: Rooms

1.  _Focals are resident and visited by an aggressor._ This is a resident-mode
    scenario where eight agents from the focal population are joined by one
    agent sampled from a background population. The background population
    consists of agents that, although they did not themselves experience any
    special reward shaping, they trained alongside other agents that were
    incentivized to be extremely aggressive and destructive in their zapping
    behavior. As a result, they are fairly aggressive themselves while still
    being interested primarily in claiming territory. A focal population where
    agents have not forgotten how to defend themselves from attack will do much
    better in this scenario than a focal population that implements the same
    fair division of the resources but forgot self-defense.

1.  _Visiting a population of aggressors._ This is a visitor-mode scenario where
    one focal agent joins eight from a background population. The background
    population is the same as in scenario `1`. The focal visitor agent must be
    forever on its guard since neighboring agents may attack their territory at
    any time, permanently destroying resources on the way in. Since the larger
    set of background bots are so likely to fight among themselves, the optimal
    strategy for the focal visitor agent is often to invade after a battle and
    claim extra territory in areas where other agents have been zapped and
    removed.

1.  _Universalization_. This test exposes overly aggressive policies. They will
    perform very poorly.
