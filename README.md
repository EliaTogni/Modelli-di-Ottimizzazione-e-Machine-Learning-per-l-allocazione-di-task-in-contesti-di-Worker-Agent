Introduction {#sec:intro}
============

Crowdworking is an emerging paradigm, which is challenging the labor
market [@crowd]. In its essence, employers ask for the completion of
specific *tasks* by publishing their description on an open platform.
Freelance experts can then apply, proposing themselves for the
completion of these tasks at a price. Employers can then assign each
task to one expert (or even more than one to minimize risks), possibly
through a negotiation phase about job effort and price.

Online services have therefore flourished, to meet demand and offer on a
massive global scale. To reduce the complexity of task assignment in
such a scenario, these online services implement automatic matching.
Pre-defined policies are normally used, like "first offer" (immediately
assign the task once an offer is made), "cheapest offer" (assign each
task to the expert asking lowest price), "best quality" (assign each
task to the expert having highest rank among applicants), or suitable
combinations of them. These policies, which work one task at a time, do
not provide specific guarantees, neither in terms of global cost
minimization, nor quality maximization, nor effort balancing over
experts.

Unfortunately, when more than a single task is considered at once, a
combinatorial problem arises which is further complicated by its
nondeterministic nature: the performance of each expert on each specific
task is a-priori unknown.

In fact, crowdworking *games* are used even in procedures like personnel
selection to assess applicats skills in problem solving. It is the case
of Agile Manager [@agile], which is an online platform proposing the
following game. A set of tasks is given, each having a value, a
difficulty level, a required effort and a boolean flag 'deadline'; these
data are made available to the player. A set of worker agents is given,
each having a certaing high quality output probability, a maximum
productivity limit, a stamina level and a backlog of tasks. These data
(or part of them) might however be *unknown to the player.* Only an
aggregated performance indicator is always given, whose intuition is to
encode a *reputation* of the worker agent. The *logic* used by the
worker agents to schedule their assigned tasks is also unknown to the
player. The game proceeds in rounds: the player is given the set of
tasks with all their data, and part of the data of each worker agent
including reputation, and is asked to assign tasks to the worker agents.
Then a simulation is carried out: each worker agent might complete all
or a subset of its assigned tasks, including those in backlog, with
either high or low quality. In fact, the worker agent maximum
productivity or stamina might not be enough to complete all tasks within
the timelimit of one round. Uncompleted tasks flagged as 'deadline' are
lost, the others are added to the agent backlog. The simulation logic is
kept unknown to the player. After each round the player gets a *score*,
depending on which tasks have been completed and which have been
completed with high quality output. The player also gets new reputation
values for each worker agent, depending on how the agent has performed
during the round. The final player score is the sum of scores in all the
rounds of the match. We report that the online platform outputs also a
set of indices concerning user experience, which are however not
relevant for our study.

In this paper we carry out the following computational evaluation.
First, we analyze how regression models perform in forecasting the
outcome of a task subset allocation to a specific agent. Second, we
encode these models as terms of a mathematical programming model, which
aims at maximizing the player score in each round. We analyze the
behaviour of our models exploiting a dataset of logs of real matches,
made available in [@dataset]. Our interest into the problem is twofold:
first, it is a benchmark for the integration of data-driven models
(regression ones) in a mathematical programming framework, which is an
emerging trend in optimization [@ML4CO], and a line of research we are
pursuing [@DDfeas]; second, it gives an insight into the possible
application of optimization models for automatic matching in massive
online crowdworking services.

In Section [2](#sec:models){reference-type="ref" reference="sec:models"}
we formalize the problem and we introduce our models, in Section
[3](#sec:experiments){reference-type="ref" reference="sec:experiments"}
we report and discuss our experiments, and in Section
[4](#sec:conclusions){reference-type="ref" reference="sec:conclusions"}
we collect some conclusions.

Models {#sec:models}
======

Let $I$ be a set of tasks. Each task $i \in I$ is described by a value
$v_i$, a difficulty $d_i$, an effort $e_i$ and a deadline flag $u_i$.
Let $J$ be a set of Worker Agents (WA). Each WA $j \in J$ is described
by a maximum productivity level $m_j$, a high quality output probability
$p_j$ and a reputation $r_j$. An allocation of a subset
$\bar I \subseteq I$ of tasks to a WA $j$ produces a player score
$\sigma(\bar I, j)$ and an agent reputation update $\rho(\bar I, j)$.

The player objective is to find a partitioning of the tasks in one
subset $\bar I_j$ for WA $j$, in such a way that the total score
$\sum_{j \in J} \sigma(\bar I_j, j)$ is maximized.

We have considered different modeling options, trying to balance the
following four aspects (a) the complexity of a model forecasting the
player score (b) the set of features it requires (c) the complexity of
its representation in terms of mathematical programming (d) the
complexity of the overall optimization model. In the following we
describe only the modeling line which showed to be the most promising in
our computational evaluation.

Score forecasting models
------------------------

As discussed in the introduction, the rules for producing a score and a
reputation update given the tasks and WA details are unknown to the
player. Additionally, neither the disaggregated score is given to the
player after the round, but only the reputation update. Finally, also
the backlog of tasks of each WA remains unknown. Nevertheless, we assume
that (a) the score is linearly correlated with reputation update,
allowing to use the latter as a proxy for the former (b) the reputation
update is a function of tasks and worker agent data. In details we
assume
$$\sigma(\bar I, j) \simeq \ \alpha \cdot \rho(\bar I, j) \simeq f(\{(v_i, d_i, e_i) : i \in \bar I\}, m_j, p_j, r_j).$$

In such a way, we are able to estimate $f()$ with a supervised learning
approach. That is, we assume to be given a set of player matches $T$.
For each $t \in T$ the subset of allocated tasks $I^t \subseteq I$ and
the corresponding WA $j^t$ are given, together with the reputation
update $\rho^t$ which was produced. We estimate $f()$ by a multivariate
regression approach, as a function mapping each predictor tuple
$(v^t_1, \ldots, v^t_{|I|}, d^t_1, \ldots , d^t_{|I|}, e^t_1, \ldots, e_{|I|}^t, m_{j^t}, p_{j^t}, r_{j^t})$
(where $v^t_i = v_i$ if $i \in I^t$, $0$ otherwise, $d_i^t$ and $e_i^t$
being defined similarly) to the response variable $\rho^t$. Note that
deadline flags are not chosen to be part of the predictor variables.

In Section [3](#sec:experiments){reference-type="ref"
reference="sec:experiments"} we compare a few regression modeling
techniques. One of them is linear regression, with a different
regression model $f^j()$ created for each WA. That is, we assume
$$\rho^t \simeq \sum_{i \in I} (\alpha_{ij} v_i + \beta_{ij} d_i + \gamma_{ij} e_i) + \mu_j m_j + \pi_j p_j + \nu_j r_j + \epsilon_j$$
and we find parameters $\alpha_{ij}$, $\beta_{ij}$, $\gamma_{ij}$,
$\mu_j$, $\pi_j$, $\nu_j$ and $\epsilon_j$ by supervised learning
methods, that is training $|J|$ regression models, each on the set of
attempts which refer to WA $j$. That is, intitively, a model with a
simple structure, but a large set of features.

Optimization models
-------------------

Once each mapping $f^j()$ from tasks and WA data to the score is
determined, as described in the previous subsection, an optimization
model can be formulated as follows: $$\begin{aligned}
\text{max.} & \sum_{j \in J} \sum_{i \in I} \left(\alpha_{ij} v_i + \beta_{ij} d_i + \gamma_{ij} e_i\right) x_{ij} &+ \sum_{j \in J} \mu_j m_j + \pi_j p_j + \nu_j r_j + \epsilon_j   \label{obj} \\
\text{s.t.} & \sum_{j \in J} x_{ij} = 1 & \forall i \in I \label{assign} \\
        & \sum_{i \in I: u_i = 1} e_i x_{ij} \leq m_j & \forall j \in J \label{cap} \\
        & x_{ij} \in \{0,1\} & \forall i \in I, \forall j \in J \label{int} \end{aligned}$$
It is a Generalized Assignment Problem (GAP) [@gap]. Variables $x_{ij}$
take value $1$ if task $i$ is allocated to WA $j$, $0$ otherwise.
Constraints [\[assign\]](#assign){reference-type="eqref"
reference="assign"} impose that each task is assigned to exactly one WA.
Constraints [\[cap\]](#cap){reference-type="eqref" reference="cap"}
impose that the sum of effort of 'deadline' tasks assigned to the same
WA does not exceed the WA maximum productivity. Terms
[\[int\]](#int){reference-type="eqref" reference="int"} are integrality
conditions. The objective function encodes the linear regression model:
when $x_{ij} = 0$, the corresponding terms in the first double summation
are $0$, thus matching the definition of the predictor tuple terms. The
second summation is a constant, which does not affect the optimization
process. Model [\[obj\]](#obj){reference-type="eqref" reference="obj"}
-- [\[int\]](#int){reference-type="eqref" reference="int"} is therefore
maximizing the overall reputation update, which we assume to imply the
maximization of the score. We finally notice that constraints
[\[cap\]](#cap){reference-type="eqref" reference="cap"} tend to be
automatically respected by an implicit violation penalty, which is
learnt by the regression models. In fact, those players who violate them
produce tasks which are lost, thereby lowering the final WA reputation.

Computational evaluation {#sec:experiments}
========================

In [@dataset] a large set of logs has been collected, analyzed and
shared for public research. They come from real players of the Agile
Manager [@agile] online platform. More specifically, it contains details
of about 50000 play rounds in matches of either 5 or 10 rounds,
performed by 1141 players. Matches are of various types: a selection of
a few parameters is given to the player before the match starts. The
most important is a *difficulty* setting, which can simply be high or
low. In low difficult setting high quality output probability and
maximum productivity of WAs are directly proportional. In high
difficulty setting they are inversely proportional. We remark that no
randomization is performed on matches: given the same parameter and
level selection, the tasks required to be assigned in a specific round
remain the same. The combinatorics of the game, however, make it
difficult for a human player to perform optimal choices, even by
repeatedly playing the same level. Data were not always consistent, as
casual players probably tend to interrupt matches before their end. We
found instead data about the set of players with $30$ or more active
matches to be reliable. Therefore we have restricted our dataset to
their matches in our analyses. Levels 5 and 6 (the highest ones) were
the more representative ones, since 1-4 are likely played by casual
players. Level 5 is set to low difficulty, level 6 to high difficulty.
The final dataset consists of about $2000$ rounds of matches at levels 5
and 6, composing more than $200$ matches.

Data processing was performed by scripts in python 3.9. Regression
models exploit python scikit-learn library, and optimization model
implementations are based on the python-mip library, using CBC as a
mathematical programming solver. Tests were run on a notebook equipped
with a Ryzen 2400U 3.6GHz CPU and 16GB or RAM. On this setting, each
regression training took only a few seconds (except SVR whose training
required a few minutes), and each MIP optimization only fractions of
seconds.

Quality of regression models
----------------------------

In the first experiment we trained and compared different regression
models, namely linear regression (Linear), Ridge Regression (Ridge),
Stochastic Gradient Descent (SGD) and Support Vector Regression (SVR).
In Table [1](#tab:scores){reference-type="ref" reference="tab:scores"}
we report their average coefficient of determination $R^2$, when
considered to model the reputation update over the same round of all
matches. Unfortunately, the reputation update after round 10 is not
included in the source data; therefore we could not estimate the quality
of the models at round 10.

::: {#tab:scores}
    Round Linear     Ridge      SGD        SVR
  ------- ---------- ---------- ---------- ----------
        1 0.239429   0.239429   0.218955   
        2 0.796824   0.796824   0.792330   0.653779
        3 0.886585   0.886585   0.883128   0.775409
        4 0.927480   0.927479   0.925204   0.790667
        5 0.886406   0.886403   0.884115   0.764091
        6 0.914572   0.914569   0.913163   0.780832
        7 0.931019   0.931016   0.928841   0.776808
        8 0.946479   0.946476   0.945075   0.776849
        9 0.953715   0.953712   0.952456   0.788343

  : Score of different regression models
:::

The outcome is clear, and similar for all regression models: they
provide poor results in round 1, improving on round 2 and getting very
accurate from round 3 onwards. This is most probably due to the
contribution given by better starting reputations to rely on. These very
high score reflect an important characteristic of our game: WAs are
actually algorithms, whose logic is unknown but probably simple. As
such, their behaviour tends to be very predictable by barely looking at
data; however, data dimensionality and the combinatorics of the game
tends to make them hard to foresee for a human player. We finally report
SVR to provide lower scores than the other models, and to have
convergence problems in the training of round 1. Since Linear, Ridge and
SDG provide very similar scores we have restricted our efforts to
Linear, being also the simplest to be efficiently embedded in a MIP.

We conjectured the starting reputation to be a key feature for
prediction. Therefore we have also repeated these experiments by either
excluding it or keeping only such a feature. In both cases we obtained
much inferior performances. That is, starting reputation is actually a
key feature, but is not enough to be used alone for predictions. A
similar experiment on maximum WA workload, instead, showed it to
contribute but not being central for predictions.

Integration of regression models in mixed integer programs
----------------------------------------------------------

We then started to use the MIP models
[\[obj\]](#obj){reference-type="eqref" reference="obj"} --
[\[int\]](#int){reference-type="eqref" reference="int"}, integrating the
linear regression models trained as explained above, to play matches. As
already reported, we did not have direct access to the gaming software,
but only to the historical logs which have been produced by players.
Therefore, we could not perform a full comparison between human play and
autopilot MIP play over all rounds: the resulting internal status of WAs
after the rounds of a match can only be tracked on those match traces
fully contained on the logs. That is, MIPs would produce solutions whose
effect in the internal WAs status is unknown. We could however check the
behavior of MIPs in playing one round $k$, after rounds $1$ to $k-1$
which have been played by a human.

Hence, in a second experiment, we compared the performance of our models
in improving the reputation of WAs (and thus the overall score),
benchmarking them with those of human players. In Tables
[2](#tab:reputation5){reference-type="ref" reference="tab:reputation5"}
and [3](#tab:reputation6){reference-type="ref"
reference="tab:reputation6"} we report for both humans (left) and MIP
solutions (right) the average reputation obtained after each round, and
the corresponding variance. Table
[2](#tab:reputation5){reference-type="ref" reference="tab:reputation5"}
refers to matches at level 5, while Table
[3](#tab:reputation6){reference-type="ref" reference="tab:reputation6"}
refers to those at level 6. We remark that the reputation values
reported for MIP solutions are values predicted by our regression model.

::: {#tab:reputation5}
  ------- ------- ---------- ------- ----------
                                     
    Round Mean    Variance   Mean    Variance
        1 6.593   0.181      1.781   0.000
        2 6.780   0.166      5.201   0.106
        3 6.638   0.252      6.038   0.120
        4 6.471   0.337      6.347   0.205
        5 6.343   0.429      5.998   0.251
        6 6.243   0.523      6.083   0.343
        7 6.161   0.580      6.040   0.430
        8 6.107   0.609      6.030   0.514
        9 6.083   0.679      5.973   0.542
  ------- ------- ---------- ------- ----------

  : Average reputation, and variance, of WAs after each round at level 5
  played by both humans and MIPs (predicted).
:::

::: {#tab:reputation6}
  ------- ------- ---------- ------- ----------
                                     
    Round Mean    Variance   Mean    Variance
        1 6.439   0.205      1.513   0.000
        2 6.468   0.209      5.471   0.117
        3 6.218   0.245      5.764   0.148
        4 5.961   0.310      5.780   0.198
        5 5.793   0.364      5.221   0.223
        6 5.683   0.429      5.315   0.284
        7 5.576   0.474      5.308   0.352
        8 5.490   0.531      5.285   0.406
        9 5.413   0.538      5.262   0.468
  ------- ------- ---------- ------- ----------

  : Average reputation, and variance, of WAs after each round at level 6
  played by both humans and MIPs (predicted).
:::

That is, the main target of the experiment is to check if, once
different regression models are integrated into a single MIP for
optimization, their predictions remain consistent. In this regard our
experiments confirm our expectations. Level $6$ is by design more
difficult than level $5$, being required skill and effort directly
correlated. In fact both human players and MIPs produce slightly lower
reputations. Round $1$ is troublesome for MIPs, producing poor
reputation update w.r.t. human players. The gap is filled from round $3$
in matches of level $5$ and from round $4$ in those of level $6$,
showing regression models to need some additional data on the latter
setting to start being fully performant. From that point, MIPs
reputations are just a few percentage points lower than those of human
players.

Comparing optimization models with allocation policies
------------------------------------------------------

Finally, in an effort for assessing the potential of our methods in
automatic decision support tools, we have compared the performance of
our MIPs in terms of score produced, benchmarking with policies employed
by human players. In details, the Agile Manager platform itself asks to
the user which of the following policies was more similar to their play:
random (no specific pattern of assignment of tasks to WAs), load balance
(evenly distribute tasks to WAs by looking at tasks effort values),
reputation based (try to distribute first highly valued tasks to WAs
with high reputation). Such an indication is therefore purely
qualitative and retrieved only as a feedback from the player (e.g. by
random it is not meant to allocate tasks with uniform probabilities). We
also remark that each round of each level is the same in terms of given
tasks: while human players may non-deterministically play differently in
different matches, our MIPs deterministically play always in the same
way, which is that producing best predicted reputation score. To be as
challenging as possible in our benchmarking, we have therefore
considered for both level $5$ and level $6$ the match of the *best human
player* declaring to use a specific policy, that is those yielding to
*highest* scores in the logs. We report results for round $9$ (resp.
round $2$) as a case in which regression models have very high (resp.
barely fair) accuracy. We simulated by manual computations the full
solution structure obtained by both the best policy players and our MIP
models, following the logic of the WAs described in [@agile] and
[@dataset]. This produces a faithful representation on the outcome in
terms of value of tasks successfully completed.

Our results are reported in Table
[4](#tab:humansVSMIP){reference-type="ref" reference="tab:humansVSMIP"}
which includes in turn the match level, the reference assignment policy,
the corresponding value of successful tasks in both Round 2 and Round 9.

::: {#tab:humansVSMIP}
  Level   Policy             Round 2   Round 9
  ------- ------------------ --------- ---------
  5       MIP                53/89     31/89
          load balance       40/89     40/89
          random             30/89     30/89
          reputation based   35/89     71/89
  6       MIP                49/89     25/89
          load balance       40/89     40/89
          random             30/89     30/89
          reputation based   44/89     24/89

  : Value of tasks successfully completed in MIP and human solutions.
:::

As expected, reputation based policies tend to work best on level 5,
while load balance ones work best on level 6. The reason is the
following: when effort and difficulty are direcly proportional (level 5)
only few skilled WAs can accomplish complex tasks, which are at the same
time those having more stamina. When task complexity and effort are
inversely proportional (level 6), good allocations would need to assign
only complex tasks to skilled WAs. However, if these tend to be also
those having high reputation, reputation based policies tend to overload
them, thereby fully consuming their stamina too early; in this way
complex tasks appearing in last rounds cannot be completed.

At round 9 MIPs are not competitive with the best players, despite the
accuracy of inner regression models. It is particularly evident in level
5, where the structure of the specific instance makes a simple greedy
assignment, in order of complexity and reputation, the best choice:
players managing to have high reputation values from earlier rounds find
it easy to successfully complete the match. A simple reason making MIPs
not competitive can also be a mismatch between score and reputation,
that we are not able to assess from data. At round 2, instead, MIP was
producing better solutions than human players in both levels. A full
explanation of this phenomenon certainly requires more investigation. We
conjecture that, at round 9, players in successful matches had room to
learn how to apply good heuristic rules, while at round 2 more flat
reputation values and less experience in play make it harder to apply
good choices. Data driven MIP models, instead, can rely on the earlier
learning from a larger set of players, exploring different policies.

We also report a qualitative, but insightful, observation. MIP models
tend to allocate tasks to the weakest WA that can carry them out; this
helps in preserving stamina on the more skilled WAs. For instance, at
round 2 of level 5, MIP allocates tasks 20, 22 and 26, having difficulty
0.2, to WA number 2, which is that of minimum skill that can accomplish
them. This ability of detecting feasibility is actually matching
previous experiments on other combinatorial optimization problems
[@DDfeas].

Conclusions {#sec:conclusions}
===========

We have introduced data-driven and mathematical programming models, in
light of supporting decision making both in this specific game and in
more general crowdworking applications.

Our modeling choices show that indeed such an approach of embedding
regression models in a MIP, predicting a suitable performance proxy
given instance data, is promising. As expected, key steps in our work
were (a) identifying such a proxy (b) choosing a suitable feature space
(c) keeping the MIP model simple. We found it useful to keep also the
regression model as simple as possible: linear ones produced good
results in our experiments, while being at the same time easy to
formulate for the embedding in a MIP.

Concerning model testing, we have found our combination of regression
and MIP to keep on being predictive, even if one regression model is
trained for each worker agent, independently, and all of them are
subsequently combined in a single MIP: predicted reputations were just a
few percentage points away from real ones.

Concerning the comparison of MIP solutions to those of skilled human
players, we have found the latter to outperform the former at latest
rounds. Such a good behaviour of skilled human players at latest rounds
confirm that the combinatorics of this specific game still allow to
develop good heuristics, which we believe to be a prerequisite also for
our quantitative MIP approach to work well. However, the *right*
heuristic to apply is highly instance dependent: this is where our data
driven regression models prove useful.

At an early round of both levels we have analyzed, instead, we report
MIP solutions to allow more completed tasks than all solutions produced
by human players, showing our models to be promising in decision support
for these types of problems.

99.

Jäger, G., Zilian, L.S., Hofer, C. and Füllsack, M. "Crowdworking:
working with or against the crowd?". Journal of Economic Interaction and
Coordination 14, 761--788 (2019)

Han Yu, Zhiqi Shen, Chunyan Miao, Cyril Leung, Yiqiang Chen, Simon
Fauvel, Jun Lin, Lizhen Cui, Zhengxiang Pan, Qiang Yang "A dataset of
human decision-making in teamwork management", Scientific Data 4 (2017)

Han Yu, Xinjia Yu, Su Fang Lim, Jun Lin, Zhiqi Shen, Chunyan Miao "A
multi-agent game for studying human decision-making", Proceedings of the
2014 international conference on Autonomous agents and multi-agent
systems (2014)

Yoshua Bengio, Andrea Lodi, Antoine Prouvost "Machine learning for
combinatorial optimization: A methodological tour d'horizon", European
Journal of Operational Research, Volume 290, Issue 2 (2021)

T. Öncan "A survey of the generalized assignment problem and its
applications", INFOR 45 (2007)

Marco Casazza, Alberto Ceselli "Heuristic Data-Driven Feasibility on
Integrated Planning and Scheduling", Proc. of ODS -- Advances in
Optimization and Decision Science for Society, Services and Enterprises
(2019)
