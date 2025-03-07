---
title: 'pyMassEvac: A Python package for simulating multi-domain mass evacuation operations'
tags:
  - Python
  - evacuation
  - Arctic
  - security
  - Reinforcement Learning
authors:
  - name: Mark Rempel
    orcid: 0000-0002-6248-1722
    corresponding: true
    affiliation: 1 
affiliations:
 - name: Defence Research and Development Canada, Canada
   index: 1
date: 7 March 2025
bibliography: paper.bib

---

# Summary

`pyMassEvac` is a Python package whose aim is to study mass evacuation 
scenarios. In particular, it is designed to simulate single- and multi-domain 
mass evacuation operations in which: 

- the individuals to be evacuated are at a remote location, such as in
the Arctic, where access to immediate medical care is limited or non-existent; 
- each individual's medical condition may change over time, perhaps 
due to environmental conditions, injury, or care being provided; and 
- the individuals must be transported from the evacuation site to a Forward 
Operating Location (FOL).

An example of a multi-domain mass evacuation operation, where the objective is 
to maximize the number of lives saved by transporting individuals to an FOL, is depicted in \autoref{example}.

![Evacuation plan via air with medical assistance provided at the evacuation site via ship. Colours of individuals at the evacuation site represent those in different triage categories (white, green, yellow, red, black; black represents deceased). For a full description, see @rempel2024a.\label{example}](arctic_map_mass_evac_joss.png)

Within this context, `pyMassEvac` may be used to provide decision support to 
defence and security planners in two ways. First, through exploring the impact 
of policies used to make the three decisions depicted in \autoref{example} (see 
right panel):

- **Decision policy 1**: the policy that determines which individuals are 
loaded onto a vehicle, such as a helicopter, for transport from the evacuation 
site to the FOL; 
- **Decision policy 2**: the policy that determines which individuals receive 
medical care (if available) at the evacuation site, such as onboard a nearby 
ship; and 
- **Decision policy 3**: the policy that determines which individuals are 
removed from the group receiving medical care, for reasons such as limited 
capacity or that the individuals' medical condition has been sufficiently 
improved, and returned to the group ready to be transported to the FOL. 

Second, assuming decision policies are selected, decision support may be 
provided by using `pyMassEvac` to explore the selected policies' robustness
to changes in a scenario's parameters. For example, `pyMassEvac` may be used 
to explore how robust a set of decision policies are in terms of 
the number of lives saved with respect to:

- the initial arrival time of one or more transport vehicles after the 
individuals have arrived at the evacuation site; 
- the travel time between the evacuation site and the FOL; and 
- the rate at which an individual's medical condition becomes better (through 
receiving medical care) or worse (due to injury or exposure to environmental 
conditions) over time.

Changes in a scenario's parameters from baseline values may reflect a variety 
of real-world strategic and operational decisions beyond the tactical decisions 
made within scenario itself. For example:

- the reduction in the initial arrival time of transport vehicles 
may reflect an operational decision to pre-position vehicles during the summer 
season;
- the reduction in the travel time between the evacuation site and FOL may reflect 
a strategic decision to build a new aerodrome; and
- the decrease in the rate at which an individual's medical condition worsens
may reflect an operational decision to invest in improved medical kit.

Thus, `pyMassEvac` is designed to be primarily used by operational researchers who study humanitarian or defence and security operations.

`pyMassEvac` is accessible at `https://github.com/mrempel/pyMassEvac` and is 
installed via a `setup.py` script. In addition, published evacuation scenarios 
that have studied using this package (or one of its earlier developmental versions) 
are described in @rempel2021a, @rempel2023a, and @rempel2024a.

# Statement of need

The significant decrease in Arctic sea ice in recent decades has resulted in 
increased activity in the Arctic across a range of sectors, such as oil and 
gas, mining, fishing, and tourism. With respect to tourism and the potential 
increase in the number of Arctic cruise ships, Arctic nations are concerned 
with both the potential increase in the number of Search and Rescue (SAR) 
incidents that may occur and the increased size of those incidents in terms 
of the number of individuals in need of evacuation. This is evidenced by 
recent exercises that have been conducted, such as the SARex series in 
Norway [@solberg2016a; @solberg2018a], a table-top exercise including the 
United States, Canada, and the cruise ship industry [@mcnutt2016a], and 
NANOOK-TATIGIT 21 by the Canadian Armed Forces [@nationaldefence2021a].

While software exists to support planning for and executing evacuation 
operations, this software either requires a paid license [@sarresponse; 
@massevac], focuses on search planning [@sarops], or addressed specific 
situations such as wildfires [@guman2024a]. With this in mind, `pyMassEvac` 
aims to provide an open source software package that enables researchers 
to both assess the impact of strategic and operational decisions
made prior to an evacuation operation occurring, as well as the impact of
tactical decisions made within the operation itself. 

# Features

## Defining an evacuation operation

Mass evacuation operations are modelled in `pyMassEvac` as a sequential 
decision problem under uncertainty using Powell's universal framework for 
sequential decisions [@powell2022a]. See Section 4 of @rempel2024a for the
complete description of the model. Given this framework, a scenario's 
parameters are specified via the initial state variable $S_0$, which 
consists of the following elements:

- $m^e$: Vector of mean time (hours) for an individual's medical condition
to worsen and transition from a triage category $t \in \mathcal{T} \setminus \{b\}$ 
to the next lower triage category $t^\prime \in \mathcal{T} \setminus \{w\}$ at the evacuation 
site, i.e., $m^e_w$ is the mean transition time from the white ($w$) to green ($g$) tag category. 
The set of triage categories is given as $\mathcal{T} = \{w, g, y, r, b\}$; 
- $m^s$: Vector of mean time (hours) for an individual's medical condition to
improve and transition from a triage category $t \in \mathcal{T} \setminus \{w, b\}$ to the next 
higher triage category $t^\prime \in \mathcal{T} \setminus \{r, b\}$ while receiving medical care, 
i.e., $m^s_r$ is the mean transition time from the red ($r$) to yellow ($y$) tag category;
- $c^h$: Total capacity for individuals onboard a transport vehicle, such as a helicopter;
- $c^s$: Total capacity for individuals to receive medical care, such as onboard a ship;
- $\delta^h$: Vector of capacity consumed by each triage category $t \in 
\mathcal{T} \setminus \{b\}$ onboard a transport vehicle. Individual in the black ($b$) tag
category are not transported as they are deceased and are assumed to be 
recovered at the end of the rescue operation;
- $\delta^s$: Vector of capacity consumed by each triage category $t \in 
\mathcal{T} \setminus \{b\}$ when receiving medical care;
- $\eta^h$: Total time (hours) for a transport vehicle to load individuals at the evacuation
site, transport them to the FOL, unload the individuals, and return
to the evacuation site;
- $\eta^{sl}$: Total time (hours) to transfer individuals at the evacuation site to the 
local facility (such as a ship) in which they will receive medical care, plus 
the time until a decision is made as to which individuals to transfer back to 
the evacuation site;
- $\eta^{su}$: Total time to transfer individuals from the local facility (such 
as a ship) in which they are receiving medical care to the evacuation site, 
plus the time until a decision is made as to which individuals to transport to 
the FOL;
- $\tau^h$: Vector of initial arrival time (hours) of each transport vehicle after the 
individuals have arrived at the evacuation site; and
- $\tau^s$: Vector of initial arrival time (hours) of each medical care facility (such 
as a ship) after the individuals have arrived at the evacuation site.

Note that the initial state in `pyMassEvac` differs from @rempel2024a as this package
includes both $\tau^h$ and $\tau^s$. In @rempel2024a these two parameters were specified
separately in the case study presented in Section 5.

An example of an initial state, with one transport vehicle and one medical care facility, 
is given in the tutorial found in `tutorial\tutorial.ipynb`.

## Example decision policies

`pyMassEvac` provides a set of decision policies that implements those 
described in @rempel2024a. All policies are defined in 
`mass_evacuation_policy.py` and are summarized as follows:

- `green_first_loading_policy`: This policy may be used for either
**Decision policy 1** or **Decision policy 2** and puts an emphasis 
on loading healthier individuals prior to those with worse medical
conditions;
- `yellow_first_loading_policy`: This policy is similar to the
green-first loading policy, with the exception that it focuses on 
those individuals that require near-term care, followed by those in
descending order in triage category. This policy may be used for
either **Decision policy 1** or **Decision policy 2**;
- `critical_first_loading_policy`: This policy prioritizes those
individuals that require immediate attention before moving onto
less critical categories. This policy may be used for
either **Decision policy 1** or **Decision policy 2**;
- `random_loading_policy`: This policy randomly selects individuals,
regardless of their triage category. This policy may be used for
either **Decision policy 1** or **Decision policy 2**;
- `random_unloading_policy`: This policy randomly selects individuals,
regardless of their triage category. This policy may be used for
**Decision policy 3**; and
- `white_unloading_policy`: This policy only removes individuals from 
the medical facility whose medical condition has improved such that they
are assigned a white ($w$) tag. This policy may be used for **Decision policy 3**.

In addition, a `do_nothing` policy is provided to model situations in which a
decision is to be delayed or a model the lack of transport of medical care.

The tutorial found in `tutorial\tutorial.ipynb` demonstrates how to use
these decision policies. Specifically, it uses the 
`green_first_loading_policy` for **Decision policy 1** and 
**Decision policy 2**, and the `white_unloading_policy` for
**Decision policy 3**.

## Ready for reinforcement learning

`pyMassEvac` is implemented as a custom Gymnasium environment [@towers2024a].
An example of its use as an environment with fixed decision policies is 
provided in `tutorial\tutorial.ipynb`. `pyMassEvac` may also be used in 
combination with a reinforcement learning to seek optimal, or at least 
near-optimal, decision policies. Among the many considerations that must be 
made when selecting or designing a learning algorithm for this environment 
is that the set of valid actions are dependent on both the state variable 
$S_k$ and the parameters defined in the initial state $S_0$---see Section 
4.1 of @rempel2024a. Thus, when using a reinforcement learning algorithm a 
form of invalid action masking [@huang2022a; @hou2023a] must be implemented.

# Acknowledgements

I acknowledge contributions from both Nicholi Shiell and Kaeden Tessier, who 
are co-authors on related papers. These collaborations inspired the development 
of this package.

# References
