---
title: 'pyMassEvac: A Python package for simulating multi-domain mass evacuation scenarios'
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
date: 27 February 2025
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
to maximize the number of lives saved by transporting them to the FOL, that may 
be modelled using `pyMassEvac` is described in @rempel2024a and is depicted in \autoref{fig:example}.

![Evacuation plan via air with medical assistance provided via ship\label{example}](arctic_map_mass_evac_joss.png)

Within this context, `pyMassEvac` may be used to provide decision support to 
defence and security planners in two ways. First, through exploring the impact 
of the policies to make the three decisions depicted in \autoref{example}:

- **Decision policy 1**: the policy that determines which individuals are loaded onto a vehicle, 
such as a helicopter, for transport to the FOL; 
- **Decision policy 2**: the policy that determines which individuals receive medical care (if 
available) at the evacuation site, such as onboard a nearby ship; and 
- **Decision policy 3**: the policy that determines which individuals are removed from the group 
receiving medical care, for reasons such as limited capacity or that the individuals'
medical condition has been sufficiently improved, and returned to the group ready to be 
transported to the FOL. 

Second, given a set three decision policies, decision support may be provided by using 
`pyMassEvac` to explore their robustness by modifying a scenario's parameters. For example,
`pyMassEvac` may be used to explore how robust a set of decision policies is in terms of 
the number of lives saved with respect to:

- the arrival time of the initial transport vehicle after the individuals have arrived at
the evacuation site; 
- the distance, and thus travel time, between the evacuation site and the FOL; and 
- the rate at which an individual's medical condition becomes better (through receiving
medical care) or worse (due to injury or exposure to environmental conditions) over time.

Changes in such parameters from baseline values may reflect a variety of real-world events, such as:

- the reduction in the arrival time of the initial transport vehicle 
may reflect the pre-positioning of vehicles during the summer season;
- the reduction in the distance between the evacuation site and FOL may reflect 
the building a new aerodrome; and
- the decrease in the rate at which an individual's medical condition worsens
may reflect the use of improved medical kit.

Thus, `pyMassEvac` is designed to be primarily used by operational researchers who study humanitarian or defence and security operations.

`pyMassEvac` is accessible at `https://github.com/mrempel/pyMassEvac` and is 
installed via a `setup.py` script. In addition, published evacuation scenarios 
that have studied using this package (or its earlier developmental versions) 
are described in @rempel2021a, @rempel2023a, and @rempel2024a.

# Statement of need

The significant decrease in Arctic sea ice in recent decades has resulted in increased
activity in the Arctic across a range of sectors, such as oil and gas, mining, fishing, 
and tourism. As the ability to navigate the Arctic's primary sea routes---the Northwest passage, 
Northern Sea Route, and Transpolar Sea Route (see the left panel of \autoref{fig:example})---
becomes more commonplace, their use for both trade and the transport of individuals will follow. In regard to the transport of individuals, for example via cruise ships, Arctic nations are concerned with both the potential increase in the number of Search and Rescue (SAR) incidents that may occur, and the increased size of those incidents in terms of the number of individuals in need of evacuation. This is evidenced by recent exercises that have been conducted, such as the SARex series in Norway @solberg2016a; @solberg2018a, a table-top exercise including the United States, Canada, and the cruise ship industry @mcnutt2016a, and NANOOK-TATIGIT 21 by the Canadian Armed Forces @nationaldefence2021a.

- "mass evacuation" "software" - review what MassEvac can do and how does it not fit this need?
- reference Camur (2021)

While software exists to support planning for and executing evacuation operations, this software either requires a paid license, does not enable a researcher to study the impact of different decision policies, or ... 

With this in mind, `pyMassEvac` aims to enable researchers to study the ...

# Features



Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Limitations

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"



# Acknowledgements

I acknowledge contributions from Nicholi Shiell and Kaeden Tessier who are co-authors 
on related papers. These collaborations inspired the development of this package.

# References
