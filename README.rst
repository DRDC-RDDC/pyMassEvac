==========
pyMassEvac
==========

|license| |joss|

.. |license| image:: https://img.shields.io/github/license/DRDC-RDDC/pyMassEvac
   :alt: GitHub License

.. |joss| image:: https://joss.theoj.org/papers/26c55ba4b5fd0de04ebdddde202142d5/status.svg
    :alt: joss
    :target: https://joss.theoj.org/papers/26c55ba4b5fd0de04ebdddde202142d5


**pyMassEvac** is a Python library that aims to study single- and multi-domain mass evacuation operations. While a wide range of operations exist within this context, **pyMassEvac** is specifically designed to study operations in which:

* the individuals to be evacuated are at a remote location, such as in the Arctic, where access to immediate medical care is limited or non-existent; 
* each individual's medical condition may change over time, perhaps due to environmental conditions, injury, or care being provided; and 
* the individuals must be transported from the evacuation site to a Forward Operating Location (FOL).

**pyMassEvac** models these operations using Powell's universal modelling framework for sequential decisions, with the full mathematical model is described in `this paper <https://www.sciencedirect.com/science/article/abs/pii/S0925753523003211>`_. This implementation is intended to be primarily used by operational researchers in two ways: (i) to study the impact of various decision policies; and (ii) to study the robustness of selected decision policies to the operation's initial state.

Decision policies
=================

**pyMassEvac** may be used to study the impact of three decision policies within the mass evacuation operation. These policies are:

* the policy that selects which individuals are loaded onto a vehicle for transport from the evacuation site to the FOL; 
* the policy that determines which individuals receive medical care (if available) at the evacuation site; and
* the policy that determines which individuals are removed from the group receiving medical care and returned to the group ready to be transported to the FOL. 

Example decision policies may be found in :code:`pyMassEvac/mass_evacuation_policy.py`.

Initial state
=============

Assuming decision policies are selected, **pyMassEvac** may be used to explore the selected policies' robustness to changes in a scenario's parameters, e.g., how robust a set of decision policies is in terms of the number of lives saved with respect to:

* the initial arrival time of one or more transport vehicles after the individuals have arrived at the evacuation site; 
* the travel time between the evacuation site and the FOL; and 
* the rate at which an individual's medical condition becomes better or worse over time.

An operation's initial state is defined by 11 parameters. These are defined as follows:

* $m^e$: Vector of mean time (hours) for an individual's medical condition to worsen and transition from a triage category $t \\in \\mathcal{T} \\setminus \\{b\\}$ to the next lower triage category $t^\\prime \\in \\mathcal{T} \\setminus \\{w\\}$ at the evacuation site, i.e., $m^e_w$ is the mean transition time from the white ($w$) to green ($g$) tag category. The set of triage categories is given as $\\mathcal{T} = \\{w, g, y, r, b\\}$; 
* $m^s$: Vector of mean time (hours) for an individual's medical condition to improve and transition from a triage category $t \\in \\mathcal{T} \\setminus \\{w, b\\}$ to the next higher triage category $t^\\prime \\in \\mathcal{T} \\setminus \\{r, b\\}$ while receiving medical care, i.e., $m^s_r$ is the mean transition time from the red ($r$) to yellow ($y$) tag category;
* $c^h$: Total capacity for individuals onboard a transport vehicle, such as a helicopter;
* $c^s$: Total capacity for individuals to receive medical care, such as onboard a ship;
* $\\delta^h$: Vector of capacity consumed by each triage category $t \\in \\mathcal{T} \\setminus \\{b\\}$ onboard a transport vehicle. Individual in the black ($b$) tag category are not transported as they are deceased and are assumed to be recovered at the end of the rescue operation;
* $\\delta^s$: Vector of capacity consumed by each triage category $t \\in \\mathcal{T} \\setminus \\{b\\}$ when receiving medical care;
* $\\eta^h$: Total time (hours) for a transport vehicle to load individuals at the evacuation site, transport them to the FOL, unload the individuals, and return to the evacuation site;
* $\\eta^{sl}$: Total time (hours) to transfer individuals at the evacuation site to the local facility (such as a ship) in which they will receive medical care, plus the time until a decision is made as to which individuals to transfer back to the evacuation site;
* $\\eta^{su}$: Total time to transfer individuals from the local facility (such as a ship) in which they are receiving medical care to the evacuation site, plus the time until a decision is made as to which individuals to transport to the FOL;
* $\\tau^h$: Vector of initial arrival time (hours) of each transport vehicle after the individuals have arrived at the evacuation site; and
* $\\tau^s$: Vector of initial arrival time (hours) of each medical care facility (such as a ship) after the individuals have arrived at the evacuation site.

Installation
############

**pyMassEvac** can be cloned from GitHub. Doing so will enable all dependencies to be installed automatically.

.. code-block::

    git clone https://github.com/DRDC-RDDC/pyMassEvac.git
    cd pyMassEvac
    conda env create -f environment.yml
    conda activate pyMassEvac
    pip install -e .

Example usage
#############

Please refer to the `tutorial directory <https://github.com/DRDC-RDDC/pyMassEvac/blob/master/tutorial>`_ for a tutorial on how to use **pyMassEvac** to model a mass evacuation scenario and study the effectiveness of a set of decision policies.

Contributing
############

For major changes, please open an issue first to discuss what you would like to change. 
For more details, `click here <https://github.com/DRDC-RDDC/pyMassEvac/blob/master/CONTRIBUTING.md>`_.

Contact
#######

For any questions, feedback, bug reports, feature requests, etc. please
first present your thoughts via GitHub issues. For further assistance
please contact mark.rempel@forces.gc.ca.

Copyright
#####################

Copyright (c) His Majesty the King in Right of Canada, as represented
by the Minister of National Defence, 2025.

