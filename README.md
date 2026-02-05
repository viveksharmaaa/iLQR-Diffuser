# iLQR-Diff: Diffusion policies enforcing Dynamically Admissible robot Trajectories

In this project we create a novel diffusion architecture to generate dynamically feasible robot trajectories by incorporating autoregressive projections in the training and inference phase of a diffusion transformer.


## Overview

Diffusion models are stochastic by nature.
Thus, the trajectories they generate **cannot** satisfy exactly the equations of motions of robots.
When deploying such *infeasible* trajectories, the actual robot diverges from the prediction and most likely fails to accomplish its task.
Previous works have thus focused on replanning the entire trajectory very frequently.
We propose to address the root cause of the problem by forcing our diffusion models to generate **feasible** trajectories.


## Theory

We assume to have a black-box discrete-time simulator $f$ of a robot state 

$$s_{t+1} = f(s_t, a_t)$$

controlled by actions $a_t \in \mathcal A$.
A trajectory $\tau = \\{s_0, s_1, ...\\}$ is **admissible** if each $s_{t+1}$ is reachable from its predecessor $s_t$ with an action $a_t \in \mathcal{A}$. In other words $s_{t+1}$ must be in the **reachable set** $\mathcal{R}(s_t)$ of its predecessor $s_t$ where

$$\mathcal{R}(s_t) = \\{ f(s_t, a)\ \text{for all}\ a \in \mathcal{A} \\}.$$

To generate admissible trajectories we then design autoregressive projectors $\mathcal P$ iteratively projecting each $s_{t+1}$ onto $\mathcal{R}(s_t)$ for all $t$.
These projectors are incorporated into the training and inference of our diffusion models.

## Organization

- `code` : our implementation of iLQR with diffusion transformers, and trained models.

## Acknowledgments

Our code is largely modified from source code based on  [DDAT](https://github.com/labicon/DDAT).
