# iLQR-Diffuser: Diffusion Trajectory Generation with iLQR-Based Dynamic Feasibility
In this project, we propose a novel diffusion-based trajectory generation architecture that produces **dynamically feasible** robot trajectories by integrating a **multiple-shooting iLQR projection** step into both training and inference of a diffusion transformer.
Our approach combines a diffusion transformer with a multiple-shooting iLQR projection layer that enforces consistency with robot dynamics and actuation limits.


## Model specifications

Each diffusion transformer model is uniquely identified by a list of several parameters composing its name:
```
<<modality>>_<<conditioning>>_ODE_<<env_name>>_<<projector_name>>_<<projection_curriculum>>_specs_<<DiT_size>>.pt
```
- modality : whether the model predicts only states `S`, states and actions `SA`, or only actions `A`.
- conditioning : whether the model is conditioned `Cond`. All action-only models are conditioned on the initial state $s_0$. The GO1 and GO2 models are also conditioned on the velocity command.
- env_name : name of the environment either `Hopper`, `Walker`, `HalfCheetah`, `Quadcopter`, `GO1`, or `GO2`.
- projector_name : name of the projection scheme with which the diffusion model has been trained. Either `Ref_proj` for the reference projector, `A_proj` for the action projector, or `SA_proj` for the state-action projector with learned feedback correction. More details in the paper.
- projection_curriculum : the noise levels between which partial projections occur, typically `sigma_0.0021_0.2` as the lowest noise level of the DiT is `0.002`. More details in the paper.

## Code

You can load and evaluate pre-trained models with `evaluate.py`.

Training new models simply requires running `train.py`.


The diffusion transformer model is located in folder `DiT` comprising the transformer blocks in `DiT.py`, the diffusion model in `ODE.py`, and the planner to generate trajectories in `planner.py`.

The projectors are found in `utils/projectors.py`.

Each environment has its own folder containing the environment code and plotting functions along with dataset of trajectories and pre-trained models with the following structure.

```bash
.
└── <<env_name>>
    ├── <<env_name>>.py # code for the Gym-like environment
    ├── plots.py        # plotting functions specific to the environment
    ├── datasets
    │   └── <<env_name>>_<<N>>trajs_<<H>>steps.npz # dataset of N state-action trajectories of horizon H.
    └── trained_models
        └── <<modality>>_<<conditioning>>_ODE_<<env_name>>_<<projector_name>>_<<projection_curriculum>>_specs_<<DiT_size>>.pt 
```

## Citation
```
TBD
```

## Acknowledgments

Our code is largely modified from source code based on  [DDAT](https://github.com/labicon/DDAT).

