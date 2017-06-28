# WavePI Todo List

## Direct Solver

* grid adaptivity
* discretized parameters and rhs should also work in WaveEquation (Parameter class and subclasses DiscretizedParameter, ContinuousParameter? take references to parameters as arguments in WaveEquation? a and q must be in the same fashion?)
* Interpolation of time steps (DiscretizedFunction::at)

## Inversion

* implement L^2-Landweber / Shrinkage scheme (as  a class?) 
