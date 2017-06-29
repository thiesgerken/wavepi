# WavePI Todo List

## Direct Solver

* grid adaptivity
* discretized parameters and rhs should also work in WaveEquation (Parameter class and subclasses DiscretizedParameter, ContinuousParameter? take references to parameters as arguments in WaveEquation? a and q must be in the same fashion?)
* rhs in H^{-1} has to be accepted -> DistributionRightHandSide with (f1,v)+(f2,nabla v)
* Interpolation of time steps (DiscretizedFunction::at)
* WaveEq: give more than a step size, but the whole discretization instead (-> index discrete params by this?)

## Inversion

* implement L^2-Landweber / Shrinkage scheme (as  a class?) 
