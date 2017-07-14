# WavePI Todo List

## General 

- [ ] Output of steps (settings?)
- [ ] "Abort now" like in old code
- [ ] Measurements
- [ ] multiple RHS 

## Direct Solver

- [ ] grid adaptivity (decide on a structure, then all those `DoFHandler*`s have to be replaced 
- [ ] Interpolation of time steps (`DiscretizedFunction::at`)
- [ ] adjointness still not good, cg unstable when using few time steps (scalar product inconsistent with crank-nicolson?)
- [ ] find a better way of treating multiple boundaries

## Inversion

- [ ] maybe look at cg-directions visually to figure out where it goes wrong when unstable
- [ ] linear Tikhonov (using "Tikhonov-CG"?)
- [ ] Adjoints for a, nu and q (base problem class for linearizations? `LinearizedWaveProblem`)
- [ ] **implement stop criteria in all regularization methods** how to communicate this to the caller?
- [ ] REGINN tolerance choice (-> class)
- [ ] REGINN: if linear method does not converge (diverge or maximum iterations reached) increase tolerance and try again
- [ ] add Shrinkage step to Landweber

## Documentation and Tests

- [ ] better documentation