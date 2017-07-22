# WavePI Todo List

## General 

- [x] `solve_space_time_mass` and same function for only time
- [x] logfilter
- [ ] Output of steps (settings?)
- [ ] "Abort now" like in old code
- [ ] Measurements
- [ ] multiple RHS 

## Direct Solver

- [x] adjointness of `WaveEquationAdjoint` is way better, but seems to have an implementation error (complete comments)
- [ ] grid adaptivity (decide on a structure, then all those `DoFHandler*`s have to be replaced 
- [ ] Interpolation of time steps (`DiscretizedFunction::at`)
- [ ] find a better way of treating multiple boundaries
- [ ] `WaveEquationBase` with getters/setters and so on for Parameters and RHS as well as assembly of matrices A, B and C

## Inversion

- [x] implement stop criteria in all regularization methods
- [x] cg still unstable (solved: L2ProductRightHandSide is evil)
- [ ] Adjointness with nu /= 0 is not so good
- [ ] linear Tikhonov (using "Tikhonov-CG"?)
- [ ] Adjoints for a, nu and q (base problem class for linearizations? `LinearizedWaveProblem`)
- [ ] ** REGINN: tolerance choice ** (-> abstract class, implement strategy from Rieder's book/paper) 
- [ ] REGINN: improve tolerance choice (if linear method does not converge because of maximum iter count increase tolerance and try again)
- [ ] add Shrinkage step to Landweber

## Documentation and Tests

- [x] restructure and clean up tests
- [x] tests `solve_mass` and `mul_mass`
- [ ] better documentation