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
- [x] `WaveEquationBase` (idea dismissed because it would make code unreadable)
- [ ] grid adaptivity (decide on a implementation, change code to use transfer functions (add them to `SpaceTimeMesh`)  
- [ ] Interpolation of time steps (`DiscretizedFunction::at`)
- [ ] find a better way of treating multiple boundaries
- [ ] Adjointness with ν≠0 is not so good

## Inversion

- [x] implement stop criteria in all regularization methods
- [x] cg still unstable (solved: L2ProductRightHandSide is evil)
- [ ] linear Tikhonov (using "Tikhonov-CG"?)
- [ ] Adjoints for a, nu and q (base problem class for linearizations? `LinearizedWaveProblem`)
- [ ] ** REGINN: tolerance choice ** (-> abstract class, implement strategy from Rieder's book/paper) 
- [ ] REGINN: improve tolerance choice (if linear method does not converge because of maximum iter count increase tolerance and try again)
- [ ] add Shrinkage step to Landweber

## Documentation and Tests

- [x] restructure and clean up tests
- [x] tests `solve_mass` and `mul_mass`
- [ ] better documentation