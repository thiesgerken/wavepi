# WavePI Todo List

## General 

- [x] organize code into sub-namespaces (`direct`, `inversion`, ...)
- [x] `DiscretizedFunction::discretize` should have an overload for a time vector and a single dofhandler
- [x] `NonlinearProblem::get_derivative` should also get the current solution of the forward problem
- [x] class `QLinearizedProblem` in `test_inverse.cpp`
- [ ] watch dog for linear methods
- [ ] Maybe for Products of Basis functions as RHS you need a better quadrature rule (it is a higher order polynomial)
- [ ] Move functions in `L2ProductRightHandSide.cpp` into the class (like `L2RightHandSide`) 
- [ ] Logging for Landweber, Output of steps (settings?)
- [ ] "Abort now" like in old code
- [ ] Measurements
- [ ] multiple RHS 
- [ ] `shared_ptr` instead of regular pointers everywhere
- [ ] copy constructors should have `const` qualifiers for args 
- [ ] norms are not really l2 norms (only for uniform grids, but I guess this does not depend on the FE-degree?)

## Direct Solver

- [ ] grid adaptivity
- [ ] Interpolation of time steps (`DiscretizedFunction::at`)

## Inversion

- [x] implement $`L^2`$-Landweber scheme (as  a class?) $`c_{k+1} = c_k + \omega (S' c_k)^* (g - S c_k)`$
- [ ] add Shrinkage step
- [x] base class: IterativeRegularization (virtual calculateStep = 0, start, test) and maybe even Regularization
- [x] Adjoint: use $`L^2`$ (almost self-adjoint, integrate backward in time) 
- [x] after that REGINN (+ LinearRegularization + some possibilities for that (CG, LW) ) 
- [ ] make REGINN-CG work (CG has a problem)
- [ ] REGINN tolerance choice (-> class)
- [ ] gradient method instead of conjugate gradients (maybe more robust?)
- [ ] check adjointness? (for P2-elements it seems to work, why?)
- [ ] maybe look at cg-directions visually to figure out where it goes wrong (boundary?)
- [ ] linear Tikhonov (using "Tikhonov-CG"?)
- [ ] Adjoints for a, nu and q (base problem class for linearizations? `LinearizedWaveProblem`)

## Documentation and Tests

- [ ] solution for discretized parameters+rhs the same as when passing those as functions (have to obfuscate that they are DiscretizedFunctions)  
- [ ] Class documentation