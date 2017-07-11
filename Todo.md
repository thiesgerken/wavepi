# WavePI Todo List

## General 

- [x] organize code into sub-namespaces (`direct`, `inversion`, ...)
- [x] `DiscretizedFunction::discretize` should have an overload for a time vector and a single dofhandler
- [x] `NonlinearProblem::get_derivative` should also get the current solution of the forward problem
- [x] class `QLinearizedProblem` in `test_inverse.cpp`
- [x] `shared_ptr` instead of regular pointers (almost) everywhere
- [x] Maybe for Products of Basis functions as RHS you need a better quadrature rule (it is a higher order polynomial)
- [x] Move functions in `L2ProductRightHandSide.cpp` into the class (like `L2RightHandSide`) 
- [x] copy constructors should have `const` qualifiers for args 
- [ ] Output of steps (settings?)
- [ ] Move assignment operator for `DiscretizedFunction`
- [ ] "Abort now" like in old code
- [ ] Measurements
- [ ] multiple RHS 
- [ ] inverse problem functions (forward, adjoint) should expect pointers (they have to save `DiscretizedFunction`s)
- [ ] norms are not really l2 norms (only for uniform grids, but I guess this does not depend on the FE-degree?)
- [ ] watch dog for linear methods

## Direct Solver

- [ ] grid adaptivity (decide on a structure, then all those `DoFHandler*`s have to be replaced 
- [ ] Interpolation of time steps (`DiscretizedFunction::at`)

## Inversion

- [x] implement $`L^2`$-Landweber scheme (as  a class?) $`c_{k+1} = c_k + \omega (S' c_k)^* (g - S c_k)`$
- [x] base class: IterativeRegularization (virtual calculateStep = 0, start, test) and maybe even Regularization
- [x] Adjoint: use $`L^2`$ (almost self-adjoint, integrate backward in time) 
- [x] after that REGINN (+ LinearRegularization + some possibilities for that (CG, LW) ) 
- [ ] add Shrinkage step to Landweber
- [ ] make REGINN-CG work (CG has a problem)
- [ ] REGINN tolerance choice (-> class)
- [ ] gradient method instead of conjugate gradients (maybe more robust?)
- [ ] check adjointness? (for P2-elements it seems to work, why?)
- [ ] maybe look at cg-directions visually to figure out where it goes wrong (boundary?)
- [ ] linear Tikhonov (using "Tikhonov-CG"?)
- [ ] Adjoints for a, nu and q (base problem class for linearizations? `LinearizedWaveProblem`)

## Documentation and Tests

- [x] solution for discretized parameters+rhs the same as when passing those as functions (have to obfuscate that they are DiscretizedFunctions) all params (q and nu as well, make their influences higher, variations of q and a as well (to check all possibilities)
- [ ] test of adjointness
- [ ] Class documentation