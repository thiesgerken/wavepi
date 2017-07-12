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
- [x] norms are not really l2 norms 
- [x] `DiscretizedFunction`: settings for what `norm()` returns (also what kind of scalar product, if any)
- [x] `DiscretizedFunction`: scalar product 
- [x] Move assignment operator for `DiscretizedFunction`
- [x] organize includes
- [ ] Output of steps (settings?)
- [ ] "Abort now" like in old code
- [ ] Measurements
- [ ] multiple RHS 
- [ ] inverse problem functions (forward, adjoint) should expect pointers (they have to save `DiscretizedFunction`s)
- [ ] watch dog for linear methods
- [ ] **adjointness still not good** ...

## Direct Solver

- [ ] grid adaptivity (decide on a structure, then all those `DoFHandler*`s have to be replaced 
- [ ] Interpolation of time steps (`DiscretizedFunction::at`)

## Inversion

- [x] implement $`L^2`$-Landweber scheme (as  a class?) $`c_{k+1} = c_k + \omega (S' c_k)^* (g - S c_k)`$
- [x] base class: IterativeRegularization (virtual calculateStep = 0, start, test) and maybe even Regularization
- [x] Adjoint: use $`L^2`$ (almost self-adjoint, integrate backward in time) 
- [x] after that REGINN (+ LinearRegularization + some possibilities for that (CG, LW) ) 
- [ ] add Shrinkage step to Landweber
- [ ] ** Linear Landweber: remove initial guess and simplify **
- [ ] ** `LinearProblem` needs allocator for zero params (for `GradientDescent` and LW)  **
- [ ] **give `progress(..)` the discrepancy and norms** (calculating them could be time-consuming)
- [ ] **make REGINN-CG work** (CG has a problem, might be due to wrong norm? have to use mass matrix!)
- [ ] REGINN tolerance choice (-> class)
- [ ] gradient method instead of conjugate gradients (maybe more robust?)
- [ ] maybe look at cg-directions visually to figure out where it goes wrong (boundary?)
- [ ] linear Tikhonov (using "Tikhonov-CG"?)
- [ ] Adjoints for a, nu and q (base problem class for linearizations? `LinearizedWaveProblem`)

## Documentation and Tests

- [x] solution for discretized parameters+rhs the same as when passing those as functions
- [x] test of adjointness
- [ ] test with reference solution
- [ ] better documentation