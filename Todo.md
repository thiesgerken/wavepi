# WavePI Todo List

## General 

* organize code into sub-namespaces (`direct`, `inversion`, ...)

## Direct Solver

* grid adaptivity
* Interpolation of time steps (DiscretizedFunction::at)

## Inversion

* implement L^2-Landweber scheme (as  a class?) $c_{k+1} = c_k + w (S' c_k)* (g - S c_k)$
* add Shrinkage step
* base class: IterativeRegularization (virtual calculateStep = 0, start, test) and maybe even Regularization
* Adjoint -> use L2 (almost self-adjoint, integrate backward in time) 
* after that REGINN (+ LinearRegularization + some possibilities for that (CG, LW) ) 

## Tests

* solution for discretized parameters+rhs the same as when passing those as functions (have to obfuscate that they are DiscretizedFunctions)  