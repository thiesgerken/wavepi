# Example problems


## Reconstruction of constant $`c`$

### [`problem_c_1.cfg`](problem_c_1.cfg)

Tries to reconstruct a constant $`c = 2`$ from field with $`1\%`$ noise using $`L^2`$-Norms in time and space.
Initial guess is $`c = 0.5`$.
Very difficult problem because of bad setting (no real wave that travels through domain?)
REGINN fails to reconstruct anything meaningful and is caught inside an endless loop (due to enforcement of lower bound)

### [`problem_c_2.cfg`](problem_c_2.cfg)

Tries to reconstruct a constant $`c = 1.5`$ from field with $`1\%`$ noise using $`L^2`$-Norms in time and space.
Initial guess is $`c = 1.0`$. In contrast to [`problem_c_1.cfg`](problem_c_1.cfg), better right hand side and domain is used.

* strong oscillations in discrepancy (due to enforcement of lower bound), decreases very slowly (if at all), caught at about $`20\%`$.
* after some time, discrepancy increases and estimate becomes very large (~100)
* estimate does not look good (oscillations, error does not decrease at all).

Modifications:

* [`problem_c_4.cfg`](problem_c_4.cfg): higher `max_tol` and Fibonacci maximum iteration choice
* [`problem_c_5.cfg`](problem_c_5.cfg): higher `max_tol`
* [`problem_c_7.cfg`](problem_c_7.cfg): higher `max_tol`, Fibonacci maximum iteration choice and safeguarding for cg
* [`problem_c_8.cfg`](problem_c_8.cfg): higher `max_tol`, Fibonacci maximum iteration choice, safeguarding for cg, two right hand sides
* [`problem_c_9.cfg`](problem_c_9.cfg): higher `max_tol`, Fibonacci maximum iteration choice, safeguarding for cg, four right hand sides, coarse grid
*
### [`problem_c_h1l2_1.cfg`](problem_c_1.cfg)

Tries to reconstruct a constant $`c = 2`$ from field with $`1\%`$ noise using $`H^1([0,T], L^2)`$-Norm in domain and $`L^2([0,T], L^2)`$ in codomain (identical measurements)
Initial guess is $`c = 0.5`$.
Very difficult problem because of bad setting (no real wave that travels through domain?)

### [`problem_c_h1l2_2.cfg`](problem_c_2.cfg)

Tries to reconstruct a constant $`c = 1.5`$ from field with $`1\%`$ noise using $`L^2`$-Norms in time and space.
Initial guess is $`c = 1.0`$. In contrast to [`problem_c_h1l2_1.cfg`](problem_c_1.cfg), better right hand side and domain is used.

Modifications:

* [`problem_c_h1l2_4.cfg`](problem_c_h1l2_4.cfg): higher `max_tol` and Fibonacci maximum iteration choice
* [`problem_c_h1l2_5.cfg`](problem_c_h1l2_5.cfg): higher `max_tol`
* [`problem_c_h1l2_7.cfg`](problem_c_h1l2_7.cfg): higher `max_tol`, Fibonacci maximum iteration choice and safeguarding for cg
* [`problem_c_h1l2_8.cfg`](problem_c_h1l2_8.cfg): higher `max_tol`, Fibonacci maximum iteration choice, safeguarding for cg, two right hand sides
* [`problem_c_h1l2_9.cfg`](problem_c_h1l2_9.cfg): higher `max_tol`, Fibonacci maximum iteration choice, safeguarding for cg, four right hand sides, coarse grid


## Reconstruction of constant $`q`$

[`problem_q_1.cfg`](problem_q_1.cfg)

Like [`problem_c_2.cfg`](problem_c_2.cfg), but for $`q`$. Converges without problems, even for noise level $`10^{-4}`$ ([`problem_q_2.cfg`](problem_q_2.cfg)).

Modifications:

* [`problem_q_3.cfg`](problem_q_3.cfg): noise level $`10^{-4}`$ and two right hand sides

## Reconstruction of constant $`\nu`$

[`problem_nu_1.cfg`](problem_nu_1.cfg)

Like [`problem_c_2.cfg`](problem_c_2.cfg), but for $`\nu`$. First linear system cannot be solved, adjoint does not seem to be correct.

## Reconstruction of constant $`a`$

### [`problem_a_1.cfg`](problem_a_1.cfg)

Like [`problem_c_2.cfg`](problem_c_2.cfg), but for $`a`$. Same bad behaviour as for $`c`$.

###  [`problem_a_h1l2_1.cfg`](problem_a_h1l2_1.cfg)

Like [`problem_c_h1l2_2.cfg`](problem_c_h1l2_2.cfg), but for $`a`$.
