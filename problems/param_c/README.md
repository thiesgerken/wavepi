# problems for $`c`$

### [`h1l2_2.cfg`](2.cfg)

Tries to reconstruct a constant $`c = 1.5`$ from field with $`1\%`$ noise using $`L^2`$-Norms in time and space.
Initial guess is $`c = 1.0`$. In contrast to [`h1l2_1.cfg`](1.cfg), better right hand side and domain is used.

Modifications:

### [`1.cfg`](1.cfg)

Tries to reconstruct a constant $`c = 2`$ from field with $`1\%`$ noise using $`L^2`$-Norms in time and space.
Initial guess is $`c = 0.5`$.
Very difficult problem because of bad setting (no real wave that travels through domain?)
REGINN fails to reconstruct anything meaningful and is caught inside an endless loop (due to enforcement of lower bound)

### [`2.cfg`](2.cfg)

Tries to reconstruct a constant $`c = 1.5`$ from field with $`1\%`$ noise using $`L^2`$-Norms in time and space.
Initial guess is $`c = 1.0`$. In contrast to [`1.cfg`](1.cfg), better right hand side and domain is used.

* strong oscillations in discrepancy (due to enforcement of lower bound), decreases very slowly (if at all), caught at about $`20\%`$.
* after some time, discrepancy increases and estimate becomes very large (~100)
* estimate does not look good (oscillations, error does not decrease at all).

Modifications:

* [`4.cfg`](4.cfg): higher `max_tol` and Fibonacci maximum iteration choice
* [`5.cfg`](5.cfg): higher `max_tol`
* [`7.cfg`](7.cfg): higher `max_tol`, Fibonacci maximum iteration choice and safeguarding for cg
* [`8.cfg`](8.cfg): higher `max_tol`, Fibonacci maximum iteration choice, safeguarding for cg, two right hand sides
* [`9.cfg`](9.cfg): higher `max_tol`, Fibonacci maximum iteration choice, safeguarding for cg, four right hand sides
* [`10.cfg`](10.cfg): higher `max_tol`, Fibonacci maximum iteration choice, safeguarding for cg, four right hand sides, eps=1e-4

### [`h1l2_1.cfg`](1.cfg)

Tries to reconstruct a constant $`c = 2`$ from field with $`1\%`$ noise using $`H^1([0,T], L^2)`$-Norm in domain and $`L^2([0,T], L^2)`$ in codomain (identical measurements)
Initial guess is $`c = 0.5`$.
Very difficult problem because of bad setting (no real wave that travels through domain?)

### [`h1l2_2.cfg`](2.cfg)

Tries to reconstruct a constant $`c = 1.5`$ from field with $`1\%`$ noise using $`L^2`$-Norms in time and space.
Initial guess is $`c = 1.0`$. In contrast to [`h1l2_1.cfg`](1.cfg), better right hand side and domain is used.

Modifications:

* [`h1l2_4.cfg`](h1l2_4.cfg): higher `max_tol` and Fibonacci maximum iteration choice
* [`h1l2_5.cfg`](h1l2_5.cfg): higher `max_tol`
* [`h1l2_7.cfg`](h1l2_7.cfg): higher `max_tol`, Fibonacci maximum iteration choice and safeguarding for cg
* [`h1l2_8.cfg`](h1l2_8.cfg): higher `max_tol`, Fibonacci maximum iteration choice, safeguarding for cg, two right hand sides
* [`h1l2_9.cfg`](h1l2_9.cfg): higher `max_tol`, Fibonacci maximum iteration choice, safeguarding for cg, four right hand sides
* [`h1l2_10.cfg`](h1l2_10.cfg): higher `max_tol`, Fibonacci maximum iteration choice, safeguarding for cg, four right hand sides, eps=1e-4
