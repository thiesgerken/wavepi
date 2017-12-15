# Example problems

## [`problem_c_1.cfg`](problem_c_1.cfg)

Tries to reconstruct a constant $`c = 2`$ from field with $`1\%`$ noise using $`L^2`$-Norms in time and space.
Initial guess is $`c = 0.5`$.
Very difficult problem because of bad setting (no real wave that travels through domain?)
REGINN fails to reconstruct anything meaningful and is caught inside an endless loop (due to enforcement of lower bound)

## [`problem_c_2.cfg`](problem_c_2.cfg)

Tries to reconstruct a constant $`c = 1.5`$ from field with $`1\%`$ noise using $`L^2`$-Norms in time and space.
Initial guess is $`c = 1.0`$. In contrast to [`problem_c_1.cfg`](problem_c_1.cfg), better right hand side and domain is used.

* strong oscillations in discrepancy (due to enforcement of lower bound), decreases very slowly (if at all), caught at about $`20\%`$.
* after some time, discrepancy increases and estimate becomes very large (~100)
* estimate does not look good (oscillations, error does not decrease at all).

## [`problem_c_3.cfg`](problem_c_3.cfg)

Like [`problem_c_2.cfg`](problem_c_2.cfg), but with 5 instead of 6 initial global refines.

## [`problem_a_1.cfg`](problem_a_1.cfg)

Like [`problem_c_2.cfg`](problem_c_2.cfg), but for $`a`$.

## [`problem_q_1.cfg`](problem_q_1.cfg)

Like [`problem_c_2.cfg`](problem_c_2.cfg), but for $`q`$.

## [`problem_nu_1.cfg`](problem_nu_1.cfg)

Like [`problem_c_2.cfg`](problem_c_2.cfg), but for $`\nu`$.

## [`problem_c_4.cfg`](problem_c_4.cfg)

Like [`problem_c_2.cfg`](problem_c_2.cfg), but with higher `max_tol` and Fibonacci maximum iteration choice

## [`problem_c_5.cfg`](problem_c_5.cfg)

Like [`problem_c_2.cfg`](problem_c_2.cfg), but with higher `max_tol`.

## [`problem_c_6.cfg`](problem_c_6.cfg)

Like [`problem_c_4.cfg`](problem_c_4.cfg), but with 5 instead of 6 initial global refines.

## [`problem_c_7.cfg`](problem_c_7.cfg)

Like [`problem_c_4.cfg`](problem_c_4.cfg), but with safeguarding for cg.
