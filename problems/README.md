# Example problems

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
