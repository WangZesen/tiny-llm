---
title: Algorithms and estimators
description: Parameter mixing, local AdamW moments, and curvature diagnostics in mathematical notation.
---

## Local gradients and clipping

Let $x_i^t$ be worker $i$'s parameters before update $t$, and $f_i^t$ its local minibatch mean loss. Each worker computes $g_i^t=\nabla f_i^t(x_i^t)$. With clipping threshold $c$, the gradient passed to AdamW is

$$
\widetilde g_i^t=g_i^t\min\left(1,\frac{c}{\lVert g_i^t\rVert_2+\epsilon_c}\right),
\qquad \epsilon_c=10^{-6}.
$$

The small denominator constant follows the gradient-clipping implementation. With clipping disabled, $\widetilde g_i^t=g_i^t$. Gradients are normalized by each worker's valid local targets, not by the number of workers. Nonfinite gradients stop training in either mode.

## AdamW and mixing order

The first and second moments remain local:

$$
\begin{aligned}
m_i^{t+1}&=\beta_1m_i^t+(1-\beta_1)\widetilde g_i^t,\\
v_i^{t+1}&=\beta_2v_i^t+(1-\beta_2)(\widetilde g_i^t)^2,\\
u_i^t&=\frac{m_i^{t+1}/(1-\beta_1^{t+1})}
{\sqrt{v_i^{t+1}/(1-\beta_2^{t+1})}+\epsilon}.
\end{aligned}
$$

Products, squares, and quotients involving vectors are elementwise. Let $W_t$ be the parameter mixing matrix and let $\lambda$ denote weight decay for the parameter group. Adapt-while-combine (AWC) applies

$$
x_i^{t+1}=(1-\eta_t\lambda)\sum_j(W_t)_{ij}x_j^t-\eta_tu_i^t.
$$

The gradients were evaluated before mixing; weight decay acts on the mixed parameters. Adapt-then-combine (ATC) instead applies

$$
x_i^{t+1}=\sum_j(W_t)_{ij}\left[(1-\eta_t\lambda)x_j^t-\eta_tu_j^t\right].
$$

ATC mixes updated parameters, but it still does not mix moment buffers. For synchronous training there is one model and no parameter mixing. Both methods follow the same token-based learning-rate schedule in the tuning studies.

## AccumAdamW

AccumAdamW follows [Appendix C of *From Promise to Practice*](https://openreview.net/pdf?id=lo3nlFHOft),
with the persistent second-moment coefficient corrected to $\beta_2$ and epsilon
outside the square root, matching the
[authors' implementation](https://github.com/WangZesen/Decent-DP/blob/main/src/decent_dp/optim.py).
The optimizer uses the same clipped gradients and parameter-mixing order as above.

Using the zero-based update index $t$ from above, let $s$ be `optimizer.accum_iter`
and $k=\lfloor t/s\rfloor+1$. Initialize persistent moments $M_i,V_i$ and buffer
$b_i$ to zero. Before committing the current window, form

$$
\begin{aligned}
m_i^t &= \beta_1 M_i+(1-\beta_1)\widetilde g_i^t,\\
v_i^t &= \beta_2 V_i+(1-\beta_2)(\widetilde g_i^t)^2,\\
u_i^t &= \frac{m_i^t/(1-\beta_1^k)}{\sqrt{v_i^t/(1-\beta_2^k)}+\epsilon}.
\end{aligned}
$$

Apply the ordinary, AWC, or ATC parameter update with this direction, including
decoupled weight decay every step. Add $\widetilde g_i^t/s$ to $b_i$. Only when
$(t+1)\bmod s=0$, commit

$$
M_i\leftarrow\beta_1 M_i+(1-\beta_1)b_i,\qquad
V_i\leftarrow\beta_2 V_i+(1-\beta_2)b_i^2,\qquad b_i\leftarrow0.
$$

The second moment uses the square of the mean gradient, so opposite gradients
within a window can cancel. Parameters still update on every step. With $s=1$,
these equations reduce to AdamW. All three buffers remain local to each worker.
Incomplete windows carry across epochs and checkpoints without a forced commit.

In saved state, `exp_avg` and `exp_avg_sq` hold $M_i,V_i$, and `accum_grad` holds
$b_i$. The reference implementation stores moments pre-multiplied by their
respective betas; our unscaled representation produces the same update equations
and directly exposes the moments defined here.

## Topology and evaluation

One-peer exponential mixing uses incoming offsets 1, 2, 4, and so on below the number of workers, cycling across updates. Each row assigns half its weight to the worker itself and half to its selected incoming peer. Other supported topologies include complete averaging and an alternating one-peer ring. Details and checkpoint behavior are in the [implementation reference](implementation.md).

Packed validation evaluates the ordinary model with weights

$$
\bar x^t=\frac1N\sum_{i=1}^N x_i^t.
$$

For $N$ workers, local microbatch size $B$, and context length $C$, the full global batch is $NBC$ prediction targets. The global budget is based on one local model's parameter count. Local models receive disjoint worker partitions of that stream.

## Gradient noise and curvature

At a fixed checkpoint, let $\bar g$ be the full-epoch token-mean gradient and $H$ the Hessian of that same mean loss. A sampled effective minibatch gives gradient $g_k$, with noise $v_k=g_k-\bar g$. The reported unnormalized and normalized alignment estimators are

$$
\widehat A=\frac1S\sum_{k=1}^S v_k^\top H v_k,
\qquad
\widehat A_{\mathrm{norm}}=
\frac{\sum_{k=1}^S v_k^\top H v_k}
{\sum_{k=1}^S\lVert v_k\rVert_2^2}.
$$

The normalized estimator is a ratio of sums, not the average of individually normalized samples. Zero total noise yields an undefined normalized value, stored as null. Negative curvature is retained. Independent Rademacher directions $r_k$ provide the isotropic reference $\frac{1}{DR}\sum_k r_k^\top H r_k$, where $D$ counts unique trainable parameters.

Packed consensus errors are $e_i=x_i-\bar x$. Their alignment is evaluated using the Hessian at $\bar x$. The [analysis reference](analysis.md) describes sampling, seen/unseen data, precision, and provenance. The [adaptive-consensus investigation](adaptive-consensus.md) uses these diagnostics in a separate historical study; its recipes are not pooled into the current tuning grids.
