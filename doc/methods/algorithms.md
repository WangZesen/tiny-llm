---
archive: true
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

## Topology and evaluation

One-peer mixing uses reciprocal pairs: each worker sends to and receives from the
same peer. Each row assigns half its weight to itself and half to its peer, so
$W_t$ is symmetric and doubly stochastic. Exponential mixing pairs worker $i$ with
`i XOR (1 << (t % log2(N)))` and requires power-of-two $N$. With no local updates,
the product of $\log_2 N$ consecutive full mixing matrices is the global averaging
matrix. Ring mixing requires even $N$ and alternates adjacent pairs
`(0,1), (2,3), ...` at even $t$ and `(1,2), (3,4), ..., (N-1,0)` at odd $t$.
All topologies allow $N=1$ as a no-op; complete averaging supports any positive
worker count. Details and checkpoint behavior are in the
[implementation reference](implementation.md).

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
