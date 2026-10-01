// Standalone report. All numerical results are loaded from bundled extracts.
#let d = json("data/analysis.json")
#let navy = rgb("172D49")
#let muted = rgb("526173")
#let light = rgb("EDF2F6")
#let blue = rgb("0072B2")
#let purple = rgb("8B61B3")
#let orange = rgb("D55E00")
#set document(title: "Small-Batch Training with AdamW", date: datetime(year: 2026, month: 10, day: 1))
#set page(paper: "a4", margin: (x: 20mm, top: 19mm, bottom: 18mm), header: align(right, text(font: "Nimbus Sans", size: 8pt, fill: muted)[SMALL-BATCH TRAINING WITH ADAMW · RESEARCH REPORT]), footer: context align(center, text(font: "Nimbus Sans", size: 9pt, fill: muted)[#counter(page).display("1")]))
#set text(font: "Libertinus Serif", size: 10.8pt, lang: "en")
#set par(justify: true, leading: 0.5em)
#set heading(numbering: "1.1")
#show heading: set text(font: "Nimbus Sans", fill: navy)
#show heading.where(level: 1): set block(above: 1em, below: 0.65em)
#show heading.where(level: 2): set block(above: 0.9em, below: 0.45em)
#show math.equation: set text(font: "New Computer Modern Math")
#set math.equation(numbering: "(1)")
#set figure(gap: 7pt)
#show figure.caption: set text(size: 9.2pt)
#let num(v, digits: 6) = if v == none { "—" } else { str(calc.round(float(v), digits: digits)) }
#let signed(v, digits: 6) = if v == none { "—" } else { (if v >= 0 { "+" } else { "" }) + num(v, digits: digits) }
#let pct(v, digits: 2) = num(100 * v, digits: digits) + "%"
#let mean(xs) = xs.sum() / xs.len()
#let median(xs) = { let ys = xs.sorted(); let n = ys.len(); if calc.rem(n, 2) == 1 { ys.at(int(calc.floor(n / 2))) } else { (ys.at(int(n / 2 - 1)) + ys.at(int(n / 2))) / 2 } }
#let path(v) = text(font: "Nimbus Mono PS", size: 7.9pt, v.replace("/", "/​").replace("-", "-​"))
#let note(body) = block(fill: light, inset: 10pt, radius: 3pt, width: 100%)[#text(size: 9.8pt, body)]
#let th(body) = table.cell(fill: light)[#text(font: "Nimbus Sans", weight: "bold", fill: navy, body)]
#let tab(columns, ..cells) = {
  set text(size: 9.1pt)
  table(columns: columns, stroke: (left: none, right: none, top: none, bottom: 0.4pt + rgb("D8E0E8")), inset: (x: 5pt, y: 5pt), align: (left, horizon), ..cells)
}
#let plot(name, caption, height: 98mm) = figure(image("assets/" + name + ".svg", width: 100%, height: height, fit: "contain"), caption: caption)
#let winner(method, batch) = d.winners.at(method).find(r => r.batch == batch)
#let label(method) = (sync: "Synchronous", clipped: "Decentralized, clipped", unclipped: "Decentralized, no clipping").at(method)
#let flags(r) = {
  let xs = ()
  if r.at("lr_boundary", default: false) { xs.push("LR") }
  if r.at("beta1_boundary", default: false) { xs.push("β₁") }
  if r.at("beta2_boundary", default: false) { xs.push("β₂") }
  if xs.len() == 0 { "—" } else { xs.join(", ") }
}
#let batches = (16, 32, 64, 128)
#let gaps = batches.map(b => winner("clipped", b).mean_loss - winner("sync", b).mean_loss)
#let matched-delta(r) = r.at("loss_difference", default: r.at("mean_loss_difference", default: r.at("delta_loss", default: r.at("delta", default: none))))

#text(font: "Nimbus Sans", fill: muted, size: 10pt, tracking: 1pt)[RESEARCH RECORD · 1 OCTOBER 2026]
#v(8pt)
#text(font: "Nimbus Sans", fill: navy, weight: "bold", size: 29pt)[Small-Batch Training\ with AdamW]
#v(8pt)
#text(font: "Nimbus Sans", fill: navy, size: 16pt)[Synchronous and Four-Worker\ Decentralized Training]
#v(15pt)
#heading(level: 1, numbering: none)[Abstract]
This investigation studies the interaction between batch size, Adam moment memory, and gradient clipping at a fixed training-target budget. A 20.4-million-parameter language model is trained on C4 with global batches of 16, 32, 64, and 128 sequences. We compare synchronous AdamW with four logical decentralized workers sharing one GH200, using independent local moments and reciprocal one-peer exponential parameter mixing. The evidence combines completed synchronous tuning, a unified clipped decentralized grid, a fixed-configuration clipping ablation, and a separate no-clipping grid. Results are frozen through 1 October 2026. [S1–S5]

Smaller batches can improve final full-validation loss, but the best observed batch is 32 or 64 rather than 16. Higher first-moment retention is favored as batches shrink; second-moment tuning supports a broad near-one region rather than a strictly monotonic rule. Tuned clipped decentralized training is numerically close to synchronous training, with matching-batch loss gaps of #num(calc.min(..gaps))–#num(calc.max(..gaps)). Clipping is frequent at the smallest decentralized local batch, yet matched-grid comparisons show a larger benefit from clipping at larger global batches. Removing clipping changes the preferred learning rate. These findings are conditional on the tested model, data, token budget, and tuning protocols, and do not establish statistical equivalence or a universal scaling law.

#note[*Reading the evidence.* Lower validation loss is better. A “recipe” comprises the batch, learning rate, Adam coefficients, and other fixed optimizer settings. Unless explicitly stated otherwise, reported means use seeds 42–44 and error bars are *sample standard deviations*. These seeds contribute to tuning; they are not a fresh confirmation set.]

#heading(level: 2, numbering: none)[Companion material]
The companion presentation contains exactly 13 slides. This report provides the methods, qualified findings, individual-seed result tables, and technical appendix. Figures and numerical tables are regenerated from bundled extracts; reproducing the documents does not launch training or require access to the original campaign directories.

#pagebreak()
= Experimental design

== Controlled training and evaluation budgets
Every principal comparison uses the same decoder-only model with *20,403,520 unique parameters*, context length 512, the frozen C4 token cache, and *408,092,672 global training targets*. Each run completes 40 virtual epochs, with *20,447,232 warmup targets* and cosine decay to zero. A virtual epoch is a reporting interval defined by the target budget, not a complete pass through C4. The final metric is mean cross-entropy in *nats per target* over all *197,411,295 validation targets*. Perplexity is the exponential of that loss. [S1, S2, S5]

#tab((1.2fr, 2.4fr),
  th[Setting], th[Executed value],
  [Model], [8 transformer blocks; width 320; 5 attention heads; head dimension 64; feed-forward width 896; vocabulary 32,000; tied input/output embeddings],
  [Precision / kernels], [FP32 parameters and Adam moments; BF16 AMP; compiled SDPA; nondeterministic GPU execution],
  [AdamW], [Weight decay 0.1; epsilon $10^(-8)$; tuned LR, $beta_1$, and $beta_2$; decay omitted for one-dimensional parameter tensors],
  [Clipping], [Threshold 1 for synchronous and clipped decentralized training; disabled with `grad_clip: null` for the no-clipping study],
  [Final validation], [Batch 128; complete cached validation split; no test-set claims],
  [Persistence], [No checkpoints or weight files saved by these campaigns; configuration, metrics, result, environment, and scheduler evidence retained],
)

== Global and local batch size
The synchronous method updates one model with the full global batch. The decentralized method holds four local models and four independent pairs of Adam moment states. Each worker processes one quarter of the global batch. Thus the global batches 16, 32, 64, and 128 correspond to local batches 4, 8, 16, and 32. Each decentralized worker consumes *102,023,168 targets* over the run. Four consecutive positions in the interleaved buffered sample stream are assigned across the four workers. [S2, S5]

All four logical workers are packed onto *one GH200*. This controls a local implementation of decentralized optimization; it is not a four-GPU experiment and does not measure network communication, cross-device synchronization, or distributed scaling. Independent local moments distinguish this method from synchronous gradient averaging even when parameters are mixed frequently.

#note[*Fairness of the comparison.* Model size, context, training targets, warmup targets, validation coverage, and precision are held fixed. Search spaces and selection procedures differ across studies. The final quality comparison is therefore exploratory, not a controlled comparison of equally tuned algorithms.]

#pagebreak()
= Decentralized optimizer and measurement definitions

== AWC ordering and reciprocal mixing
At update $t$, worker $i$ computes the gradient of its local mean loss at its current parameters $theta_(i,t)$. Local losses are summed for backpropagation, so no additional factor of one quarter scales a worker's gradient. When enabled, clipping applies to that worker's full parameter-gradient norm before parameter mixing. The AWC execution order is *local forward/backward → local clipping → parameter mixing → independent AdamW update*. [S2, S5]

For four workers indexed from zero, the peer is obtained by flipping bit $t mod 2$ of the worker index. Pairings alternate between $(0,1),(2,3)$ and $(0,2),(1,3)$. Every pairing is reciprocal. If $p_t(i)$ denotes this peer,
$ y_(i,t) = (theta_(i,t) + theta_(p_t(i),t))/2. $
There is no adaptive consensus schedule. The local gradient was evaluated *before* replacing parameters by $y_(i,t)$. Write $tilde(g)_(i,t)$ for the clipped local gradient, or the unchanged local gradient when clipping is disabled. It updates only that worker's moments:
$ m_(i,t+1) = beta_1 m_(i,t) + (1-beta_1) tilde(g)_(i,t), $
$ v_(i,t+1) = beta_2 v_(i,t) + (1-beta_2) tilde(g)_(i,t)^2. $
Writing $hat(m)_(i,t+1)=m_(i,t+1)/(1-beta_1^(t+1))$ and $hat(v)_(i,t+1)=v_(i,t+1)/(1-beta_2^(t+1))$ for the bias-corrected moments, the AdamW step is
$ theta_(i,t+1) = (1-eta_t lambda) ⊙ y_(i,t) - eta_t hat(m)_(i,t+1)/(sqrt(hat(v)_(i,t+1)) + epsilon). $
Products, squares, and divisions are coordinatewise. The decay coefficient $lambda$ is 0.1 for matrix-like parameters and zero for one-dimensional tensors. Local moments are not averaged or exchanged.

== Clipping and evaluation
Enabled clipping rescales a local gradient using $min(1, 1/(norm(g)_2 + 10^(-6)))$. Logged norms are pre-clipping norms. A clipping counter records whether the norm exceeds the configured threshold. A decentralized clipping frequency pools worker-update counts; a synchronous frequency counts single-model updates. The counters are not a measure of clipping magnitude or of its causal benefit. [S3]

Final decentralized evaluation first forms the arithmetic parameter mean in FP32,
$ overline(theta) = (1/4) sum_(i=0)^3 theta_i, $
then evaluates that single model on the complete validation cache. It does not average worker losses or logits. Intermediate subset-validation measurements are not substituted for final full-validation loss. Exact token budgets, four-worker metadata, finite measurements, and final-validation coverage are checked before an observation enters the analysis. [S2, S5]

#pagebreak()
= Study structure and selection

== Synchronous tuning
The synchronous study first tunes LR and $beta_2$ at $beta_1=0.9$, separately for weight decays zero and 0.1. Seed 42 screens configurations, selected configurations receive seeds 43–44, and the procedure permits one LR endpoint extension. Earlier fresh seeds 45–47 compare a selected smaller batch with batch 128 at these original $beta_1=0.9$ recipes. A later conditional sweep varies $beta_1$ while fixing LR and $beta_2$ to their selected values; one seed-42 minimum per batch and decay is frozen before its replication. The principal synchronous reference here is this *final conditional tuning at weight decay 0.1*. Earlier fresh-seed results concern different recipes and appear separately in the appendix. [S1]

== Unified clipped decentralized tuning
The combined clipped grid contains *840 distinct seed-42 configurations*, 210 per batch. It spans six LRs, five first-moment coefficients, and seven second-moment choices, including a fixed global-target half-life. The original campaign and its near-one $beta_2$ extension each froze five configurations per batch before replication. Consequently, there are *40 configurations with seeds 42–44*, ten per batch, and *920 successful observations*. The ten replicated configurations at each batch need not be the pooled grid's top ten. This report ranks the available complete three-seed configurations together and preserves both original selections. [S2]

== Fixed-configuration ablation and no-clipping tuning
The original clipping ablation takes the top two clipped configurations at batches 16 and 32, then runs each without clipping on seeds 42–44. It contains twelve no-clipping observations and holds the other recipe settings fixed. It is followed by a separate no-clipping grid containing *480 seed-42 configurations*, 120 per batch. That grid removes $beta_1=0.9$, $beta_2=0.98$, and each batch's highest LR from the combined clipped grid. Five successful configurations per batch are frozen using seed-42 loss only, breaking exact ties by lower LR, then $beta_1$, then $beta_2$. They receive seeds 43–44, yielding *520 primary observations*. [S4, S5]

Four authenticated ablation observations at seed 42 enter the no-clipping screening grid. Existing seeds 43–44 enter its primary results only when the corresponding configuration is promoted; other old replications remain supplemental. Imports retain their original scheduler and artifact evidence. They are deduplicated by complete configuration identity and seed, including clipping configuration, rather than by a human-readable recipe key that may omit clipping.

#note[*Three different comparisons.* Separately tuned winners answer which tested recipe performed best. Fixed-configuration interventions compare clipping choices at the same settings. Matched-grid screening contrasts all common configurations at seed 42. These comparisons answer related but different questions.]

#pagebreak()
= Finding 1: smaller batches can improve quality

#plot("quality", [Three-seed final validation loss for the selected synchronous and clipped decentralized recipes. Error bars are sample SD; configurations were selected through different tuning protocols. [S1, S2]], height: 105mm)

The best observed synchronous mean is #num(winner("sync", 64).mean_loss) at global batch 64. The best clipped decentralized mean is #num(winner("clipped", 32).mean_loss) at global batch 32. Relative to their respective batch-128 selections, these improve loss by #num(winner("sync", 128).mean_loss - winner("sync", 64).mean_loss) and #num(winner("clipped", 128).mean_loss - winner("clipped", 32).mean_loss). The same fixed global-target budget therefore admits better tested recipes at smaller batches. [S1, S2]

The trend is not monotonic. At batch 16, the selected means are #num(winner("sync", 16).mean_loss) for synchronous training and #num(winner("clipped", 16).mean_loss) for clipped decentralized training. Neither is the best observed value for its method. “Smaller batches help” should therefore be read as a possibility supported by tuning, not as a directive to choose the smallest batch or an assertion that batch size alone caused the gain.

Because reducing batch size increases the number of parameter updates at a fixed target budget, the optimizer experiences both a different gradient estimator and a different number of moment updates. Adjusting LR and moment coefficients is part of the experiment. Holding those settings fixed would answer a different question from the final tuned comparison.

#note[*Practical implication.* Choose batch size jointly with optimizer settings and the compute objective. The observed quality optimum at a fixed token budget need not minimize wall-clock time.]

#pagebreak()
= Finding 2: the decentralized quality gap is small

#plot("gap", [Clipped decentralized minus synchronous final validation loss at matching global batches. Perplexity ratios are computed from the mean-loss difference. This is an exploratory tuning comparison, not an equivalence test. [S1, S2]], height: 98mm)

#let gap-cells = ()
#for b in batches {
  let a = winner("sync", b)
  let c = winner("clipped", b)
  let delta = c.mean_loss - a.mean_loss
  gap-cells += ([#b], [#num(a.mean_loss)], [#num(c.mean_loss)], [#signed(delta)], [#num(100 * (calc.exp(delta) - 1), digits: 3)%])
}
#tab((0.6fr, 1fr, 1fr, 1fr, 1fr), th[Global batch], th[Sync. mean], th[Decentr. mean], th[Loss gap], th[Perplexity gap], ..gap-cells)

All four clipped decentralized winners are numerically worse than their synchronous references, but the differences are small on this loss scale. The range #num(calc.min(..gaps))–#num(calc.max(..gaps)) corresponds to approximately #num(100 * (calc.exp(calc.min(..gaps)) - 1), digits: 2)%–#num(100 * (calc.exp(calc.max(..gaps)) - 1), digits: 2)% higher perplexity. The calculation uses $100(exp(Delta L)-1)$; it is not a percentage difference in cross-entropy.

The sample SDs characterize variation across three seeds at selected recipes. They do not correct for selecting among many candidates, and the repeated seed labels do not guarantee matched random trajectories between methods. The observed closeness is compatible with a useful decentralized optimizer, but does not establish equal population performance, a non-inferiority margin, or a four-device speed advantage.

#pagebreak()
= Finding 3: smaller batches favor longer moment memory

== First moment: a coherent token-horizon interpretation
#plot("beta1", [Selected first-moment coefficients and their global-target half-lives. The synchronous batch-16, -32, and -64 choices preserve approximately 0.56 million targets of exponential memory. This is an interpretation of the observed choices, not an independently verified scaling rule. [S1, S2, S5]], height: 105mm)

For an exponential coefficient $beta$, the half-life in updates is $-ln(2)/ln(beta)$. Multiplying by the global targets consumed per update places memory on a common data scale. A fixed $beta$ shortens this target horizon when the batch shrinks. Raising $beta_1$ can compensate for the increased update frequency.

The final synchronous choices progress from $beta_1=0.9$ at batch 128 to 0.96, 0.98, and 0.99 at batches 64, 32, and 16. Their first-moment global-target half-lives at the three smaller batches are #num(winner("sync", 64).beta1_global_half_life_targets / 1e6, digits: 3), #num(winner("sync", 32).beta1_global_half_life_targets / 1e6, digits: 3), and #num(winner("sync", 16).beta1_global_half_life_targets / 1e6, digits: 3) million targets. The decentralized selections similarly use 0.99 at batch 16 and 0.98 at batch 32, while larger batches select 0.95. [S1, S2, S5]

This consistency supports tuning moment memory in target units. It does not prove that a constant half-life is optimal: synchronous tuning was conditional on LR and $beta_2$, the candidate sets differ, and some choices lie at grid boundaries. For a decentralized worker the local-target half-life is one quarter of the global-target value; global and worker units should never be interchanged.

#pagebreak()
== Second moment: a near-one plateau, not a monotonic law

#plot("beta2", [Seed-42 screening profiles minimize over LR and first-moment coefficient at each second-moment choice. Replicated comparisons keep LR and first moment fixed. A profile minimum is a selected screening measurement, not a replicated effect estimate. [S2, S5]], height: 107mm)

The small-batch decentralized searches benefit from exploring $beta_2$ much closer to one than lower tested values such as 0.98 or 0.99. The selected clipped choices are #num(winner("clipped", 16).beta2) at batch 16 and #num(winner("clipped", 32).beta2) at batch 32, compared with #num(winner("clipped", 128).beta2) at batch 128. However, the batch-16 optimum is below the batch-32 optimum, so a strictly monotonic “smaller batch means larger $beta_2$” rule is unsupported. [S2]

#let near = d.replicated.clipped.filter(r => r.batch == 32 and r.lr == 0.004 and r.beta1 == 0.98 and r.beta2 >= 0.9999).sorted(key: r => r.beta2)
#let near-cells = ()
#for r in near { near-cells += ([#num(r.beta2)], [#num(r.seed42)], [#num(r.mean_loss)], [#num(r.std_loss)], [#num(r.replication_mean_loss)]) }
#tab((0.9fr, 1fr, 1fr, 1fr, 1fr), th[$beta_2$], th[Seed 42], th[Mean 42–44], th[Sample SD], th[Mean 43–44], ..near-cells)

At batch 32, the LR-0.004 / $beta_1=0.98$ configurations above have very similar three-seed means. Their ordering changes with the seed subset. The resulting plateau is more informative than declaring one extremely long half-life uniquely optimal. A coefficient near one also implies that its nominal half-life may exceed the entire training budget; bias correction and transient moment dynamics therefore matter. The half-life is a property of exponential weighting, not evidence that a run reached a stationary second-moment regime.

#pagebreak()
= Finding 4: clipping is frequent at very small local batches

#plot("clipping", [Clipping frequency for selected recipes, averaged over seeds 42–44. Decentralized rates pool four worker-update counters. Historical synchronous batch-128 counters are unavailable and are shown as missing. [S2, S3]], height: 108mm)

The selected clipped decentralized recipe at global batch 16 uses local batches of only four sequences. It clips roughly 84% of worker updates. At global batch 32, the winning recipe clips roughly 2.5%, while rates at batches 64 and 128 are below 1%. These measurements describe a striking change in the local optimization dynamics. [S2, S3]

The available final synchronous recipes clip far less often: below 0.64% on average across seeds at batches 16, 32, and 64. Most synchronous clipping occurs early. The earlier audit also reports epochs 3–40 separately, but that range is not exactly the post-warmup interval. Historical synchronous batch-128 logs lack the needed epoch counters, so no rate is imputed. [S3]

Clipping frequency depends on the entire selected recipe, including LR and moment memory, and on the local gradient estimator. It does not measure how far above one a norm lies, how much the direction changes after Adam preconditioning, or how useful the intervention is for final loss. In particular, a high frequency need not imply that clipping should be removed. The fixed-configuration experiment and the broader common-grid comparison test the latter question directly.

#note[*Disabled clipping counters.* Zero counters in no-clipping runs are a consequence of `grad_clip: null`. They do not show that gradient norms remained below one.]

#pagebreak()
= Clipping intervention at fixed configurations

#plot("ablation", [No clipping minus threshold-one validation loss for the original four selected configurations. Individual seeds and their three-seed averages are shown. Error bars are sample SDs of the three same-seed differences, not confidence intervals. Positive differences favor clipping. Configurations were selected from the clipped study before this intervention. [S4]], height: 100mm)

#let abl-cells = ()
#for r in d.ablation {
  abl-cells += ([#r.batch], [#num(r.lr, digits: 5)], [#num(r.beta1)], [#num(r.beta2)], [#num(r.baseline_mean_loss)], [#num(r.mean_loss)], [#signed(r.mean_loss_difference)])
}
#tab((0.5fr, 0.7fr, 0.55fr, 0.9fr, 1fr, 1fr, 1fr), th[Batch], th[LR], th[$beta_1$], th[$beta_2$], th[Clipped], th[No clip], th[Mean Δ], ..abl-cells)

Removing clipping improves the mean for both batch-16 configurations, with a larger improvement for $beta_2=0.9999$. At batch 32, both fixed configurations worsen without clipping. This is consistent with the hypothesis that the consequences of clipping depend on the batch and optimizer settings. The changes are small relative to some of the seed-to-seed differences, so neither the sign nor the magnitude should be overstated. [S4]

This intervention inherits the original clipped study's selection: these were its top two three-seed recipes at each of two batches. It does not search for the best no-clipping LR or moment coefficients, and it contains no batch-64 or batch-128 runs. The later no-clipping grid addresses this restriction. Reusing the ablation runs avoids duplicate experiments, but does not convert their old replication seeds into an independent confirmation set.

#pagebreak()
= Finding 5: clipping helps more at larger batches in the common grid

#plot("matched-grid", [Distributions of no-clipping minus clipped seed-42 loss over 120 matched configurations per batch. LR, moment coefficients, batch, and seed are fixed. All points are retained. Boxes show quartiles with 1.5-IQR whiskers; the symmetric-log loss axis is linear within ±0.025. [S2, S5]], height: 106mm)

#let match-cells = ()
#for b in batches {
  let rows = d.matched_screening.filter(r => r.batch == b)
  let diffs = rows.map(matched-delta).filter(v => v != none)
  match-cells += ([#b], [#diffs.len()], [#signed(median(diffs))], [#signed(mean(diffs))], [#diffs.filter(v => v < 0).len() / #diffs.len()])
}
#tab((0.6fr, 0.8fr, 1fr, 1fr, 1.2fr), th[Batch], th[Pairs], th[Median Δ], th[Mean Δ], th[No-clip wins], ..match-cells)

The grid contrast is broader than the original four-configuration intervention. At batch 16, no clipping wins on most common configurations and the average difference is slightly negative. At batches 32, 64, and 128, most common configurations favor clipping, and larger positive differences emerge. The mean is sensitive to badly performing unclipped runs; medians, complete distributions, and exact values are therefore reported together.

The apparent reversal between clipping frequency and usefulness is a substantive empirical result. The batch with the most frequent clipping does not show the largest benefit from retaining it. One plausible explanation is an interaction between LR, moment adaptation, and the size of occasional gradients, but the saved runs do not identify that mechanism. A single-seed grid with unequal distances from each method's optimum cannot establish that clipping inherently becomes more useful as batch size increases.

#pagebreak()
= Retuning without clipping changes the preferred learning rate

#plot("retuned", [Separately tuned three-seed winners for synchronous, clipped decentralized, and no-clipping decentralized training. Error bars are sample SD. The accompanying LR comparison distinguishes retuning from a fixed-configuration intervention. [S1, S2, S5]], height: 104mm)

#let tune-cells = ()
#for b in batches {
  let c = winner("clipped", b)
  let u = winner("unclipped", b)
  tune-cells += ([#b], [#num(c.lr, digits: 4)], [#num(u.lr, digits: 4)], [#num(u.mean_loss)], [#num(u.std_loss)], [#signed(u.mean_loss - c.mean_loss)])
}
#tab((0.55fr, 0.8fr, 0.8fr, 1fr, 0.8fr, 1fr), th[Batch], th[Clipped LR], th[No-clip LR], th[No-clip mean], th[Sample SD], th[Δ vs. clipped], ..tune-cells)

The no-clipping selections use lower LRs at batches 32–128. After retuning, no clipping is numerically best among the decentralized methods at batch 16, but the clipped choices retain lower mean loss at the other three batches. The best no-clipping recipe overall is at batch 32: LR #num(winner("unclipped", 32).lr, digits: 4), $beta_1=$#num(winner("unclipped", 32).beta1), $beta_2=$#num(winner("unclipped", 32).beta2), with mean loss #num(winner("unclipped", 32).mean_loss). [S5]

At batch 32, the lower LR changes the clipping comparison: the fixed LR-0.004 configurations favored clipping, while a smaller LR is preferred in the no-clipping search. The three-seed overlap includes LR-specific interventions and is tabulated in the appendix. Thus the conclusion is not “always clip at batch 32”; it is that clipping, LR, and moment memory should be tuned together. The independently selected winner difference must not be treated as the isolated causal effect of clipping.

#pagebreak()
= Quality, throughput, and memory

#plot("throughput", [Descriptive throughput and session duration versus global batch. Recipe pools and execution strata are retained in the bundled tables. Four logical workers share one device; no distributed-scaling claim is implied. [S1, S2, S5]], height: 111mm)

Smaller batches require more optimizer updates to consume the same number of targets. They can improve validation quality while reducing steady target throughput. The quality objective in this study is fixed-token performance; choosing a batch for minimum training time would require a different comparison, such as time to a predetermined validation target.

Steady throughput uses summed target increments divided by summed measured training-window seconds for complete logging windows that begin at or after the warmup target boundary. Windows crossing that boundary are excluded. This ratio is not an arithmetic mean of instantaneous rates. Session throughput additionally includes measured validation work and differs from steady throughput; it does not include scheduler queue time or all process-startup work. Hardware allocation and software/environment strata may differ across campaign observations, so throughput comparisons are descriptive rather than controlled benchmarks. [S1, S5]

The reported peak allocated memory includes *final evaluation at batch 128*. It is not a clean measure of training activations at the configured local batch. Fixed model and optimizer storage, evaluation allocations, compiled execution, and the measured lifetime all contribute to the peak. A slightly larger observed peak at smaller training batches therefore does not establish that smaller batches intrinsically require more activation memory. The saved counters do not isolate which allocation caused an individual peak.

#note[*Practical implication.* The observed quality gains at small batches should be weighed against update overhead and target throughput. These runs cannot establish communication efficiency or four-GPU scaling.]

#pagebreak()
= Replication, near ties, and remaining boundaries

#plot("replication", [Seed-42 screening and replication summaries illustrate the sensitivity of close rankings to the seed subset. The same frozen configurations are retained when averaging seeds 43–44. Error bars, where present, show sample SD. [S1, S2, S5]], height: 106mm)

Screening chooses configurations from one seed, so unusually favorable seed-42 measurements are likely to occur among selected candidates. Replication is useful because it reveals whether the seed-42 ordering persists. At batch 32, several clipped configurations with near-one second-moment coefficients remain close after replication; their relative ordering changes between seed 42, seeds 43–44, and all three seeds.

The “mean 43–44” column keeps each displayed recipe fixed and changes only the subset of losses averaged. It does not reselect recipes. For decentralized final rankings, these seeds contributed to selecting the best replicated recipe. For the synchronous conditional comparison, they were also involved in the earlier LR/$beta_2$ tuning. Neither use provides a fresh-seed confirmation of the final method comparison.

Boundary flags are recomputed against the appropriate executed grid. The selected clipped batch-32 configuration reaches the upper $beta_2$ endpoint. In the no-clipping grid, the batch-32 winner also reaches that endpoint, while the batch-64 and batch-128 winners reach the lower $beta_1$ endpoint. Such flags identify an unresolved tuning direction; they are not evidence that extending a grid would improve performance, and no automatic extension or replacement of frozen selections is performed. [S2, S5]

#note[*Precision of the conclusions.* With three seeds and selection across many candidates, small differences in final means should be interpreted alongside individual losses, sample SD, and the protocol. The report does not claim significance, equivalence, or a definitive ordering of near ties.]

#pagebreak()
= Limitations and conclusions

== Limits of the evidence
The study tests one model scale, one context length, one data cache, and a single fixed target budget. There is no independent test-set evaluation or downstream benchmark. Synchronous tuning is sequential and conditional; decentralized tuning uses wider joint grids, and the clipped and unclipped grids differ. The chosen winners are consequently selected observations from different procedures. Validation comparisons are exploratory even when budgets match exactly.

Three seeds provide limited information about run-to-run variability. Reused integer seeds do not ensure identical numerical trajectories, data-worker assignments, or stochastic kernels between algorithms. Reported sample SDs are descriptive and are not confidence intervals. Seed-wise differences are useful bookkeeping but do not by themselves establish a paired statistical design for every cross-method comparison.

The packed implementation provides four independent local optimizers on one physical device. It demonstrates an optimizer behavior, not multi-device speedup. Final parameter averaging is part of the evaluated algorithm. Throughput and memory are measured properties of the complete execution, including specified validation work, and should not be interpreted as pure model-compute or activation-memory costs.

The matched clipping grid contains successful finite results for every common seed-42 configuration. Nevertheless, differences in its medians or means remain conditional on the chosen grid and its LR range. The current evidence does not isolate why clipping is useful in a particular regime; more frequent clipping and greater benefit are distinct empirical quantities.

== Five conclusions
+ *Small batches can improve validation loss at fixed targets.* The strongest observed synchronous and clipped decentralized means occur at batches 64 and 32, respectively. Batch 16 is not consistently best.
+ *Adam memory should be retuned as the batch shrinks.* Higher $beta_1$ is consistently favored at small batches. Near-one $beta_2$ values merit exploration, but the second-moment trend is neither strictly monotonic nor sharply identified within the near-optimal plateau.
+ *Tuned clipped decentralized quality is close to synchronous quality.* Matching-batch gaps are small on the loss and perplexity scales, without establishing equivalence.
+ *Very small local batches clip frequently.* The extreme rate at local batch four contrasts with much lower rates in the selected synchronous runs. Frequency alone does not establish usefulness.
+ *Within the common grid, clipping helps more at larger batches.* No-clipping retuning partly compensates through smaller LRs. The intervention should therefore be considered jointly with batch, LR, and moment coefficients.

The most defensible practical recommendation is to tune the optimizer as a coupled system, report the target budget and throughput separately, and retain replicated near ties rather than overinterpreting a single winning coefficient.

#pagebreak()
#set heading(numbering: "A.1")
#counter(heading).update(0)
= Exact search protocols

== Shared budget and rounding
Context length is 512; global batch denotes sequences per update. The total update counts for batches 16, 32, 64, and 128 are respectively #num(408092672/(16*512), digits: 0), #num(408092672/(32*512), digits: 0), #num(408092672/(64*512), digits: 0), and #num(408092672/(128*512), digits: 0). Warmup ends after 2,496, 1,248, 624, and 312 updates. All runs use complete updates and 40 aligned virtual epochs. Historical batch-128 rounding is preserved by the nominal tokens-per-parameter settings: 20 and 0.5 for batch 128, and 20.001092 and 0.5000273 for smaller batches. The authoritative realized target counts are identical across all four batches. [S1, S2, S5]

With $c$ denoting global targets after an update, $C=408092672$, and $w=20447232$, the LR schedule is
$ eta(c)=cases(eta_max c/w & c < w, eta_max (1+cos(pi(c-w)/(C-w)))/2 & c >= w). $
The implementation clips cosine progress to $[0,1]$. Warmup reaches the configured peak exactly, and the final update uses zero LR. Logging intervals scale with global batch to compare target-scale progress.

== Clipped and no-clipping decentralized grids
The unified clipped search uses $beta_1$ in ${0.9,0.95,0.98,0.99,0.995}$ and $beta_2$ in ${0.98,0.99,0.999,0.9999,0.99999,0.999999,2^(-512B/10000000)}$. Its original campaign tests five second-moment values; the extension adds 0.99999 and 0.999999. The no-clipping grid drops $beta_1=0.9$, $beta_2=0.98$, and the largest LR in each row below. [S2, S5]

#tab((0.65fr, 0.65fr, 3fr), th[Global], th[Local], th[Clipped learning rates; last value omitted without clipping],
  [16], [4], [0.00025, 0.0005, 0.001, 0.002, 0.003, *0.004*],
  [32], [8], [0.0005, 0.001, 0.002, 0.003, 0.004, *0.006*],
  [64], [16], [0.001, 0.002, 0.003, 0.004, 0.006, *0.008*],
  [128], [32], [0.002, 0.003, 0.004, 0.006, 0.008, *0.012*],
)

The clipped union has $4 times 6 times 5 times 7=840$ screening points; the reduced no-clipping grid has $4 times 5 times 4 times 6=480$. At each stage and batch, five successful seed-42 candidates are frozen before additional seeds, with exact ties resolved by lower LR, $beta_1$, then $beta_2$. All primary observations completed successfully in the no-clipping study. Attempts, imports, scheduler receipts, and historical cancellations remain distinct from experimental observations.

#pagebreak()
== Synchronous grid and conditional first-moment tuning
The synchronous initial LR grid is ${0.002,0.003,0.004,0.006,0.008,0.012}$ at batches 16, 32, 64, and 128. At $beta_1=0.9$, both weight decays zero and 0.1 test two $beta_2$ policies:
$ beta_(2,"transferred")=0.98^(B/128), quad beta_(2,"10M")=2^(-512B/10000000). $
Weight decay 0.1 additionally tests fixed $beta_2=0.98$; coincident policy values are deduplicated. Selected LR endpoints may receive one factor-of-two extension. At weight decay 0.1, batches 16 and 32 add LR 0.001 for all three second-moment policies. Historical batch-128 observations span LRs 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, and 0.012 crossed with $beta_2$ values 0.95, 0.98, 0.99, and 0.999, always at $beta_1=0.9$. They remain eligible under their authenticated source identity. The current-source batch-128 screen adds the six 10M-half-life choices plus a fresh LR-0.008 / $beta_2=0.98$ repeat; the other initial fixed-0.98 choices reuse historical evidence. Full executed combinations are bundled. [S1]

The initial top two configurations per batch and decay receive seeds 43–44, followed by selection using their three-seed means. A later seed-42 first-moment diagnostic tests ${0,0.7,0.84,0.9,0.92,0.96,0.98,0.99}$ at each selected recipe's fixed LR and $beta_2$. One best $beta_1$ is frozen per batch and decay before additional seeds 43–44. The report's main synchronous curve uses the resulting weight-decay-0.1 recipes only. It does not pool the decay-zero arm into the decentralized comparison or present the diagnostic minima as replicated results.

== Result acceptance and selection eligibility
Completed results must match the frozen full configuration, source/cache identities, expected parameter count, exact training targets, complete final-validation targets, finite measurements, and original scheduler completion evidence. Decentralized results additionally require four-worker metadata and independent local optimizer execution. Saving remains disabled. Numerical failures would remain outcomes; they would not receive a finite loss, an artificial rank, or a numerical retry. A diagnosed infrastructure retry is allowed only once in a fresh attempt directory after terminal scheduler evidence. [S2, S5]

For no-clipping imports, a short recipe ID is insufficient because historical recipe keys can be identical across clipping settings. Authentication includes the resolved configuration with clipping disabled, source and cache identity, artifact hashes, original receipt task mapping, and scheduler completion. Seed-42 imports enter screening; seeds 43–44 are eligible for primary rankings only after their configuration is among the frozen top five at that batch. The remaining imported replications are supplemental and are excluded from promotion and primary ranking.

== Boundary definitions
LR, first-moment, and second-moment flags indicate equality with the minimum or maximum tested value for the relevant batch and method. Decentralized clipped flags use the entire unified grid; no-clipping flags use the reduced grid. Synchronous flags reflect its conditional protocol and retained historical boundaries. A flag is descriptive: it does not authorize further jobs, change a winner, or replace a frozen selection.

#pagebreak()
= Moment half-lives and clipping counters

== Exponential-memory units
For $0 < beta < 1$, a contribution's unnormalized exponential weight after $h$ updates is proportional to $beta^h$. Solving $beta^h=1/2$ gives
$ H_"updates"(beta)=-ln(2)/ln(beta), $
$ H_"global"(beta)=512 B H_"updates"(beta), quad H_"worker"(beta)=512 (B/W) H_"updates"(beta). $
Here $B$ is global batch and $W$ is the number of logical workers: one for synchronous training and four for decentralized training. The same definition applies separately to $beta_1$ and $beta_2$. At $beta=0$, no earlier update is retained and the half-life is zero by convention.

The special candidate $beta_2=2^(-512B/10000000)$ gives exactly 10 million global targets of nominal second-moment half-life. For four workers, its per-worker half-life is 2.5 million targets. This relation concerns consumed-target units; the workers' moment states remain independent. A nominal half-life longer than a run is valid and does not imply that the corresponding empirical moment estimate has stabilized.

== Count definitions and missing measurements
Let $I_(i,t)$ indicate that a worker's pre-clipping norm exceeds one on update $t$. For a decentralized run with $T$ updates, the pooled frequency is
$ f_"clip"=(sum_(t=1)^T sum_(i=1)^4 I_(i,t))/(4T). $
The per-worker rate is $sum_t I_(i,t)/T$. For synchronous training the denominator is $T$. Frequencies in the selected-recipe chart first pool each run's counters and then average the three seed rates; identical budgets make this equal to pooling those runs. All 40 epoch counters are included and cross-checked against text logs. [S3]

The epoch-3–40 diagnostic is explicitly a range of reporting epochs. It excludes most early training but does not equal the exact post-warmup interval. Missing historical counters remain null and are excluded from corresponding rate summaries. In particular, the final synchronous batch-128 historical reference has no reconstructed clipping frequency. Disabled clipping counters are zero by configuration and carry no claim about the unobserved threshold-exceedance rate in a counterfactual clipped run.

== Interpretation of variability
For losses $L_42,L_43,L_44$, the reported mean is $bar(L)=(L_42+L_43+L_44)/3$ and the sample SD is $sqrt(sum_s (L_s-bar(L))^2/2)$. The replication-only mean is $(L_43+L_44)/2$. For fixed-configuration interventions, paired seed differences are computed first and their sample SD is reported separately from each method's SD.

#pagebreak()
== Selected moment half-lives

Values below are in *millions of targets*. The update half-life is recovered by dividing the global-target value by $512B$. Synchronous per-worker and global values coincide. No-clipping and clipped rows retain distinct identities even when their coefficients match.

#let half-cells = ()
#for method in ("sync", "clipped", "unclipped") {
  for r in d.winners.at(method) {
    half-cells += ([#label(method)], [#r.batch], [#num(r.beta1_global_half_life_targets / 1e6, digits: 3)], [#num(r.beta1_per_worker_half_life_targets / 1e6, digits: 3)], [#num(r.beta2_global_half_life_targets / 1e6, digits: 3)], [#num(r.beta2_per_worker_half_life_targets / 1e6, digits: 3)])
  }
}
#tab((1.6fr, 0.45fr, 0.85fr, 0.85fr, 0.9fr, 0.9fr), th[Method], th[Batch], th[$beta_1$ global], th[$beta_1$ worker], th[$beta_2$ global], th[$beta_2$ worker], ..half-cells)

At the selected clipped batch-32 coefficient of $beta_2=0.999999$, the global-target half-life is #num(winner("clipped", 32).beta2_global_half_life_targets / 1e9, digits: 3) billion targets, far longer than the 408-million-target training budget. A near-optimal coefficient can therefore correspond to a very long nominal averaging horizon. This observation is descriptive; it does not establish that a larger coefficient beyond the grid would improve the final loss.

All replicated configurations' update, global-target, and per-worker target half-lives are included in the bundled machine-readable tables. Those values are derived from the stored coefficient and batch rather than inferred from a plot label, avoiding confusion between a worker half-life and a global half-life.

#let replicated-table(title, rows, source-ref) = {
  pagebreak()
  set page(paper: "a4", flipped: true, margin: (x: 15mm, top: 18mm, bottom: 17mm))
  set text(size: 10pt)
  heading(level: 1, numbering: none, title)
  [All losses are final full-validation cross-entropy. Mean and sample SD use seeds 42–44; “43–44” keeps the same recipe. Endpoint flags use that method's executed grid. #source-ref]
  v(8pt)
  let cells = ()
  for r in rows {
    cells += ([#r.batch], [#num(r.lr, digits: 5)], [#num(r.beta1, digits: 3)], [#num(r.beta2, digits: 10)], [#num(r.seed42)], [#num(r.seed43)], [#num(r.seed44)], [#num(r.mean_loss)], [#num(r.std_loss)], [#num(r.replication_mean_loss)], [#flags(r)])
  }
  tab((0.45fr, 0.65fr, 0.5fr, 1.1fr, 0.85fr, 0.85fr, 0.85fr, 0.85fr, 0.85fr, 0.85fr, 0.65fr),
    th[Batch], th[LR], th[$beta_1$], th[$beta_2$], th[Seed 42], th[Seed 43], th[Seed 44], th[Mean], th[SD], th[43–44], th[Flags], ..cells)
  v(8pt)
  text(size: 9pt)[Rows are ordered by batch and then by three-seed mean loss. Full recipe and configuration identities, source campaign, and half-lives remain in the CSV/JSON extracts. A coefficient shown to ten decimal places is a display value; full precision is retained in the data.]
}

#replicated-table([Appendix C · All clipped decentralized replications, batches 16 and 32], d.replicated.clipped.filter(r => r.batch <= 32).sorted(key: r => (r.batch, r.mean_loss)), [S2])
#replicated-table([Appendix C · All clipped decentralized replications, batches 64 and 128], d.replicated.clipped.filter(r => r.batch >= 64).sorted(key: r => (r.batch, r.mean_loss)), [S2])
#replicated-table([Appendix D · All no-clipping decentralized replications], d.replicated.unclipped.sorted(key: r => (r.batch, r.mean_loss)), [S5])

#pagebreak()
#counter(heading).update((5, 0))
#heading(level: 1, numbering: none)[Appendix E · Synchronous reference and earlier confirmation]

== Final conditional synchronous recipes
The following four weight-decay-0.1 recipes define the synchronous reference in the main figures. Their LR and $beta_2$ originate in the earlier tuning, and $beta_1$ is the frozen conditional seed-42 choice. These are not the same recipes as the earlier fresh-seed confirmation. [S1]

#let sync-cells = ()
#for r in d.replicated.sync {
  sync-cells += ([#r.batch], [#num(r.lr, digits: 4)], [#num(r.beta1)], [#num(r.beta2, digits: 10)], [#num(r.mean_loss)], [#num(r.std_loss)], [#num(r.replication_mean_loss)], [#flags(r)])
}
#tab((0.45fr, 0.6fr, 0.55fr, 1.3fr, 0.9fr, 0.9fr, 0.9fr, 0.6fr), th[Batch], th[LR], th[$beta_1$], th[$beta_2$], th[Mean], th[SD], th[43–44], th[Flags], ..sync-cells)
#let sync-seed-cells = ()
#for r in d.replicated.sync { sync-seed-cells += ([#r.batch], [#num(r.seed42)], [#num(r.seed43)], [#num(r.seed44)]) }
#tab((0.7fr, 1fr, 1fr, 1fr), th[Batch], th[Seed 42], th[Seed 43], th[Seed 44], ..sync-seed-cells)

== Earlier fresh seeds 45–47: a separate experiment
The earlier confirmation compares batch 64 with batch 128 at the original $beta_1=0.9$ recipes. The table shows the primary decay-0.1 arm; the original campaign also reports a secondary decay-zero arm. Negative differences favor batch 64. The descriptive paired 95% Student-$t$ interval has two degrees of freedom and crosses zero. These results do not confirm the later conditional $beta_1$ recipes and do not provide fresh-seed evidence for either decentralized study. [S1]

#let conf-cells = ()
#for r in d.confirmation {
  for p in r.pairs { conf-cells += ([#num(r.weight_decay, digits: 1)], [#p.seed], [#num(p.smaller_loss)], [#num(p.control_loss)], [#signed(p.delta)]) }
}
#tab((0.65fr, 0.65fr, 1fr, 1fr, 1fr), th[Decay], th[Seed], th[Batch 64], th[Batch 128], th[Difference], ..conf-cells)
#for r in d.confirmation [Decay #num(r.weight_decay, digits: 1): mean difference #signed(r.mean_delta), historical 95% interval [#signed(r.ci95_low), #signed(r.ci95_high)]; *inconclusive*. #parbreak()]

#pagebreak()
#counter(heading).update((6, 0))
#heading(level: 1, numbering: none)[Appendix F · Fixed-configuration clipping evidence]

== Individual ablation differences
The table retains all twelve original ablation observations as differences from their clipped controls at the same configuration and seed. Positive differences favor clipping. These observations are a selected subset; their reuse in the no-clipping campaign is deduplicated. [S4]

#let abl-seed-cells = ()
#for r in d.ablation {
  for i in range(3) {
    abl-seed-cells += ([#r.batch], [#num(r.lr, digits: 4)], [#num(r.beta2)], [#(42+i)], [#num(r.baseline_losses.at(i))], [#num(r.losses.at(i))], [#signed(r.seed_loss_differences.at(i))])
  }
}
#tab((0.5fr, 0.65fr, 0.95fr, 0.5fr, 1fr, 1fr, 1fr), th[Batch], th[LR], th[$beta_2$], th[Seed], th[Clipped], th[No clipping], th[Difference], ..abl-seed-cells)

== Three-seed overlap between tuning grids
Complete matched replications are available only where both tuning procedures selected the same configuration. The next page reports those mean differences and their seed-wise SDs. This overlap is a consequence of tuning, not a randomly sampled subset of the grid. A positive mean favors clipping; a negative mean favors no clipping. Each mean difference equals the arithmetic mean of the three seed differences, and both methods' individual losses remain in the bundled table. [S2, S5]

The batch-32 comparison is especially useful for interpreting LR dependence. At the larger selected clipped LR, retaining clipping improves the fixed-configuration mean. At a lower LR, a near-one $beta_2$ configuration can behave differently. This is why the final independently tuned winner comparison and the matched-configuration table are both needed.

#pagebreak()
== All matched replicated configurations

#let matched-rep-cells = ()
#for r in d.matched_replicated.sorted(key: r => (r.batch, r.lr, r.beta1, r.beta2)) {
  matched-rep-cells += ([#r.batch], [#num(r.lr, digits: 4)], [#num(r.beta1, digits: 3)], [#num(r.beta2)], [#signed(r.mean_loss_difference)], [#num(r.std_loss_difference)], [#signed(r.seed_loss_differences.at(0))], [#signed(r.seed_loss_differences.at(1))], [#signed(r.seed_loss_differences.at(2))])
}
#tab((0.4fr, 0.6fr, 0.5fr, 0.85fr, 0.95fr, 0.85fr, 0.95fr, 0.95fr, 0.95fr), th[Batch], th[LR], th[$beta_1$], th[$beta_2$], th[Mean Δ], th[SD of Δ], th[Δ42], th[Δ43], th[Δ44], ..matched-rep-cells)

All 480 common-grid seed-42 comparisons, including the exact pair of losses and the clipping-specific source identities, are supplied in the CSV/JSON extracts. Their distribution summaries use the full grid; no large-loss point is removed as an outlier. The figures may use a transformed axis for readability, but the exported losses and summary calculations remain on the original loss scale.

== Interpretation of the most informative contrasts
The batch-16 interventions suggest that disabling clipping can slightly improve some highly clipped configurations. The batch-32 interventions show that a recipe with low clipping frequency can nevertheless benefit from retaining clipping. At larger batches, the seed-42 common grid and separately tuned winners both favor clipping. These are compatible observations because clipping frequency, intervention effect at fixed settings, and the optimum after retuning are different quantities.

The apparent usefulness of clipping should not be inferred from the aggregate percentage alone. The timing and magnitude of the affected gradients, the learning-rate schedule, and the interaction with moments are all potential explanations. The saved metrics support the reported outcome comparisons but do not isolate those mechanisms.

#pagebreak()
#counter(heading).update((7, 0))
#heading(level: 1, numbering: none)[Appendix G · Provenance and reproducibility]

== Authoritative source records
The evidence package refers to the following completed repository-local records. Source IDs used in the prose and figure captions identify these records; the bundled source manifest stores file hashes and the extraction/build identities. No external research claim is needed to establish the numerical findings in this report.

#tab((0.4fr, 1.2fr, 2.8fr), th[ID], th[Record], th[Repository-relative location],
  [S1], [Synchronous tuning], [#path("runs/sync-20m-c512-small-batch/")],
  [S2], [Unified clipped decentralized tuning], [#path("runs/packed4-awc-exponential-20m-c512-small-batch-unified/")],
  [S3], [Pre-intervention clipping audit], [#path("runs/packed4-awc-exponential-20m-c512-small-batch-clipping/clipping-audit/")],
  [S4], [Fixed-configuration clipping ablation], [#path("runs/packed4-awc-exponential-20m-c512-small-batch-clipping/")],
  [S5], [No-clipping decentralized tuning], [#path("runs/packed4-awc-exponential-20m-c512-small-batch-no-clipping/")],
)

The unified clipped record indexes the original campaign and its second-moment extension while retaining their source-campaign identities and frozen promotion sets. It preserves canceled scheduler allocations and failed or partial attempts as provenance; those attempts are not counted as extra completed measurements. The no-clipping import ledger preserves the original artifact hashes, configuration identities, scheduler evidence, and historical account metadata of reused observations.

Full configuration identity includes clipping and seed; a short recipe label alone is not sufficient for observation identity. The bundled extraction deduplicates imported observations without fabricating local scheduler receipts. Promotion eligibility is verified against the frozen selections, and replications that were not selected remain supplemental. No observation is replaced because its replication result is unfavorable.

== Build from bundled extracts
From the document directory, run `./build.sh` to regenerate the figures and compile `slides.pdf` and `report.pdf`. The build uses bundled CSV/JSON data and requires no `runs/` directory or GPU. Use the documented `--refresh` option only when deliberately re-extracting the existing campaign artifacts; refresh records the source hashes and does not launch training. The presentation and report share the same derived measurements and consistent method colors.

Machine-readable extracts retain individual seeds, sample SDs, seeds-43–44 means, grid flags, optimizer half-lives, worker clipping counters, performance measurements, and source identities. They also preserve distinctions among final conditional synchronous tuning, earlier confirmation, the fixed-configuration ablation, and the no-clipping primary and supplemental observations.

#pagebreak()
== Acceptance checks and audit boundaries
The document build checks numerical aggregates against the bundled records: three-seed means and sample SDs; seed-specific differences; perplexity ratios; first- and second-moment half-lives; clipping-count denominators; frozen selection membership; and the complete 480-point matched grid. Source hashes identify the exact campaign artifacts used at refresh time. Figure values and numerical callouts are regenerated rather than transcribed from screenshots.

Presentation checks require exactly 13 slide pages, a 16:9 page size, and only 16, 32, 64, and 128 on batch axes. Both PDFs are rendered for visual inspection of text, tables, figures, and page boundaries. The report's wide tables use landscape A4 pages; other report pages use portrait A4. A portable rebuild from the bundled document directory verifies that figures and PDFs are independent of the original campaign paths.

These checks establish consistency of the document with the authenticated experiment records. They do not make the tuning procedures statistically equivalent, turn replication seeds into fresh confirmation, or justify untested scaling claims. A source identity establishes which code and configuration produced an observation; it is not proof that every possible implementation-level bias has been excluded.

#note[*Scope of this research record.* Results are those completed through 1 October 2026. No training, checkpoint creation, selection changes, or grid extensions are performed to produce the documents. The strongest claims are empirical and bounded by the model, budget, batch sizes, and tested configurations reported here.]

#v(10pt)
The frozen cache identity shared by the principal comparisons is #path(d.protocols.shared.cache_identity). The C4 dataset revision is #path(d.protocols.shared.dataset.revision); tokenizer revision is #path(d.protocols.shared.dataset.tokenizer_revision). Full source-file, manifest, selection, configuration, dependency, and extracted-result hashes are recorded in `provenance.json` and the bundled source entries, rather than replaced by mutable checkout state.

#pagebreak()
#counter(heading).update((8, 0))
#heading(level: 1, numbering: none)[Appendix H · Descriptive performance measurements]

The table gives medians across the source-specific screening pools. Rates are in millions of global targets per second; durations are measured session seconds; memory is peak allocated GiB over the recorded run, including final evaluation at batch 128. The pools differ and are not controlled repetitions of one recipe. Individual observations, quartiles, and source/environment strata remain in the bundled performance data. [S1, S2, S5]

#let perf-cells = ()
#for method in ("sync", "clipped", "unclipped") {
  for r in d.performance.filter(r => r.method == method).sorted(key: r => r.batch) {
    perf-cells += ([#label(method)], [#r.batch], [#r.at("runs", default: r.at("successful_screening_runs", default: none))], [#num(r.steady_targets_per_second / 1e6, digits: 3)], [#num(r.session_seconds, digits: 1)], [#num(r.peak_memory_gib, digits: 3)])
  }
}
#tab((1.6fr, 0.45fr, 0.5fr, 0.9fr, 0.9fr, 0.9fr), th[Method], th[Batch], th[Runs], th[Steady M/s], th[Session s], th[Peak GiB], ..perf-cells)

Synchronous screening uses the available current-source weight-decay-0.1 runs. The clipped decentralized summary uses 210 screening configurations per batch; no clipping uses the reduced 120-configuration grid. A historical selected synchronous recipe can therefore differ from the runs used for the corresponding performance pool. In particular, the historical batch-128 clipping rate is missing even though current-source batch-128 performance measurements exist. These are distinct records with distinct purposes.

The memory peaks are close in magnitude across batch sizes. They summarize the full measured execution, including fixed-batch evaluation, parameter and moment storage, and compiled execution. This measurement scope is consistent with the absence of a simple monotonic training-batch relationship, but the metrics do not identify the allocation that sets each peak. A controlled training-only memory experiment would be needed to answer that narrower question; no such experiment is added here.
