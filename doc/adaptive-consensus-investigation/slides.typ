#import "@preview/typslides:1.3.4": typslides, blank-slide

#let navy = rgb("17324D")
#let blue = rgb("0072B2")
#let orange = rgb("D55E00")
#let green = rgb("009E73")
#let purple = rgb("CC79A7")
#let gray = rgb("626C78")
#let pale = rgb("F1F5F8")
#show: typslides.with(ratio: "16-9", theme: navy, font: "Nimbus Sans", font-size: 23pt, show-page-numbers: false)
#set document(title: "Adaptive Consensus: From Diagnostics to Training Dynamics", description: "An experimental investigation of adaptive consensus and moment-scaled consensus errors in 20.4M-parameter language-model training.")
#show math.equation: set text(font: "New Computer Modern Math")
#set par(spacing: 0pt, leading: .35em)
#set block(spacing: 0pt)

// Use the package's blank-slide base with a restrained, document-local layout.
#let frame(n, title, sources, body) = {
  blank-slide[
    #set page(margin: (x: .85cm, top: .65cm, bottom: .8cm), background: none, foreground: none,
      footer: context grid(columns: (1fr, auto), text(10pt, fill: gray)[#sources · Sources and definitions in the supplement], text(11pt, fill: gray)[#n / 12]))
    #set align(top + left)
    #set text(size: 23pt, fill: navy)
    #set par(justify: false, spacing: 0pt, leading: .35em)
    #text(10pt, tracking: 1.6pt, fill: gray)[ADAPTIVE CONSENSUS / INVESTIGATION]
    #v(.16cm)
    #text(28pt, weight: "bold")[#title]
    #v(.24cm)
    #line(length: 100%, stroke: .6pt + rgb("CBD5DF"))
    #v(.35cm)
    #body
  ]
  pagebreak(weak: true)
}
#let small(body) = text(17pt, fill: gray)[#body]
#let note(body) = block(width: 100%, fill: pale, inset: (x: .35cm, y: .22cm), radius: 3pt)[#text(20pt)[#body]]
#let chart(name, alt) = image("assets/" + name + ".svg", width: 28cm, height: 8cm, fit: "contain", alt: alt)
#let metric(value, label, color: navy) = [#text(27pt, weight: "bold", fill: color)[#value] #text(19pt)[#label]]
#let two(left, right, gutter: .8cm) = grid(columns: (1fr, 1fr), gutter: gutter, left, right)
#let label(body) = text(16pt, weight: "bold", fill: gray)[#body]

#blank-slide[
  #set page(margin: (x: 1.5cm, y: 1.1cm), background: none, foreground: none, footer: none)
  #set align(top + left)
  #set text(fill: navy)
  #text(13pt, tracking: 2pt, fill: gray)[LANGUAGE-MODEL OPTIMIZATION / 11 SEPTEMBER 2026]
  #v(1.2cm)
  #text(47pt, weight: "bold")[Adaptive Consensus]
  #v(.3cm)
  #text(32pt)[From diagnostics\ to training dynamics]
  #v(.9cm)
  #line(length: 100%, stroke: 2pt + navy)
  #v(.6cm)
  #text(23pt)[A 20.4M-parameter language model.\ Four local models. One investigation.]
  #v(1cm)
  #grid(columns: (1fr, 1fr, 1fr), gutter: .7cm,
    [#text(16pt, fill: blue)[01  GRADIENT NOISE]],
    [#text(16pt, fill: green)[02  CURVATURE]],
    [#text(16pt, fill: orange)[03  TRAINING DYNAMICS]])
]
#pagebreak(weak: true)

#frame(1, [A controlled, small language-model experiment], [S1–S3])[
  #two([
    #label[MODEL]
    #v(.3cm)
    #text(36pt, weight: "bold")[20.4M] #text(22pt)[parameters]
    #v(.2cm)
    #text(22pt)[8 layers · width 320 · 5 heads\ FFN width 896 · vocabulary 32,000]
    #v(.5cm)
    #label[DATA & BUDGET]
    #v(.2cm)
    #text(24pt)[C4 · TinyLlama tokenizer]
    #v(.2cm)
    #text(22pt)[Context 1,024 · ≈408M tokens\ 40 virtual epochs · seed 42]
  ], [
    #label[OPTIMIZATION]
    #v(.3cm)
    #text(26pt, weight: "bold")[AdamW]
    #v(.2cm)
    $ (beta_1, beta_2) = (0.9, 0.999) $
    #v(.1cm)
    #text(22pt)[Weight decay 0.1\ Gradient clipping at 1.0]
    #v(.5cm)
    #label[LEARNING-RATE SCHEDULE]
    #v(.2cm)
    #text(22pt)[5% token warmup\ Cosine decay to 10% of peak]
  ])
  #v(.7cm)
  #note[Same architecture, data source, global batch, and ≈408M-token budget.]
  #v(.15cm)
  #small[The tied embedding / output matrix accounts for 10.24M parameters.]
]

#frame(2, [Four local models, independent optimizer states], [S1–S3])[
  #two([
    #label[SYNCHRONOUS]
    #v(.3cm)
    #block(width: 100%, fill: pale, inset: .35cm, radius: 4pt)[
      #align(center)[#text(23pt)[32,768-token batch]\ $ arrow.b $\ #text(26pt, weight: "bold")[One model + AdamW]]
    ]
    #v(.35cm)
    #text(22pt)[Peak LR: 0.0022]
    #v(.2cm)
    #text(22pt)[One shared gradient and state]
  ], [
    #label[PACKED DECENTRALIZED]
    #v(.3cm)
    #align(center)[#text(22pt)[4 × 8,192-token local batches]]
    #v(.2cm)
    #grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: .12cm,
      ..range(4).map(i => block(fill: pale, inset: (x: .12cm, y: .3cm), radius: 4pt)[#align(center)[#text(20pt, weight: "bold")[Model #i]\ #text(16pt)[$m_i, v_i$]]]))
    #v(.25cm)
    #align(center)[#text(20pt, fill: blue)[Mix parameters each step]]
    #v(.25cm)
    #text(22pt)[Peak LR: 0.0048]
    #v(.2cm)
    #text(22pt)[Evaluate $overline(theta) = 1/4 sum_i theta_i$]
  ])
  #v(.65cm)
  #note[Each experiment runs on one GH200; “packed” describes the implementation.]
  #v(.2cm)
  #small[Baseline topology: one peer per step, with alternating incoming offsets 1 and 2.]
]

#frame(3, [Adaptive consensus weakens mixing as LR decays], [S3, S7])[
  #text(24pt)[Mix local parameters, then apply local AdamW:]
  #v(.3cm)
  $ b_(i,t) = sum_j W_(i j,t) theta_(j,t), quad
    y_(i,t) = gamma_t b_(i,t) + (1-gamma_t) theta_(i,t) $
  #v(.3cm)
  #block(fill: pale, inset: .35cm, width: 100%, radius: 4pt)[
    $ gamma_t = cases(1 & t < t_0,
      (eta_t / eta_(max, "active"))^p & t >= t_0),
      quad t_0 = ceil(0.25 T), quad p = 2 $
  ]
  #v(.35cm)
  #two([
    #metric([$gamma = 1$], [ordinary decentralized mixing])
  ], [
    #metric([$gamma_("end") approx 0.0122$], [end of adaptive run], color: orange)
  ])
  #v(.4cm)
  #text(22pt)[Local gradient → parameter mixing → local AdamW update]
  #v(.25cm)
  #small[Gradients use pre-mixing parameters. Moments remain local. The LR normalizer is the maximum within the active interval.]
]

#frame(4, [Gradient noise grows relative to the mean gradient], [S1–S2])[
  #chart("noise", "Unseen gradient noise and mean-gradient norms over training for synchronous and four-local-model packed runs.")
  #v(.15cm)
  #two([
    #metric([1.28 → 2.61], [synchronous noise / mean], color: blue)
  ], [
    #metric([2.78 → 6.41], [packed local noise / mean], color: blue)
  ])
  #v(.18cm)
  #small[Unseen training-cache data. Ratios compare average noise norm with mean-gradient norm.\ Different local batch sizes and LRs: interpret the trend within each run.]
]

#frame(5, [Adaptive consensus shows no observed advantage], [S2–S3])[
  #chart("baseline-loss", "Subset-validation loss for ordinary packed and adaptive-consensus training with matched exponential topology.")
  #v(.12cm)
  #grid(columns: (1fr, 1fr, 1fr), gutter: .3cm,
    metric([3.5959], [plain], color: blue),
    metric([3.6062], [adaptive], color: orange),
    metric([+0.0103], [AC − plain]))
  #v(.18cm)
  #small[Callouts: final full-validation loss. Curves: fixed subset. Same topology, LR and budget; one seed.]
]

#frame(6, [Consensus error has lower curvature than noise], [S3])[
  #chart("alignment", "Log-scale Hessian Rayleigh quotient for unseen gradient noise, consensus errors and isotropic random directions.")
  #v(.06cm)
  #grid(columns: (1.25fr, 1fr), gutter: .35cm,
    text(23pt)[$ A(D; H) = (sum_i d_i^T H d_i) / (sum_i norm(d_i)^2) $],
    [#text(23pt, weight: "bold")[236× below noise] #text(19pt)[at epoch 20]])
  #v(.16cm)
  #small[$delta_i = theta_i - overline(theta)$. $H$ is evaluated at $overline(theta)$ on unseen training-cache data.\ Consensus error: local model minus global average. It scores above the random reference.]
]

#frame(7, [Weighting by v increases directional curvature], [S4])[
  #chart("scaled-alignment", "The Hessian Rayleigh quotient increases when saved adaptive-consensus errors are multiplied by raw second moments or their square roots.")
  #v(.12cm)
  #two([
    #metric([≈41×], [$delta sqrt(v)$ versus $delta$], color: purple)
  ], [
    #metric([≈2,178×], [$delta v$ versus $delta$], color: green)
  ])
  #v(.18cm)
  #small[Epoch 20, unseen data; raw local second moments. These are transformed directions at saved AC checkpoints.]
]

#frame(8, [Local second moments are highly similar], [S5])[
  #chart("moment-similarity", "Across the six worker pairs, second-moment cosine remains near one while symmetric relative L2 distance remains small.")
  #v(.12cm)
  #grid(columns: (1.25fr, 1fr, 1fr), gutter: .25cm,
    [#text(27pt, weight: "bold", fill: green)[0.9981]\ #text(19pt)[$v$ cosine]],
    [#text(27pt, weight: "bold")[0.0104]\ #text(19pt)[$m$ cosine]],
    [#text(27pt, weight: "bold")[0.0037]\ #text(19pt)[Adam-direction cosine]])
  #v(.18cm)
  #small[Epoch 40, all unique parameters. Bands show six-pair ranges. Global similarity allows coordinatewise differences.]
]

#frame(9, [Reshape the retained consensus error], [S6, S8])[
  #label[TESTED DESIGN · COMPLETE TOPOLOGY · MAIN EXPERIMENT: α = 0.5]
  #v(.35cm)
  $ b_(i,t) = overline(theta)_t, quad
    e_(i,t) = (1-gamma_t)(theta_(i,t)-b_(i,t)) $
  #v(.35cm)
  $ s_(i,t) = e_(i,t) ⊙ v_(i,t)^alpha $
  #v(.2cm)
  #block(fill: pale, inset: .4cm, width: 100%, radius: 4pt)[
    $ y_(i,t) = b_(i,t) + (norm(e_(i,t))_2 / norm(s_(i,t))_2) s_(i,t) $
  ]
  #v(.4cm)
  #text(23pt)[Then apply the same local AdamW update.]
  #v(.3cm)
  #two([
    #text(22pt)[$v_(i,t)$: previous-step raw local moment]
  ], [
    #text(22pt)[Normalize across all parameters per worker]
  ])
  #v(.3cm)
  #small[$⊙$ denotes elementwise multiplication. Reweighting preserves $norm(e_i)_2$; the global average can shift.\ Zero transformed direction falls back to the original consensus error.]
]

#frame(10, [Moment-scaled training does not beat full mixing], [S6])[
  #chart("scaled-loss", "Representative complete-topology runs show validation spikes under positive moment exponents and no improvement over full mixing.")
  #v(.1cm)
  #grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: .15cm,
    [#text(23pt, weight: "bold", fill: green)[3.5799]\ #text(17pt)[Full mixing]],
    [#text(23pt, weight: "bold", fill: blue)[3.5827]\ #text(17pt)[$alpha = -0.5$]],
    [#text(23pt, weight: "bold", fill: orange)[3.6002]\ #text(17pt)[$alpha = 0.5$]],
    [#text(23pt, weight: "bold", fill: purple)[4.0264]\ #text(17pt)[$alpha = 0.1$]])
  #v(.14cm)
  #small[Final full-validation loss; one seed. Curves use the subset.\ This review lacks the complete-topology, all-parameter adaptive $alpha = 0$ control.]
]

#frame(11, [A fixed norm can hide extreme concentration], [S6])[
  #chart("diagnosis", "In the alpha equals one-half run, observed consensus error energy concentrates into one coordinate while worker second moments diverge.")
  #v(.04cm)
  #grid(columns: (1.35fr, 1fr), gutter: .35cm,
    text(22pt)[$ e_j^((k))/e_l^((k)) = e_j^((0))/e_l^((0)) (v_j/v_l)^(k alpha) $],
    [#text(26pt, weight: "bold", fill: orange)[95.7%] #text(19pt)[in one coordinate]])
  #v(.16cm)
  #small[Illustration: fixed positive $v$, two symmetric workers, complete topology, no optimizer forcing.\ Callout: worst worker, epoch 40. Repeated reweighting amplifies coordinate ratios.]
]

#frame(12, [A better diagnostic does not guarantee better training], [S3–S6])[
  #grid(columns: (.55cm, 1fr), column-gutter: .35cm, row-gutter: .4cm,
    text(23pt, fill: blue)[1], text(23pt)[Noise becomes more prominent relative to the mean gradient.],
    text(23pt, fill: green)[2], text(23pt)[Moment weighting improves curvature on saved AC states.],
    text(23pt, fill: orange)[3], text(23pt)[Repeated weighting of consensus error concentrates its energy.])
  #v(.65cm)
  #block(fill: pale, inset: .45cm, width: 100%, radius: 4pt)[
    #text(14pt, weight: "bold", fill: gray)[PROPOSED AT THE REVIEW STAGE · NO TRAINING EVIDENCE IN THIS REVIEW]
    #v(.25cm)
    #text(25pt, weight: "bold")[Shape fresh optimizer-update disagreement]
    #v(.25cm)
    #text(22pt)[Shared, bounded gains · common normalization\ Preserve the mean update; let ordinary AC contract existing error.]
  ]
  #v(.35cm)
  #small[Proposed evaluation: add the matched control and judge replicated validation results.\ Scope ends at the moment-scaling review; subsequent trials are outside this presentation.]
]
