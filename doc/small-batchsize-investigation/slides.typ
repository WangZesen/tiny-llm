#import "@preview/typslides:1.3.4": typslides, blank-slide

#let d = json("data/analysis.json")
#let navy = rgb("17324D")
#let blue = rgb("0072B2")
#let purple = rgb("8B61B3")
#let orange = rgb("D55E00")
#let gray = rgb("626C78")
#let pale = rgb("F1F5F8")
#let num(x, digits: 6) = str(calc.round(x, digits: digits))
#let signed(x, digits: 6) = if x < 0 { "−" + num(-x, digits: digits) } else { "+" + num(x, digits: digits) }
#let winner(method, batch) = d.winners.at(method).find(r => r.batch == batch)
#let best(method) = d.winners.at(method).sorted(key: r => r.mean_loss).first()
#let sync-best = best("sync")
#let clipped-best = best("clipped")
#let gaps = (16, 32, 64, 128).map(b => winner("clipped", b).mean_loss - winner("sync", b).mean_loss)
#let gain(method) = winner(method, 128).mean_loss - best(method).mean_loss
#let unclip-delta(batch) = winner("unclipped", batch).mean_loss - winner("clipped", batch).mean_loss

#show: typslides.with(ratio: "16-9", theme: navy, font: "Nimbus Sans", font-size: 23pt, show-page-numbers: false)
#set document(title: "Small-Batch Training with AdamW", description: "Synchronous and four-worker decentralized training: batch size, moment memory, and gradient clipping.", date: datetime(year: 2026, month: 10, day: 1))
#show math.equation: set text(font: "New Computer Modern Math")
#set par(spacing: 0pt, leading: .3em)
#set block(spacing: 0pt)

#let frame(n, title, sources, body) = {
  blank-slide[
    #set page(margin: (x: .85cm, top: .65cm, bottom: .8cm), background: none, foreground: none,
      footer: context grid(columns: (1fr, auto), text(10pt, fill: gray)[#sources · Definitions and provenance in the report], text(11pt, fill: gray)[#n / 13]))
    #set align(top + left)
    #set text(size: 23pt, fill: navy)
    #set par(justify: false, spacing: 0pt, leading: .3em)
    #text(10pt, tracking: 1.5pt, fill: gray)[SMALL-BATCH TRAINING / INVESTIGATION]
    #v(.16cm)
    #text(28pt, weight: "bold")[#title]
    #v(.24cm)
    #line(length: 100%, stroke: .6pt + rgb("CBD5DF"))
    #v(.32cm)
    #body
  ]
  pagebreak(weak: true)
}
#let small(body) = text(15pt, fill: gray)[#body]
#let label(body) = text(16pt, weight: "bold", fill: gray)[#body]
#let note(body) = block(width: 100%, fill: pale, inset: (x: .3cm, y: .2cm), radius: 3pt)[#text(20pt)[#body]]
#let metric(value, body, color: navy) = [#text(26pt, weight: "bold", fill: color)[#value] #text(18pt)[#body]]
#let two(left, right) = grid(columns: (1fr, 1fr), gutter: .6cm, left, right)
#let chart(name, alt) = image("assets/" + name + ".svg", width: 28cm, height: 8cm, fit: "contain", alt: alt)

#blank-slide[
  #set page(margin: (x: 1.45cm, y: 1.0cm), background: none, foreground: none,
    footer: align(right, text(font: "Nimbus Sans", size: 11pt, fill: gray)[1 / 13]))
  #set align(top + left)
  #set text(fill: navy)
  #text(12pt, tracking: 1.8pt, fill: gray)[LANGUAGE-MODEL OPTIMIZATION / 1 OCTOBER 2026]
  #v(1.2cm)
  #text(43pt, weight: "bold")[Small-Batch Training\ with AdamW]
  #v(.55cm)
  #text(27pt)[Synchronous and four-worker\ decentralized training]
  #v(.65cm)
  #line(length: 100%, stroke: 2pt + navy)
  #v(.55cm)
  #text(22pt)[How do batch size, moment memory, and gradient clipping\ affect validation quality at a fixed token budget?]
  #v(.85cm)
  #grid(columns: (1fr, 1fr, 1fr), gutter: .4cm,
    text(17pt, fill: blue)[01  BATCH & MEMORY],
    text(17pt, fill: purple)[02  DECENTRALIZATION],
    text(17pt, fill: orange)[03  CLIPPING])
]
#pagebreak(weak: true)

#frame(2, [One token budget, two optimization systems], [S1–S5])[
  #two([
    #label[MODEL, DATA, AND EVALUATION]
    #v(.25cm)
    #text(31pt, weight: "bold")[20.4M parameters]
    #v(.15cm)
    #text(21pt)[C4 · context 512 · 40 virtual epochs\ 408,092,672 global training targets\ 20,447,232 warmup targets\ 197,411,295 final validation targets]
    #v(.5cm)
    #label[SHARED OPTIMIZER SETTINGS]
    #v(.2cm)
    #text(20pt)[AdamW · weight decay 0.1 · ε = 10⁻⁸\ Cosine LR decay to zero\ FP32 parameters / moments · BF16 AMP]
  ], [
    #label[GLOBAL BATCH: 16, 32, 64, 128]
    #v(.25cm)
    #block(fill: pale, inset: .3cm, width: 100%, radius: 3pt)[
      #text(21pt, weight: "bold", fill: blue)[Synchronous]
      #v(.1cm)
      #text(20pt)[One model, one gradient, one Adam state]
    ]
    #v(.3cm)
    #block(fill: pale, inset: .3cm, width: 100%, radius: 3pt)[
      #text(21pt, weight: "bold", fill: purple)[Four-worker decentralized]
      #v(.1cm)
      #text(20pt)[Local batches: 4, 8, 16, 32\ Independent moments; reciprocal AWC\ Evaluate the FP32 mean of four models]
    ]
    #v(.3cm)
    #text(20pt)[Four logical workers share one GH200.]
  ])
  #v(.65cm)
  #note[Same training targets and full validation; different tuning procedures.]
  #v(.18cm)
  #small[Synchronous: conditional LR / β₂ then β₁ tuning. Decentralized: joint grids and frozen replication selections.\ No adaptive consensus, checkpoints, or saved weights. All quality comparisons are exploratory.]
]

#frame(3, [Small batches can improve validation quality], [S1–S2])[
  #chart("quality", "Three-seed mean validation loss and sample standard deviation for synchronous and clipped decentralized training at global batches 16, 32, 64, and 128.")
  #v(.15cm)
  #two(
    metric(num(sync-best.mean_loss), [synchronous · batch #sync-best.batch], color: blue),
    metric(num(clipped-best.mean_loss), [decentralized · batch #clipped-best.batch], color: purple))
  #v(.2cm)
  #small[Improvement over batch 128: #num(gain("sync")) synchronous; #num(gain("clipped")) decentralized.\ Lower loss is better. Batch 16 is not best. Bars: sample SD over seeds 42–44, not confidence intervals.]
]

#frame(4, [Decentralized training approaches synchronous quality], [S1–S2])[
  #chart("gap", "Differences between separately tuned clipped decentralized and synchronous validation means; all four observed gaps favor synchronous training.")
  #v(.12cm)
  #two(
    metric([#num(calc.min(..gaps))–#num(calc.max(..gaps))], [loss gap], color: purple),
    metric([#num(100 * (calc.exp(calc.min(..gaps)) - 1), digits: 2)–#num(100 * (calc.exp(calc.max(..gaps)) - 1), digits: 2)%], [higher perplexity]))
  #v(.22cm)
  #small[Clipping threshold 1 for both methods. These are observed differences between selected recipes,\ not a statistical equivalence test. Identical seed numbers do not imply identical trajectories.]
]

#frame(5, [Higher β₁ preserves memory as batches shrink], [S1–S2, S5])[
  #chart("beta1", "Selected first-moment decay coefficients and their global-token half-lives across batch sizes.")
  #v(.08cm)
  #two([
    #text(22pt)[$ H_("global") = -(B dot 512 dot ln 2) / (ln beta) $]
  ], [#metric([≈0.56M], [global targets], color: blue)\ #text(18pt)[Synchronous β₁ memory at B = 16–64]])
  #v(.2cm)
  #small[At fixed β, a smaller batch shortens memory in tokens. Increasing β can preserve the token horizon.\ This interpretation of selected recipes is not a validated universal scaling law.]
]

#frame(6, [Long second-moment memory helps small batches], [S1–S2])[
  #chart("beta2", "Seed-42 second-moment profiles and replicated near-optimal high-beta2 recipes, showing diminishing gains near one.")
  #v(.1cm)
  #let near = d.replicated.clipped.filter(r => r.batch == 32 and r.lr == 0.004 and r.beta1 == 0.98 and r.beta2 >= 0.9999)
  #note[At batch 32, three β₂ choices near one differ by only #num(calc.max(..near.map(r => r.mean_loss)) - calc.min(..near.map(r => r.mean_loss))) in mean loss.]
  #v(.2cm)
  #small[Profiles minimize over LR and β₁ using seed 42: screening diagnostics, not replicated estimates.\ The β₂ trend is not monotonic; the largest tested value is not consistently best.]
]

#frame(7, [Small local batches clip frequently], [S2–S3])[
  #chart("clipping", "Clipping frequencies for selected recipes: very frequent local clipping at decentralized local batch four, compared with infrequent synchronous clipping.")
  #v(.1cm)
  #let c16 = d.clipping_selected.clipped.find(r => r.batch == 16).clipping_frequency
  #let c32 = d.clipping_selected.clipped.find(r => r.batch == 32).clipping_frequency
  #two(metric([#num(100 * c16, digits: 2)%], [local batch 4], color: purple),
    metric([#num(100 * c32, digits: 2)%], [local batch 8], color: purple))
  #v(.2cm)
  #small[Threshold 1; mean over workers and seeds. Synchronous selected batches 16–64 clip on less than 0.64% of updates.\ Historical synchronous batch-128 counters are unavailable. Frequency does not establish benefit.]
]

#frame(8, [Frequent clipping need not be beneficial], [S3–S4])[
  #chart("ablation", "For four fixed decentralized configurations, removing clipping improves the two batch-16 means and worsens the two batch-32 means, with substantial seed variation.")
  #v(.12cm)
  #let a16 = winner("unclipped", 16).mean_loss - winner("clipped", 16).mean_loss
  #note[Batch 16 can improve when clipping is removed; the original batch-32 recipes favor clipping.]
  #v(.17cm)
  #small[Δ = no clipping − clipping; negative favors removal. All other recipe settings are held fixed.\ Four selected configurations × three seeds. Controls were selected on these seeds; this is not fresh confirmation.]
]

#frame(9, [Clipping helps more across the larger-batch grids], [S2, S5])[
  #chart("matched-grid", "Loss-difference distributions for 120 matched seed-42 configurations per batch, retaining all poor outcomes and showing medians and no-clipping win counts.")
  #v(.08cm)
  #let wins = (16, 32, 64, 128).map(b => d.matched_screening.filter(r => r.batch == b and r.loss_difference < 0).len())
  #note[No-clipping wins: #wins.at(0)/120 → #wins.at(1)/120 → #wins.at(2)/120 → #wins.at(3)/120 as batch increases.]
  #v(.18cm)
  #small[All 480 matched seed-42 pairs succeeded. Symmetric-log loss axis; linear within ±0.025.\ LR, β₁, β₂, batch, and seed match within each pair. Poor configurations skew mean differences.]
]

#frame(10, [Removing clipping changes the preferred LR], [S1–S2, S4–S5])[
  #chart("retuned", "Final separately tuned means for all three methods and lower selected learning rates after disabling clipping.")
  #v(.08cm)
  #two(metric(signed(unclip-delta(16)), [retuned Δ at batch 16], color: orange),
    metric(signed(unclip-delta(128)), [retuned Δ at batch 128], color: orange))
  #v(.17cm)
  #small[At batch 32, removal helps matched replicated LR=.003 recipes but hurts the original LR=.004 controls.\ Clipping interacts with LR: separately retuned winners do not isolate its causal effect.]
]

#frame(11, [Better token efficiency has a throughput cost], [S1–S2, S5])[
  #chart("throughput", "Median steady global training-target throughput versus batch size for synchronous and decentralized campaign runs.")
  #v(.08cm)
  #note[A fixed training-token budget is not a fixed wall-clock budget.]
  #v(.2cm)
  #small[Descriptive medians from the recorded execution stratum; configuration pools and job placements differ.\ Four logical workers use one GH200, not four physical GPUs. Memory peaks include batch-128 evaluation.]
]

#frame(12, [Replication changes fine-grained β₂ rankings], [S2, S5])[
  #chart("replication", "Seed-42 and three-seed comparisons illustrate uncertainty in choosing among near-optimal second-moment coefficients.")
  #v(.08cm)
  #note[Batch 16 without clipping: β₂=0.999999 leads screening; β₂=0.9999 leads after replication.]
  #v(.18cm)
  #small[Seeds 43–44 contribute to final ranking; they are not fresh confirmation. Remaining no-clipping endpoints:\ upper β₂ at batch 32; lower β₁ at 64 and 128. Neither endpoint establishes an optimum beyond the grid.]
]

#frame(13, [Five findings, one joint tuning problem], [S1–S5])[
  #let finding(n, heading, body) = grid(columns: (1cm, 1fr), gutter: .2cm,
    text(26pt, weight: "bold", fill: purple)[#n],
    [#text(22pt, weight: "bold")[#heading] #text(20pt)[#body]])
  #finding([01], [Small batches can improve quality.], [The best observed batches are 32–64, not always 16.])
  #v(.33cm)
  #finding([02], [Moment memory matters.], [Smaller batches favor higher β₁; β₂ benefits plateau near one.])
  #v(.33cm)
  #finding([03], [The decentralized gap is small.], [Clipped recipes approach synchronous quality in this experiment.])
  #v(.33cm)
  #finding([04], [Small local batches clip often.], [This rate measures intervention frequency, not usefulness.])
  #v(.33cm)
  #finding([05], [Clipping helps larger batches more here.], [Its value still depends on LR and the tested grid.])
  #v(.6cm)
  #note[Tune batch size, LR, β₁, β₂, and clipping together; judge quality and cost separately.]
  #v(.25cm)
  #small[Lowest observed means: synchronous B#sync-best.batch, #num(sync-best.mean_loss) · clipped decentralized B#clipped-best.batch, #num(clipped-best.mean_loss).\ Three-seed tuning results support these findings; they do not establish universal optima or statistical equivalence.]
]
