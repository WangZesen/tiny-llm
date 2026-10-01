# Small-Batch Training with AdamW

[Presentation — 13 slides](slides.pdf) · [Report and technical appendix — 28 pages](report.pdf)

A formal investigation of synchronous and four-worker decentralized AdamW,
using completed evidence through **1 October 2026**. The presentation has one
title slide and twelve content slides. The report includes the full replicated
results, exact protocols, definitions, and provenance.

## Findings

- At a fixed training-token budget, tuned batches 32–64 can outperform batch 128;
  batch 16 is not consistently best.
- Smaller batches favor higher β₁ and longer moment memory in update units.
  The β₂ trend is not strictly monotonic, with small differences near one.
- Clipped four-worker decentralized training approaches synchronous validation
  quality: observed gaps are 0.004224–0.008296 loss, about 0.42–0.83% perplexity.
- Local gradient clipping becomes very frequent at the smallest decentralized
  batch; synchronous selected recipes clip much less frequently.
- Within the tested grids, clipping offers greater benefit at larger batches,
  despite acting less frequently. Its benefit depends on the learning rate.

These are bounded experimental findings. Three-seed error bars show sample SD,
not confidence intervals. Tuning procedures differ between methods, and seeds
43–44 contributed to selection. The final recipes have no fresh-seed confirmation.
Four logical workers share one GH200; the study does not measure four-GPU scaling.

## Build

Requirements: Typst **0.15.1 or later**, Python with Matplotlib and NumPy, and
Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`). The Typst template is pinned to
`@preview/typslides:1.3.4`; it may be downloaded on first use. Text uses Nimbus
Sans and Libertinus Serif, with New Computer Modern Math for equations.

From the repository root:

```bash
bash doc/small-batchsize-investigation/build.sh
```

This regenerates the vector figures from bundled data, compiles both PDFs, and
checks the statistics and document structure. It requires neither training
checkpoints nor access to `runs/`. The build selects the repository's
`.venv-x86_64/bin/python` when available, otherwise `python3`. Override it with
`TINY_LLM_DOC_PYTHON` if needed.

Refresh the bundled evidence from the existing campaigns:

```bash
bash doc/small-batchsize-investigation/build.sh --refresh
```

Refresh only reads saved experiment artifacts. It does not train, change frozen
selections, or submit jobs. Review the prose and figures if refreshed evidence
changes any finding. To check the original source hashes as well as the portable
document package, run `verify.py --check-sources` with the same Python environment.

To compile just the editable Typst sources, run from this directory:

```bash
typst compile --root . slides.typ slides.pdf
typst compile --root . report.typ report.pdf
```

## Evidence and contents

The source identifiers used in both documents are:

| ID | Evidence |
| --- | --- |
| S1 | Synchronous small-batch study and final conditional β₁ replication |
| S2 | Unified clipped decentralized study: 840 screened and 40 replicated configurations |
| S3 | Clipping-rate audit of selected recipes and synchronous screening runs |
| S4 | Fixed-configuration, three-seed clipping ablation |
| S5 | No-clipping tuning study: 480 screened and 20 replicated configurations |

The no-clipping study's primary dataset has 520 observations, including eight
authenticated imports. Four unselected historical replications remain
supplemental. The original 12-run clipping ablation overlaps these observations;
it is not twelve additional independent measurements. Screening, fixed-recipe
interventions, and comparisons of separately tuned winners remain distinct.

- `slides.typ` and `report.typ`: editable document sources.
- `build_figures.py`: evidence extraction and figure generation.
- `data/`: bundled measurements, definitions, and derived analysis.
- `assets/`: vector PDF and SVG figures.
- `provenance.json`: original source identities and bundled extract hashes.
- `verify.py` and `validation.json`: numerical and document acceptance checks.
- `review.json`: scientific, visual, and portable-rebuild review record.
- `build-audit/`: bounded final verification record and pinned inputs.

Figures use blue for synchronous training, purple for clipped decentralized
training, and orange for decentralized training without clipping. Batch axes
show only 16, 32, 64, and 128. Distribution figures retain poor configurations;
screening profiles are explicitly labeled as seed-42 diagnostics.

Clipping frequency counts clipped updates, not the magnitude or benefit of the
intervention. Disabled clipping produces zero counters by configuration, not
because gradient norms stayed below one. Peak allocated memory includes final
evaluation at batch 128. Throughput medians describe the recorded run pools and
placements rather than a controlled speed benchmark.

## Visual review

Render either PDF with Poppler, for example:

```bash
pdftoppm -scale-to 1600 -png slides.pdf /tmp/small-batch-slide
pdftoppm -scale-to 1600 -png report.pdf /tmp/small-batch-report
```

The automated checks verify page counts, dimensions, text bounds, numerical
callouts, and evidence consistency. All rendered pages are also reviewed for
overlap, legibility, scientific notation, and figure labels.

Final acceptance records: [numerical and provenance checks](validation.json), [visual and portable-rebuild review](review.json), and [bounded audit](build-audit/manifest.json).
