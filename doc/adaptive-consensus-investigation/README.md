# Adaptive Consensus: From Diagnostics to Training Dynamics

[Presentation — 13 slides](slides.pdf) · [Technical supplement](supplement.pdf)

The presentation contains a title and 12 content slides for an ML research audience.
The companion document provides the exact recipes and algorithms, estimator definitions,
complete results, original figures, derivations, and the evidence behind the diagnosis.

The narrative ends at `runs/consensus-scaling-review-2026-09-11`. The fresh-update
proposal is presented at that historical stage. Later files under
`runs/fresh-update-trial-2026-09-11` are outside this presentation's evidence boundary;
the deck does not claim that those later experiments have not happened.

## Build

Requirements: Typst 0.15.1 or later (for native PDF figure embedding), Python with
Matplotlib and NumPy, and Poppler's `pdfinfo` and `pdftotext` for the final checks.
The template import is pinned to `@preview/typslides:1.3.4`; Typst downloads the
package on its first use. The text fonts are Nimbus Sans and Libertinus Serif;
equations use New Computer Modern Math.

From the repository root:

```bash
bash doc/adaptive-consensus-investigation/build.sh
```

This rebuilds all redrawn figures from the included extracted data, compiles both
PDFs, and validates page count, source figures, and numerical callouts. It needs
neither training checkpoints nor the original `runs/` directory. The script selects
the repository's `.venv-x86_64/bin/python` when available, otherwise `python3`.
Set `TINY_LLM_DOC_PYTHON` to use another Python executable.

To refresh extracted inputs from the original experiment folders first:

```bash
bash doc/adaptive-consensus-investigation/build.sh --refresh
```

Refreshing requires the repository's saved runs. It only reads existing results;
it does not run training, similarity analysis, or Hessian-vector products. The
supplement's historical prose and embedded original figures are a fixed record:
review them explicitly if refreshing from changed experiment results.

To compile the editable documents directly, run from this directory:

```bash
typst compile --root . slides.typ slides.pdf
typst compile --root . supplement.typ supplement.pdf
```

## Contents and provenance

- `slides.typ` and `supplement.typ`: editable document sources.
- `build_figures.py` and `build_review_figure.py`: extraction and figure generation; preserve measurements
  without smoothing. Exports PDF and SVG vector assets.
- `data/`: extracted plotting tables, resolved recipes, and computed callouts.
- `provenance.json`: input paths, SHA-256 hashes, and plotting transformations.
- `historical-source-provenance.json`: branch reference and SHA-256 hashes of the
  executed source snapshots used to check the algorithm equations.
- `assets/originals/`: unchanged copies of the six requested source figures;
  `sources.json` records their original paths and hashes.
- `supplement-data.json`: the supplement's saved numerical and source information.
- `verify.py`: document acceptance checks; `validation.json` records the result.
- `review.json`: visual, scientific, and portable-rebuild review record.

Slide footers use source identifiers mapped in the supplement. The initial
plain-versus-adaptive comparison uses exponential topology; the scaling review
uses complete topology. Subset-validation curves and final full-validation values
are labeled separately. Main diagnostic plots use unseen training-cache data;
the supplement retains both seen and unseen series. Its review figure uses the
requested “consensus error” terminology; the unchanged original remains archived.

The moment-scaling formulas are anchored to historical executed source snapshots
and resolved recipes. At document creation, the committed
`feature/adaptive-consensus-moment-scaling` reference points to `26a3bddb`, whose
tree contains ordinary adaptive consensus. The branch name alone is therefore
insufficient provenance for the scaling implementation.

## Visual checks

Render the compiled deck with Poppler for review:

```bash
mkdir -p /tmp/adaptive-consensus-slides
pdftoppm -scale-to 1600 -png slides.pdf /tmp/adaptive-consensus-slides/slide
```

The main deck is 16:9 with white backgrounds, navy text, and consistent method
colors. Figures occupy a 28 cm wide area, with 16 pt axis labels and tick labels.
Five supplementary original figures retain their source styling; the review
figure is redrawn with updated terminology. These figures are enlarged
on A4 pages for inspection.
