---
title: Development and publication maintenance
description: Retain source evidence, regenerate the publication, and build the documentation website.
---

## Publication workflow

The current publication is self-contained under `doc/data/current-training/`.
Its 1,782 seed/horizon records include metrics, final results, resolved
configurations, receipts, referenced environment records, campaign definitions,
coverage, audits, and WSD continuation metadata. Checkpoints and environment
installation bundles are not publication inputs.

Retained artifacts are grouped into gzipped JSONL containers, one line per
artifact, so the bundle is roughly ninety files rather than twelve thousand:

```json
{"path":"logs/cosine/sync/h20/lr-0.01-b1-0.9-b2-0.99/s42/metrics.jsonl",
 "bytes":71410,"sha256":"…","text":"<the artifact verbatim>"}
```

Placement is a pure rule of schedule, training mode, and horizon, so a misfiled or
renamed container fails verification. Metrics containers concatenate one gzip member
per line, and `sources.json` records each log's offset, so a single run stays directly
addressable without inflating its container. `sources.json` is the index: it maps every
artifact to its container and keeps the checksum of the original file, while
`checksums.json` covers the containers as stored. Campaign directory names are
provenance; public labels use schedule, training mode, horizon, and hyperparameters.

Import an authoritative source snapshot once:

```bash
uv run python scripts/export_publication.py --import-runs runs
```

The import reads the selected campaign reports and their successful attempts,
checks their identities, and retains the necessary files.

### What is retained, and what is not

Most retained artifacts are byte-identical to the files training wrote. The exception is
the metrics logs of the decentralized runs, which had two members deleted from every
train event: `local_grad_norms` and `local_losses`. These are per-worker diagnostics for
the packed models. No published figure, table, or curve reads them, and they were 37% of
the logged bytes.

The transform is a deletion, not a re-serialization: no other byte of any line, and no
line, was changed, added, or reordered. `sources.json` keeps each log's original digest
but can no longer recompute it from this copy — for the cosine runs that digest stays
independently attested by the retained `receipt.json`, which records the trainer's own
hash and is checked on every verification run. The complete pre-repack bundle, with the
arrays intact, is archived at
[`zesen-kth/tiny-llm`](https://huggingface.co/datasets/zesen-kth/tiny-llm) at revision
`per-file-original`.

A bundle still in the original per-file layout migrates once with `--repack`, which
verifies every artifact against its recorded digest, proves the deletion against the
bytes it replaces, and refuses to publish if any measurement moves.

### Hugging Face mirror

The same bundle is published as a dataset, so regeneration does not require the git
copy and the logs stay available independently of repository history:

```bash
uv run python scripts/export_publication.py --publish-hf
uv run python scripts/export_publication.py --check --bundle hf://zesen-kth/tiny-llm
```

Publishing needs a token with write access. Nothing else does: the website build, the
tests, and CI read only the committed bundle and never reach the network.

Regenerate results, curves, performance, and publication figures from the
retained documentation bundle:

```bash
uv run python scripts/export_publication.py
uv run python scripts/export_publication.py --check
```

No source campaign directories, Slurm commands, checkpoints, or GPUs are needed
for regeneration. Python uses YAML and Matplotlib, without importing the trainer.
To validate another copy or write a separate export, pass `--bundle PATH` and
`--output PATH`. Outputs are validated in a staging directory before replacing
the previous publication. Missing required inputs leave the prior snapshot intact.

The exporter recomputes full-validation rankings and sample SDs, matches
hyperparameters and seeds, reconstructs WSD continuation curves, and calculates
per-run steady throughput. It also updates marked tables in the README, results
article, and performance article. Keep those table markers intact.

## Authoring and preview

Markdown under `doc/` is authoritative. Articles have a title, optional description,
and `archive: true` for archived references. Keep relative Markdown links;
the build translates article, configuration, and data links for the website base.

Use natural scientific labels. Put source path details and campaign identities
in downloadable provenance. Results, Usage, and Performance are the current
navigation; historical reports and original explorers remain accessible in the
archive and retain their URLs. Archived pages and maintenance instructions are
excluded from current search.

The website uses its independent Node 24+ environment:

```bash
npm --prefix site ci
npm --prefix site run build
npm --prefix site run dev
```

Open `http://127.0.0.1:4321/tiny-llm/`. Production builds generate local search;
use `npm --prefix site run preview` to inspect it. Website builds validate
committed publication checksums and statistics using Node, without Python or
campaign access. Generated data, copied assets, and build outputs are ignored by Git.

## Validation

```bash
scripts/test-cpu.sh tests/test_publication_export.py
uv run python scripts/export_publication.py --check
npm --prefix site run check
npm --prefix site test
npm --prefix site run build
cd site
npx playwright install chromium
npm run test:browser
```

Checks cover retained evidence, ranking/seed statistics, continuation boundaries,
throughput windows, missing inputs, internal links, math, filters, URL restoration,
downloads, light/dark themes, mobile layout, and non-JavaScript summaries.

## Development

```bash
uv run pyright
scripts/test-cpu.sh
uv run ruff check .
uv run ruff format --check .
```

The CPU launcher reuses a node-local environment and one PyTorch CPU thread.
GPU tests require a CUDA allocation. Slow compilation/figure checks and tokenizer
multiprocessing checks can be enabled separately:

```bash
scripts/test-cpu.sh --run-slow -m slow
scripts/test-cpu.sh --run-integration -m integration
```

## Refresh recorded curves

The historical cosine exporter is retained for archived studies:

```bash
npm --prefix site run data:refresh:archive -- --runs-root /path/to/runs
```

It refreshes only the older 753-result publication, which retains at least 726
recorded curves. Use the current publication exporter for current experiments.
Archive refresh should use the authoritative source copy to preserve coverage.

## WSD horizon campaigns

The original WSD publication and explorer remain archived. Their committed
dataset, checksum validation, and reconstructed trajectories remain available.
The current publication additionally retains the raw logs, requests, and
continuation records necessary to regenerate these measurements from `doc/`.
Continuations include parent history only through the pre-decay cursor;
longer horizons are not independent fresh training runs.

## GitHub Pages

Pull requests build and validate the site. Successful pushes to `main`
publish through the existing GitHub Pages workflow, with manual dispatch also
available. Deployment uses the `github-pages` environment and the repository's
Pages source must be GitHub Actions. Build artifacts are uploaded rather than
committed to a deployment branch.
