# ASR Implementation Plan for mne-denoise

This document is a planning artifact only. It does not implement Artifact
Subspace Reconstruction (ASR), and it should be used as the implementation
contract before adding package code.

All research paths below are under `refs/asr/notes/` unless a full path is
shown. The `.docx` research file was read by extracting `word/document.xml`
from the Office Open XML archive and parsing paragraph text.

## 1. Executive Summary

ASR belongs in `mne-denoise` because it is a denoising method for exactly the
class of non-stationary M/EEG artifacts that DSS and ZapLine do not cover well:
short, high-amplitude, spatially structured bursts from blinks, motion, cable
sway, muscle, and mobile EEG recording conditions. The package already exposes
MNE-compatible, auditable denoisers (`DSS`, `ZapLine`, `ICanClean`), shared MNE
object reconstruction helpers, QC metrics, and method-agnostic plotting. ASR can
fit into that architecture if it is implemented as a scikit-learn-style
estimator with explicit calibration state, logged window decisions, and bounded
memory behavior.

Recommended implementation scope:

| Variant | Status for `mne-denoise` | Rationale |
|---|---:|---|
| Standard Euclidean ASR, clean-room Python core | Phase 1 | Canonical baseline; enough implementation detail exists from `clean_rawdata`, ASRpy, MEEGkit, and the research notes. |
| `clean_windows`-style calibration window selection | Phase 1 | Required for practical offline use and MATLAB parity. |
| Correction mode (ASR-C) | Phase 1 | Main ASR use case: reconstruct affected subspaces while preserving sample count. |
| Rejection masks / ASR-R-style sample masks | Phase 2 | Useful, but should sit on top of a correct reconstruction core. |
| Riemannian ASR / rASR | Phase 2, experimental at first | Strong evidence and Python references exist, but it needs geometry primitives, SPD regularization, and separate validation. |
| Juggler's ASR / ASRDBSCAN / ASRGEV | Research-only until reproducible details are obtained | Promising for extreme MoBI, but public code and full algorithmic details are incomplete in the supplied corpus. |
| Adaptive ASR / AASR / PSW-ASR / PSP-ASR | Phase 3 or experimental | Meaningful for online BCI, but adds dynamic calibration and a different validation burden. |
| Infant/newborn NEAR-style profiles | Phase 3 profile layer | NEAR changes parameter calibration and bad-channel handling, not ASR math. |
| Sleep Dusk2Dawn-style profiles | Phase 3 profile layer | Sleep requires calibration workflow and slow-wave preservation checks, not a new ASR core. |
| Single-channel ASR via decomposition | Do not implement yet | Structurally different from multichannel MNE workflows. |
| IMU-integrated ASR | Do not implement yet | The corpus mostly shows ASR as a comparator beside IMU methods, not a fully specified fused ASR variant. |

Main technical risks:

- Faithful ASR is more than sliding-window PCA. Critical details include the
  spectral pre-emphasis filter, block geometric median covariance, truncated
  generalized Gaussian fit, direction-dependent threshold matrix, exact
  reconstruction matrix, look-ahead state, and raised-cosine blending.
- ASR is sensitive to rank. Average reference before ASR, active projections,
  flat channels, and excessive interpolation can make covariance matrices
  singular or ill-conditioned.
- Low cutoff values can overclean. The corpus contains conflicting default
  recommendations: early/default values around `k=5`, Chang-style adult EEG
  recommendations around `k=20-30`, and workshop/methodological warnings that
  much larger values can be safer for mild cleaning.
- Existing Python ports are useful references but should not be copied blindly.
  ASRpy is MNE-oriented but limited and not heavily tested; MEEGkit is the best
  algorithmic Python reference but is array-oriented and not designed around
  MNE streaming; eegprep exposes MATLAB-like parameters but explicitly has
  incomplete `maxmem` behavior.
- Licensing needs review before porting code. The plan should use a clean-room
  implementation informed by behavior and tests, not copied UC Regents or
  `clean_rawdata` core code.

Main validation strategy:

1. Unit-test each numerical primitive against controlled arrays.
2. Build small MATLAB reference outputs from `clean_rawdata` for calibration,
   processing, and `clean_windows` parity.
3. Add synthetic artifact tests with known ground truth.
4. Validate MNE object round-trips and metadata preservation.
5. Add benchmark scripts and examples for seated, mobile, gait, ERP, and SSVEP
   datasets.
6. Treat rASR and calibration variants as separate benchmark axes, not silent
   replacements for standard ASR.

Expected final deliverables after implementation:

- `mne_denoise.asr.ASR` estimator with standard ASR.
- Internal array core for calibration, processing, window selection, robust
  statistics, and diagnostics.
- Optional experimental `method="riemannian"` backend once Phase 1 is stable.
- MNE Raw support first; NumPy continuous arrays alongside it; Epochs support
  once per-epoch and concatenated semantics are tested; Evoked transform only
  after external calibration, with warnings.
- Tests, parity fixtures, examples, docs, and QC/report helpers.

## 2. Current mne-denoise Architecture Review

### Package layout

Current tracked package layout relevant to ASR:

| Path | Current role | Reuse for ASR |
|---|---|---|
| `mne_denoise/__init__.py` | Exposes package modules `dss`, `zapline`, `icanclean`; version `0.0.1`. | Add `asr` module export after implementation. |
| `mne_denoise/utils.py` | `extract_data_from_mne()` and `reconstruct_mne_object()` for Raw/Epochs/Evoked/arrays. | Reuse, but ASR will need a pick-aware wrapper that only cleans selected channels and preserves non-picked channels. |
| `mne_denoise/dss/linear.py` | `compute_dss()` and `DSS(BaseEstimator, TransformerMixin)`. Handles MNE and NumPy. | Follow estimator style and fitted-attribute conventions. Reuse covariance/whitening ideas only where appropriate. |
| `mne_denoise/dss/utils/covariance.py` | Empirical, shrinkage, OAS, and MCD covariance helpers with weights. | Reuse for optional covariance estimators, but ASR needs its own moving covariance and geometric median behavior. |
| `mne_denoise/dss/utils/whitening.py` | PCA whitener with rank handling and regularization. | Reuse logic/patterns for SPD eigenvalue clipping and tests. |
| `mne_denoise/dss/utils/selection.py` | Iterative outlier component selection for DSS/ZapLine. | Useful for QC thresholds, not a substitute for ASR's `fit_eeg_distribution`. |
| `mne_denoise/zapline/core.py` | `ZapLine(DSS)` estimator with standard/adaptive mode and fitted diagnostics. | Good precedent for a specialized denoiser class with adaptive/standard branches and MNE reconstruction. |
| `mne_denoise/zapline/adaptive.py` | Frequency detection, segmentation, QA functions. | ASR can reuse the idea of chunk/segment diagnostics and spectral QA, but not the line-noise logic. |
| `mne_denoise/icanclean/core.py` | `compute_icanclean()` array core plus `ICanClean` estimator; sliding/global/calibrated/hybrid modes; QC attrs. | Strong precedent for separating array core from estimator, maintaining per-window QC, and handling epochs. |
| `mne_denoise/qa.py` | Estimator-agnostic spectral and variance metrics. | Extend with ASR-specific variance, reconstructed-window, rank, and covariance metrics. |
| `mne_denoise/viz/` | Method-agnostic component, signal, spectra, stats, and summary plotting. | Reuse existing plots: component score curve, window score traces, component patterns, PSD comparison, denoising summary, window count series. Add ASR-specific wrapper panels if needed. |
| `docs/api.rst` | Autosummary API reference for DSS, ZapLine, iCanClean, denoisers, variants, viz. | Add ASR autosummary entries after implementation. |
| `examples/dss/`, `examples/zapline/` | Sphinx-gallery examples and README files. | Add `examples/asr/README.rst` and runnable examples. |
| `tests/` | Pytest unit tests split by feature; `tests/parity/` contains MATLAB parity scaffolding for DSS/ZapLine. | Add `tests/test_asr_*.py`, `tests/asr/`, and `tests/parity/test_asr_parity.py`. |

### Existing estimator and API patterns

- `DSS` and `ICanClean` inherit `sklearn.base.BaseEstimator` and
  `TransformerMixin`.
- `ZapLine` inherits `DSS` but overrides `fit`, `transform`, and
  `fit_transform` for denoising semantics.
- Estimators expose fitted attributes with trailing underscores:
  `filters_`, `patterns_`, `mixing_`, `eigenvalues_`, `explained_variance_`,
  `n_removed_`, `adaptive_results_`, `correlations_`, `n_windows_`,
  `removed_idx_`, and related QC arrays.
- Constructors store parameters as public attributes without doing data work.
- Transform methods generally return the same type as the input when returning
  cleaned data, by using `reconstruct_mne_object()`.

ASR should follow this style:

- Public class: `mne_denoise.asr.ASR`.
- Low-level array functions: `calibrate_asr()` and `process_asr()` or private
  equivalents.
- Fitted attributes should be inspectable and small enough to serialize in
  reports.
- Avoid hidden in-place edits. Use `copy=True` by default and explicit
  `inplace=True` only if implemented safely.

### Existing fit/transform conventions

- `DSS.fit(X)` learns filters and `DSS.transform(X)` returns either sources or
  reconstructed MNE objects depending on `return_type`.
- `ZapLine.fit_transform(X)` supports adaptive mode even when `fit()` is not
  meaningful as a separate step.
- `ICanClean.fit()` is a no-op and `transform()` performs the cleaning because
  its default mode is inherently window-local.

ASR should differ from `ICanClean`: `fit()` should perform calibration and
`transform()` should apply the calibrated model. This is important because ASR
has a real learned state (`M`, `T`, filter coefficients, thresholds, calibration
window mask). `fit_transform(raw)` may use the same Raw both for auto
calibration and cleaning, but the calibration subset must be logged.

### Existing support for MNE objects and arrays

Current support:

- `DSS` accepts Raw, Epochs, Evoked, and NumPy arrays.
- `ICanClean` supports Raw, Epochs, and 2D NumPy arrays.
- `ZapLine` supports Raw/Epochs/Evoked/array paths through inherited and custom
  helpers, but its strongest tested path is continuous arrays/Raw.
- `utils.extract_data_from_mne()` returns data, sampling frequency, type string,
  and original instance. Epochs are returned in MNE orientation
  `(n_epochs, n_channels, n_times)`.

Recommended ASR support:

- Phase 1: Raw and 2D NumPy arrays `(n_channels, n_times)`.
- Phase 1/2: Epochs only when processing each epoch independently or when
  explicitly concatenating epochs for calibration; document which semantics are
  used.
- Phase 2: Evoked transform only after fitting on Raw/Epochs/calibration array.
  Fitting ASR on an Evoked object is not statistically meaningful because ASR
  needs variance distributions over time windows.
- All MNE support must use `picks="eeg"` by default and preserve non-picked
  channels unchanged.

### Existing QC, plot, and report infrastructure

Reusable QC:

- `mne_denoise.qa.variance_removed()`.
- PSD and distortion helpers in `mne_denoise.qa`.
- `mne_denoise.viz.plot_component_score_curve()`.
- `mne_denoise.viz.plot_window_score_traces()`.
- `mne_denoise.viz.plot_component_patterns()`.
- `mne_denoise.viz.plot_psd_comparison()`.
- `mne_denoise.viz.plot_denoising_summary()`.
- `mne_denoise.viz.plot_window_count_series()`.

Needed additions:

- `compute_asr_qa_metrics(raw_before, raw_after, asr)` or equivalent.
- ASR-specific metrics: fraction of reconstructed samples/windows, number of
  affected components per window, effective rank change, covariance distance to
  calibration baseline, per-channel variance ratio, and calibration coverage.
- Report helper in a future module, for example
  `mne_denoise.report.add_asr_qc(report, raw_before, raw_after, asr)`, if a
  report module is introduced.

### Existing tests and documentation style

Testing conventions:

- Pytest with fixtures in `tests/conftest.py` and module-specific fixtures.
- Synthetic-data unit tests with deterministic `np.random.default_rng(42)`.
- Numeric assertions use `numpy.testing.assert_allclose`.
- Visualization tests force the `Agg` backend.
- Parity tests exist under `tests/parity/` with MATLAB reference artifacts.

Documentation conventions:

- Sphinx with `docs/index.rst`, `docs/api.rst`, and Sphinx-gallery examples.
- API docs use numpydoc-style docstrings.
- Examples are Python scripts under `examples/<method>/plot_*.py`.
- Some current docs are stale relative to code (`docs/dss.md` mentions APIs not
  present or signatures that differ). ASR docs should be generated from
  docstrings and examples to avoid drift.

### Where ASR should live

Proposed package paths:

| New path | Purpose |
|---|---|
| `mne_denoise/asr/__init__.py` | Public exports: `ASR`, eventually `calibrate_asr`, `process_asr`, and diagnostics classes if public. |
| `mne_denoise/asr/core.py` | Public estimator class and high-level orchestration. |
| `mne_denoise/asr/_calibration.py` | Clean-window selection, robust covariance, threshold fitting. |
| `mne_denoise/asr/_process.py` | Sliding-window processing, reconstruction matrix generation, blending, chunk state. |
| `mne_denoise/asr/_stats.py` | `fit_eeg_distribution`, geometric median, RMS distributions, robust z-score utilities. |
| `mne_denoise/asr/_filters.py` | Spectral pre-emphasis filter design and stateful filtering. |
| `mne_denoise/asr/_mne.py` | Pick handling, Raw/Epochs array extraction, annotations, reconstruction into original object. |
| `mne_denoise/asr/_diagnostics.py` | Dataclasses for calibration and processing diagnostics. |
| `mne_denoise/asr/_riemann.py` | Phase 2 Riemannian geometry primitives and optional pyRiemann bridge. |
| `tests/asr/` | Focused ASR unit tests. |
| `examples/asr/` | Sphinx-gallery examples. |
| `docs/asr.rst` or `docs/asr.md` | User guide page included from `docs/index.rst`. |

## 3. Research Corpus Inventory

The following ASR research files were read and deduplicated.

| # | File path | Type | Main topic | Key information extracted | Refs | Impl | Data | Confidence |
|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `Artifact Subspace Reconstruction (ASR) for mne-denoise  Algorithm, Variants, and Implementation Evidence Pack.md` | md | Broad evidence pack | Taxonomy, MATLAB/Python implementation map, datasets, roadmap, risks. | Yes | Yes | Yes | High |
| 2 | `Artifact Subspace Reconstruction (ASR) for Mobile EEG (Evidence Pack).docx` | docx | Mobile EEG evidence | Implementation inventory, parameter recommendations, calibration best practices, treadmill defaults, MoBI gaps. | Yes | Yes | Yes | Medium; some claims conflict with other files. |
| 3 | `Audit of MATLAB ASR Implementations for Python MNE Parity.md` | md | MATLAB parity | `clean_rawdata`, `asr_calibrate`, `asr_process`, `clean_windows`, rASRMatlab, AASR, memory, online state. | Yes | Yes | No | High |
| 4 | `Benchmark and QC Framework for ASR in mne-denoise.md` | md | QC/benchmarks | Metric taxonomy, 17 metrics, QC panels, MNE Report proposal, parity thresholds. | Yes | Yes | Yes | High |
| 5 | `compass_artifact_wf-9aed17a1-a067-4da9-9d78-36b970470e16_text_markdown.md` | md | Implementation reconstruction | Canonical standard-ASR algorithm, `fit_eeg_distribution`, geometric median, Yule-Walker filter, reconstruction operator, tests. | Sparse | Yes | No | High |
| 6 | `compass_artifact_wf-b6760f40-9ba1-4b13-8606-e830f2ff79c5_text_markdown.md` | md | Technical evidence pack | Roadmap, Python-port divergence, datasets, numerical risks. | Yes | Yes | Yes | High |
| 7 | `Deep research task__Create an implementation-ready.md` | md | rASR review | AIRM metric, Karcher mean, PGA, pyRiemann vs custom SciPy, regularization, tests, API options. | Yes | Yes | Yes | High |
| 8 | `Deep research task__Propose a Python_MNE-compatibl.md` | md | API architecture | Public estimator designs, diagnostics attributes, annotations, bad channel/rank policy, roadmap. | Yes | Yes | No | High |
| 9 | `deep-research-report (1).md` | md | QC summary | Mandatory metrics, QC figures, implementation parity thresholds. | No | Partial | Partial | Medium |
| 10 | `deep-research-report (2).md` | md | Documentation outline | User guide/API/tutorial warning text, methods paragraph, FAQ, citations, QC plots. | Yes | Partial | No | Medium |
| 11 | `deep-research-report (3).md` | md | Efficiency | Long-recording complexity, memory mapping, streaming pseudocode. | No | Yes | No | Medium |
| 12 | `deep-research-report (4).md` | md | Test plan | Synthetic tests, MATLAB parity protocol, tolerance recommendations, minimal CI suite. | Yes | Yes | Yes | High |
| 13 | `deep-research-report (5).md` | md | Dataset ranking | Dataset categories, ranking, CI tiny-data strategy, MNE-BIDS notes. | Partial | No | Yes | High |
| 14 | `deep-research-report (6).md` | md | Condensed evidence report | Variant taxonomy, foundational algorithm, implementation gaps, roadmap, risks. | Yes | Yes | Yes | Medium; duplicates #1. |
| 15 | `Efficient ASR for Long EEG Recordings in Python.md` | md | Performance | Memory blow-ups, `MaxMem`, streaming covariance, shapes to avoid, efficient pseudocode. | Yes | Yes | No | High |
| 16 | `Implementation-Level Reconstruction of the Artifact Subspace Reconstruction (ASR) Algorithm.md` | md | Standard ASR spec | Calibration state, clean-window selection, covariance, thresholding, reconstruction, online state, pseudocode. | Yes | Yes | No | High |
| 17 | `Open-access datasets for validating ASR on mobile and gait EEG in Python.md` | md | Validation datasets | Detailed dataset table, ranking, MNE-BIDS loading, mini-test strategy. | Yes | No | Yes | High |
| 18 | `Python Implementations of Artifact Subspace Reconstruction (ASR)  Landscape, Failure Modes, and Design Lessons.md` | md | Python ecosystem | ASRpy, MEEGkit, Timeflux rASR, EEG-ASR-Python, eegprep, failure modes, reuse vs rewrite. | Yes | Yes | No | High |
| 19 | `Validation of Artifact Subspace Reconstruction (ASR) for EEG Artifact Removal  Evidence Matrix and Benchmarks.md` | md | Validation literature | Chang, rASR, Juggler, NEAR, gait/running, EEG-cleanse, RELAX, warnings. | Yes | Yes | Yes | High |
| 20 | `Variants of Artifact Subspace Reconstruction (ASR) for EEG Artifact Correction.md` | md | Variant taxonomy | Standard, BCILAB, clean_rawdata, ASRpy, rASR, Juggler, AASR, NEAR, Dusk2Dawn, single-channel, IMU. | Yes | Yes | Yes | High |

Duplicated information merged:

- The standard ASR algorithm appears in files 1, 5, 14, and 16. File 16 and file
  5 provide the most implementation-ready details.
- The Python implementation landscape appears in files 1, 2, 6, and 18. File 18
  is the most complete and resolves several overstatements from file 2.
- Dataset recommendations appear in files 1, 4, 12, 13, and 17. File 17 is the
  authoritative table.
- rASR appears in files 1, 3, 7, 14, 19, and 20. File 7 is the implementation
  review.

Contradictions or uncertainty flagged:

- Default cutoff `k` is inconsistent across sources: API default 5, GUI or
  practical defaults 20, Chang recommendation 20-30, and workshop/perspective
  material warning that mild cleaning may require far higher values. The
  implementation should expose `cutoff` explicitly and avoid calling any value
  universally optimal.
- The `.docx` file states MEEGkit accepts MNE Raw; the more detailed Python
  implementation survey says MEEGkit is array-oriented and MNE integration is
  user-land. Treat MEEGkit as an array reference, not an MNE-native API.
- The `.docx` file states clean_rawdata "integrates rASR" if the Riemannian
  toolbox is in path; other files describe rASRMatlab as a plugin/drop-in
  override. Treat rASR integration as path-dependent and experimental, not a
  stable clean_rawdata default.
- Juggler's ASR public code: a community Python implementation now exists at
  `https://github.com/thiagorroque/asrpy` — a fork of `DiGyt/asrpy` adding
  ASRDBSCAN and ASRGEV per Kim et al. 2025. Marked WIP/under test by the
  author. The previously cited `https://code.ornl.gov/fub/juggler` URL is a
  different unrelated CUDA GPU project, not Juggler's ASR; ignore it. The
  ASRpy issue thread (`DiGyt/asrpy#15`) points to the thiagorroque fork as
  the reference implementation.
- IMU-integrated ASR is not clearly specified in the corpus. Some sources
  mention IMU-informed or IMU-enhanced artifact removal, but ASR is usually a
  baseline, not the fused algorithm.

## 4. Complete Reference Inventory

This inventory deduplicates repeated mirror URLs and incomplete search-result
snippets by work. The "status" column indicates whether the reference is
essential, useful, optional, or not recommended for implementing ASR in
`mne-denoise`. "Source files" names the research files that mention the item.

### 4.1 Core ASR Papers

| Reference | Authors/year in corpus | URL/DOI if present | Why it matters | Status | Source files |
|---|---|---|---|---|---|
| The Artifact Subspace Reconstruction Method, SCCN slide deck | Kothe / SCCN; year incomplete | `https://sccn.ucsd.edu/githubwiki/files/asr-final-export.pdf` | Most explicit practical ASR description: high-pass/full-rank preconditions, robust covariance, thresholds, reconstruction, online state. | Essential | Files 1, 3, 4, 5, 6, 7, 15, 16, 18, 19, 20 |
| Real-time neuroimaging and cognitive monitoring using wearable dry EEG | Mullen et al., 2015 | `https://pmc.ncbi.nlm.nih.gov/articles/PMC4710679/` | Demonstrates online ASR in wearable dry EEG and BCI context. | Essential | File 20; also summarized in files 1, 2 |
| BCILAB: a platform for brain-computer interface development | Kothe and Makeig, 2013 | `https://pubmed.ncbi.nlm.nih.gov/23985960/` | Historical BCILAB context and online BCI platform where ASR originated. | Essential | Files 3, 20 |
| Artifact Subspace Reconstruction patent | Kothe and Jung, 2016 in corpus | Patent `US20160113587A1`; direct URL not fully provided | Defines original bad-subspace removal claims and reinforces licensing/patent caution. | Useful | Files 2, 20 |
| Evaluation of Artifact Subspace Reconstruction for Automatic Artifact Components Removal in Multi-Channel EEG Recordings | Chang, Hsu, Pion-Tonachini, Jung; 2018/2019/2020 depending source | `https://pubmed.ncbi.nlm.nih.gov/31329105/`; also Semantic Scholar | Core parameter validation. Shows k around 20-30 is a safer adult EEG region than low k. | Essential | Files 1, 2, 4, 18, 19, 20 |
| EEGLAB Workshop 2021 ASR Performance Evaluation II | SCCN workshop; 2021 | `https://sccn.ucsd.edu/githubwiki/files/EEGLAB_Workshop2021%20ASR%20Performance%20Evaluation%20II.pdf` | Warns about overcleaning and high sensitivity of ASR to cutoff and calibration. | Useful | File 19 |
| Makoto's preprocessing pipeline | Miyakoshi / SCCN wiki; ongoing | `https://eeglab.ucsd.edu/wiki/Makoto's_preprocessing_pipeline` | Practical pipeline context and Juggler/Miyakoshi perspective. | Useful | Files 1, 2 |
| EEGLAB clean_rawdata documentation | SCCN/EEGLAB docs | `https://eeglab.org/plugins/clean_rawdata/Documentation.html`; `https://eeglab.org/plugins/clean_rawdata/`; `https://eeglab.org/tutorials/06_RejectArtifacts/cleanrawdata.html` | Public parameter behavior, defaults, pipeline order, warnings. | Essential | Files 1, 3, 5, 15, 16, 18, 20 |
| How Sensitive Are EEG Results to Preprocessing Methods | Authors incomplete in corpus; 2020 | `https://pubmed.ncbi.nlm.nih.gov/32217478/`; bioRxiv PDF also listed | Broad preprocessing sensitivity context. | Optional | File 4 |
| Pedroni et al. automated EEG artifact rejection / Braindance workshop | Pedroni et al.; 2019 in corpus | Direct URL incomplete | Best-practice context for combining ASR with other methods. | Optional | File 2 |

### 4.2 Variant Papers

| Reference | Authors/year in corpus | URL/DOI if present | Why it matters | Status | Source files |
|---|---|---|---|---|---|
| A Riemannian Modification of Artifact Subspace Reconstruction for EEG Artifact Handling | Blum, Jacobsen, Bleichner, Debener; 2019 | `https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2019.00141/full`; DOI `10.3389/fnhum.2019.00141`; PubMed `31105543`; PMC `PMC6499032` | Defines rASR, AIRM/Karcher/PGA motivation, validation metrics, mobile EEG benchmark. | Essential for Phase 2 | Files 1, 2, 3, 4, 7, 15, 16, 17, 18, 19, 20 |
| Evaluation of Riemannian ASR on cEEGrid data | Authors incomplete; 2018 abstract | `https://www.frontiersin.org/10.3389%2Fconf.fnhum.2018.227.00134/event_abstract`; ResearchGate link in docx | Additional rASR/cEEGrid evidence. | Optional | Files 2, 4, 18, 20 |
| Juggler's ASR: Unpacking the principles of artifact subspace reconstruction for revision toward extreme MoBI | Kim et al.; 2025 (J Neurosci Methods) | `https://pubmed.ncbi.nlm.nih.gov/40324599/`; ScienceDirect PII `S0165027025001062` | Defines ASRDBSCAN and ASRGEV calibration-selection variants for extreme MoBI. | Phase 3; reference Python implementation now exists (see thiagorroque/asrpy in §4.4) | Files 1, 2, 3, 4, 8, 19, 20 |
| Adaptive Artifact Subspace Reconstruction / Development of an Adaptive ASR Based on Hebbian/Anti-Hebbian Learning Networks | Tsai et al.; EMBC 2022 / journal 2023 | `https://pubmed.ncbi.nlm.nih.gov/35714085/`; NYCU pages | Defines PSP-ASR and PSW-ASR adaptive subspace learning for BCI. | Phase 3 | Files 3, 20 |
| NEAR: An artifact removal pipeline for human newborn EEG data | Kumaravel et al.; 2022 | `https://pmc.ncbi.nlm.nih.gov/articles/PMC8800139/`; `https://pubmed.ncbi.nlm.nih.gov/35085870/` | Newborn/infant ASR parameter calibration and LOF bad-channel strategy. | Useful profile layer | Files 1, 19, 20 |
| Dusk2Dawn: EEGLAB plugin for automatic cleaning of whole-night sleep EEG | Somervail et al.; 2023 | `https://pubmed.ncbi.nlm.nih.gov/37542730/`; `https://github.com/rsomervail/dusk2dawn`; Semantic Scholar link | Shows naive ASR can remove sleep slow waves; motivates sleep-specific calibration/profile. | Useful profile layer | File 20 |
| Adapting Artifact Subspace Reconstruction Method for Single-Channel EEG using Signal Decomposition Techniques | Kaongoen and Jo; 2023 | `https://pubmed.ncbi.nlm.nih.gov/38083141/`; KAIST page | Pseudo-multichannel decomposition approach for single-channel EEG. | Do not implement yet | File 20 |
| Dynamical Embedding of Single-Channel EEG for ASR | Authors incomplete; 2024 | `https://pmc.ncbi.nlm.nih.gov/articles/PMC11510769/`; PubMed `39460214`; arXiv `2407.04727` | Another single-channel/E-ASR style approach. | Do not implement yet | File 1 |
| Movement Artifact Suppression in Wearable Low-Density and Dry EEG | Authors incomplete; 2023 | `https://pubmed.ncbi.nlm.nih.gov/37751338/` | Shows limitations of ASR with low-density dry systems. | Optional | File 20 |
| Mobile EEG artifact correction on limited hardware using ASR | Maanen et al.; 2022 | `https://arxiv.org/abs/2204.13444`; ar5iv mirror | Embedded/mobile hardware ASR and memory/performance context. | Useful for online/embedded design | Files 1, 2, 4, 17 |
| Artifact Subspace Reconstruction for EEG artifact removal must be optimized for each unique dataset | Bloniasz; 2022 in corpus | `https://www.qeios.com/read/SMEI50/pdf` | Argues k must be optimized per dataset; reports very high optimal k in some cases. | Useful caution | File 2 |
| The influence of motor tasks and cut-off parameter selection on ASR in EEG | Anders et al.; 2020 in corpus | `https://pmc.ncbi.nlm.nih.gov/articles/PMC7560919/` | Warns low k can distort motor/mobile EEG. | Useful caution | Files 2, 19 |
| Optimizing EEG ICA decomposition with data cleaning in stationary and mobile experiments | Dias et al.; 2023 | `https://www.nature.com/articles/s41598-024-64919-3` in docx, year/title tension in corpus | Shows modality-specific thresholds and baseline requirements; supports ASR before ICA. | Useful | File 2 |
| A Comparison of Approaches for Motion Artifact Removal from EEG During Running | Authors incomplete; 2025 | `https://pmc.ncbi.nlm.nih.gov/articles/PMC12349273/` | Compares ASR, iCanClean, ICA in running; iCanClean often stronger but ASR useful. | Useful benchmark | Files 4, 19 |
| Shredding artifacts: extracting brain activity in EEG from extreme motion during skateboarding | Authors incomplete; 2024 | `https://www.frontiersin.org/journals/neuroergonomics/articles/10.3389/fnrgo.2024.1358660/full` | ASR+ICA in extreme motion, task-level decoding metric. | Optional benchmark | File 19 |
| EEG-cleanse automated pipeline | Authors incomplete; 2025 | PubMed `41323113`; PMC `PMC12664388`; ScienceDirect `S2215016125005461` | Full-body movement ASR-centered pipeline; logging reconstruction fractions. | Useful benchmark | File 19 |
| RELAX automated preprocessing pipeline | Authors incomplete; 2022 | ANU portal; bioRxiv `10.1101/2022.03.08.483548` | ASR as part of a broader automated pipeline. | Optional | File 19 |
| A walk in the park? Characterizing gait-related artifacts in mobile EEG | Authors incomplete; 2020 | `https://onlinelibrary.wiley.com/doi/10.1111/ejn.14965`; PubMed `32909315` | Gait artifact caution and benchmarking context. | Useful benchmark | Files 4, 19 |
| Isolating gait-related movement artifacts in EEG | Authors incomplete; 2016 | `https://pmc.ncbi.nlm.nih.gov/articles/PMC4946867/` | Phantom/isolated gait artifact context. | Useful benchmark | Files 4, 19 |
| Independent Component Analysis of Gait-Related Movement Artifact | Authors incomplete | `https://pmc.ncbi.nlm.nih.gov/articles/PMC4664645/` | ICA gait artifact comparison context. | Optional | File 4 |
| Faster gait speeds reduce alpha and beta EEG spectral power | Authors incomplete | IEEE PDF URL in file 19 | Gait-speed spectral context. | Optional | File 19 |
| Visual Evoked Responses During Standing and Walking | Authors incomplete; 2010 | Frontiers URL | ERP preservation in mobile/walking context. | Optional | File 1 |
| IMU-Enhanced EEG Motion Artifact Removal with Fine-Tuned Large Brain Models | Authors incomplete; arXiv 2025 | `https://arxiv.org/abs/2509.01073`; HTML mirrors | ASR+ICA appears as comparator; not an ASR variant. | Do not implement as ASR | File 20 |
| IMU-integrated ASR for wearable EEG | Kumaravel and Farella; 2023 in docx | Direct source incomplete | Mentioned as a future direction; enough detail not present. | Incomplete, research-only | File 2 |
| Eye-movement artifact correction in infant EEG | Authors incomplete; 2025 | `https://pmc.ncbi.nlm.nih.gov/articles/PMC12753043/` | Notes NEAR/ASR value for infant variable artifacts. | Optional | File 20 |

### 4.3 MATLAB Code References

| Reference | URL/identifier | Why it matters | Status | Source files |
|---|---|---|---|---|
| `sccn/clean_rawdata` repository | `https://github.com/sccn/clean_rawdata` | Canonical EEGLAB ASR implementation and wrapper pipeline. | Essential reference, not source to copy | Files 2, 3, 16, 18 |
| `clean_artifacts.m` | In `sccn/clean_rawdata` | Pipeline order: flatlines, drifts, channels, ASR, windows; parameter defaults. | Essential behavior map | Files 3, 16 |
| `clean_asr.m` | `https://github.com/sccn/clean_rawdata/blob/master/clean_asr.m` | Wrapper around calibration and processing; auto calibration selection. | Essential behavior map | Files 3, 16 |
| `asr_calibrate.m` / `asr_calibrate_r.m` | `https://github.com/sccn/clean_rawdata/blob/master/asr_calibrate.m`; `_r` URL in file 3 | Calibration state, robust covariance, thresholds, spectral shaping. | Essential behavior map | Files 3, 16 |
| `asr_process.m` | `https://github.com/sccn/clean_rawdata/blob/master/asr_process.m` | Core streaming ASR: moving covariance, threshold, `R`, blending, carry state, `maxmem`. | Essential behavior map | Files 1, 3, 15, 16 |
| `clean_windows.m` | In `sccn/clean_rawdata` | Final bad-window rejection and reference-window selection behavior. | Essential for calibration selection | Files 1, 3, 5, 16 |
| `clean_channels.m` / `clean_channels_nolocs.m` | In `sccn/clean_rawdata` | Pre-ASR bad-channel assumptions and channel-rejection strategy. | Useful, not Phase 1 ASR core | Files 3, 16 |
| `clean_flatlines.m` / `clean_drifts.m` | In `sccn/clean_rawdata` | Preconditions: no flat channels, high-pass/zero-mean data. | Useful | Files 3, 16 |
| `fit_eeg_distribution` | Helper in ASR code | Truncated generalized Gaussian fit used in `clean_windows` and ASR calibration. | Essential to reimplement | Files 5, 16, 18 |
| `block_geometric_median` / `geometric_median` | Helper in ASR code | Robust covariance aggregation using Weiszfeld-style L1 median. | Essential to reimplement | Files 5, 16 |
| `moving_average` | Helper inside `asr_process.m` | Stateful moving average of vectorized outer products. | Essential to reimplement | Files 5, 15, 16 |
| `yulewalk` / precomputed filter coefficients | MATLAB Signal Processing Toolbox/helper behavior | Statistics-only spectral pre-emphasis filter. | Essential or compatibility option | Files 5, 16 |
| `hlp_varargin2struct`, `hlp_memfree`, `hlp_microcache` | BCILAB helpers | Parameter parsing, memory heuristics, memoization in MATLAB. | Translate behavior, not code | Files 3 |
| `rASRMatlab` | `https://github.com/s4rify/rASRMatlab` | MATLAB rASR plugin; Manopt dependency; drop-in clean_rawdata path behavior. | Phase 2 reference | Files 2, 3, 7 |
| Manopt | Mentioned as rASR dependency | MATLAB manifold optimization used by rASRMatlab. | Optional context | Files 3, 7 |
| AASR repository | `https://github.com/t5i0m7/AASR` | Adaptive ASR/PSW-ASR MATLAB implementation using `update()` and `reconstruct()`. | Phase 3 reference | Files 3, 20 |
| Dusk2Dawn | `https://github.com/rsomervail/dusk2dawn` | Sleep-specific ASR wrapper. | Profile-layer reference | File 20 |
| NEAR scripts | GitHub repository mentioned but exact URL incomplete in corpus | Newborn/infant pipeline and parameter calibration. | Useful but incomplete URL | Files 1, 19, 20 |
| ORNL `juggler` GitLab (misattribution) | `https://code.ornl.gov/fub/juggler` | The URL was previously suspected to host Juggler's ASR. Verified 2026-05-06 to be an unrelated CUDA GPU runtime project, not Juggler's ASR. Not relevant. | Not relevant — different project | File 3 |
| mASR / Maanen limited-hardware implementation | GitHub/Zenodo mentioned, exact repo incomplete | Embedded C++/MATLAB ASR. | Optional online reference | File 2 |

### 4.4 Python Code References

| Reference | URL | Why it matters | Status | Source files |
|---|---|---|---|---|
| ASRpy (`DiGyt/asrpy`) | `https://github.com/DiGyt/asrpy`; docs `https://digyt.github.io/asrpy/`; PyPI `asrpy-eh` | MNE Raw-oriented standard ASR API reference. | Reference only; do not copy blindly | Files 1, 2, 8, 15, 18, 20 |
| ASRpy example gist/blog/issues | `https://gist.github.com/DiGyt`; `https://digyt.github.io/introducing_ASRpy/`; `https://github.com/DiGyt/asrpy/issues` | Shows usage and open issues, including planned rASR/Juggler. | Useful | Files 1, 2, 18, 20 |
| `thiagorroque/asrpy` (Juggler fork) | `https://github.com/thiagorroque/asrpy` | Community fork of ASRpy implementing Juggler's ASR (ASRDBSCAN + ASRGEV) per Kim et al. 2025. Marked WIP/under test. Cross-referenced from `DiGyt/asrpy#15`. | Phase 3 Juggler reference implementation | This session (2026-05-06) |
| MEEGkit ASR | `https://github.com/nbara/python-meegkit`; docs `https://nbara.github.io/python-meegkit/modules/meegkit.asr.html`; example page | Best Python algorithmic reference for standard ASR and rASR; array-oriented. | Primary Python reference | Files 1, 2, 15, 16, 18 |
| MEEGkit conda package | `https://anaconda.org/conda-forge/meegkit` | Install/reference context. | Optional | File 2 |
| MEEGkit issue tracker snapshot | `https://issues.ecosyste.ms/hosts/GitHub/repositories/nbara%2Fpython-meegkit/issues?label=bug` | Documents ASR edge cases: covariance SPD failures, transform state bugs. | Useful caution | File 18 |
| Timeflux rASR | `https://github.com/timeflux/timeflux_rasr`; PyPI `https://pypi.org/project/timeflux-rasr/`; source docs | Streaming rASR implementation using pyRiemann and sklearn style. | Phase 2/online reference | Files 2, 18 |
| EEG-ASR-Python / pyASR | `https://github.com/moeinrazavi/EEG-ASR-Python` | Standalone research code requiring patched pymanopt/TensorFlow. | Not recommended for reuse | Files 1, 18 |
| `eegprep.clean_asr` | `https://eegprep.org/api/generated/eegprep.clean_asr.html`; API index | MATLAB-like Python port; warns `maxmem` chunking not implemented and rASR flag not implemented. | Useful caution | File 18 |
| pyRiemann | Mentioned; direct docs not consistently listed | Possible rASR backend for AIRM means/distances/tangent space. | Optional dependency candidate | Files 7, 18 |
| geomstats | Mentioned in rASR review | Alternative SPD manifold library. | Optional research dependency | File 7 |
| pymanopt | `https://pymanopt.org/docs/stable/_modules/pymanopt/optimizers/trust_regions.html` | Manifold optimization context; not needed for Phase 1. | Avoid as required dependency | File 18 |
| MNE issue `#7479` / `#9302` and MNE forum ASR threads | `https://github.com/mne-tools/mne-python/issues/7479`; `https://mne.discourse.group/t/asr-in-python-for-eeg/6295`; walking-artifact forum link | Shows MNE ecosystem position: ASR remains external, users use ASRpy/MEEGkit. | Useful API context | Files 2, 18 |
| Neuropype neural docs | `https://www.neuropype.io/docs/nodes/neural.html` | Mentions ASR parameters and min clean fraction. | Optional | File 2 |
| Automated EEG cleaning comparison | `https://digyt.github.io/automated_EEG_cleaning_comparison/`; Colab link | Broader cleaning comparison. | Optional | Files 2, 18 |
| MNE SSP tutorial | `https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html` | Comparator artifact-cleaning method. | Optional | File 18 |
| memray docs | `https://bloomberg.github.io/memray/python_allocators.html` | Suggested memory profiling tool. | Optional engineering reference | File 18 |
| `pyasr` ancestral sequence reconstruction | `https://pypi.org/project/pyasr/`; `https://github.com/Zsailer/pyasr` | Name collision with unrelated package. | Not recommended; document only to avoid confusion | File 18 |
| SpeechBrain / NVIDIA Riva ASR pages | Speech recognition links in file 18 | Unrelated automatic speech recognition search hits. | Not relevant | File 18 |

### 4.5 Dataset References

| Dataset/resource | Authors/year in corpus | URL/DOI if present | Why it matters | Status | Source files |
|---|---|---|---|---|---|
| SRM Resting-state EEG | Authors incomplete; BIDS resting EEG | OpenNeuro `ds003775`; `10.18112/openneuro.ds003775.v1.0.0`; PMC/PubMed/ScienceDirect links | Clean/seated baseline sanity checks and overcleaning tests. | Essential CI/full benchmark | File 17 |
| BCIT Baseline Driving | Authors incomplete; driving BCI | OpenNeuro `ds004120`; EEGDash; `10.18112/openneuro.ds004120.v1.0.0` | Simulated driving/time-on-task data related to Chang-style validation. | Useful full benchmark | File 17 |
| Electrode Walking Study / Mobile EEG walking oddball | Scanlon et al. implied; 2023 data descriptor | OpenNeuro `ds004033`; `10.18112/openneuro.ds004033.v1.0.0`; PMC `PMC9852920`; PubMed `36687153` | Best BIDS outdoor walking dataset with oddball ERP and step events. | Essential mobile benchmark | Files 1, 17, 20 |
| Treadmill BCI MoBI dataset | He et al.; 2018 | Nature Scientific Data `10.1038/sdata.2018.74`; PMC `PMC5914288`; PubMed `29688217` | Classic MoBI treadmill walking with EEG and joint angles. | Useful benchmark | Files 1, 17, 19 |
| Mobile brain-body imaging indoor treadmill/outdoor visual search | Authors incomplete; 2024 | IEEE Dataport DOI `10.21227/H24T0V`; Figshare DOI `10.6084/m9.figshare.6741734`; PMC `PMC11470569`; PubMed `39398473` | Large high-density indoor/outdoor MoBI with IMU/EMG/forces. | Full benchmark, not CI | File 17 |
| Mind in Motion Young Adults Walking Over Uneven Terrain | Mind in Motion group | OpenNeuro `ds004625`; versions `1.0.0`, `1.0.2`; uneven terrain PMC `PMC9762558` | High-density uneven terrain treadmill BIDS/Motion-BIDS stress test. | Essential full benchmark; possible CI slice | File 17 |
| Mind in Motion Older Adults Walking Over Uneven Terrain | Mind in Motion group | OpenNeuro `ds006095`; HuggingFace EEGDash mirror; `10.18112/openneuro.ds006095.v1.0.0` | Large older-adult population, uneven terrain, high channel count. | Full benchmark | Files 1, 17 |
| PhysioNet multimodal gait dataset | Authors incomplete | `https://physionet.org/content/multimodal-gait-dataset/1.0.0/` | EEG+EMG+IMU+forces for cross-modal gait artifact metrics. | Useful benchmark | File 17 |
| Mobile BCI ERP/SSVEP during walking/running | Lee et al. implied; 2021 | OSF DOI `10.17605/OSF.IO/R7S9B`; Nature Scientific Data `s41597-021-01094-4`; GitHub `youngeun1209/MobileBCI_Data` | ERP and SSVEP preservation under speeds 0-2 m/s. | Essential benchmark | Files 2, 17 |
| Mobile Brain-Body Imaging dual-tasking Go/NoGo | Authors incomplete | Dryad `10.5061/dryad.mgqnk9947` and `10.5061/dryad.xsj3tx9nb` | Cognitive ERPs during sitting/walking in young/older adults. | Useful full benchmark | File 17 |
| Motion-BIDS | Authors incomplete; 2024 | PMC `PMC11219788`; Nature Scientific Data `s41597-024-03559-8` | BIDS extension for synchronized motion data. | Useful for data loading | File 17 |
| NEMAR / ESS | NEMAR paper; 2022 | PubMed `36367313`; ESS described in dataset file | Dataset discovery and ESS import/export path for non-BIDS MoBI. | Useful infrastructure | Files 1, 17 |
| Original rASR smartphone EEG | Blum et al.; 2019 | No open dataset URL in corpus | Reference validation conditions: 27 adults, indoor/outdoor, VEP/blinks. | Useful target, incomplete public access | Files 7, 19 |
| Chang simulated driving EEG | Chang et al.; 2020 | Paper URL only; raw repository not linked | Core k-sweep validation target. | Useful target, incomplete public access | Files 1, 19 |
| NEAR newborn/infant datasets | Kumaravel et al.; 2022 | NEAR paper/repository mentioned; exact data URL incomplete | Infant/newborn simulation and effect recovery. | Phase 3 profile benchmark, incomplete | Files 1, 19, 20 |
| Shredding artifacts skateboard EEG | Authors incomplete; 2024 | Frontiers article URL | Extreme motion ASR+ICA task metric. | Optional, likely not CI | File 19 |
| EEG-cleanse 305-session full-body movement data | Authors incomplete; 2025 | PubMed/PMC/ScienceDirect links | Large full-body motion pipeline benchmark and reconstruction fraction target. | Optional full benchmark | File 19 |
| Dry/low-density EEG datasets for single-channel ASR | Kaongoen/Jo and low-density paper | Paper URLs only | Single-channel/low-density scope. | Do not target Phase 1 | File 20 |
| OpenNeuro / EEGDash / HuggingFace mirrors | Various | OpenNeuro search, EEGDash table, HuggingFace ds004033/ds006095 | Dataset access and metadata mirrors. | Useful infrastructure | File 17 |
| Brainlife ds004033 mirror | Brainlife OpenNeuro URL | Alternate access to Electrode Walking Study. | Optional | File 17 |
| Deep BCI Open DB | `http://deepbci.korea.ac.kr/opensource/opendb/` | Alternate access for Mobile BCI dataset. | Optional | File 17 |

## 5. ASR Variant Taxonomy

| Variant | Main paper/source | Main algorithmic difference | Required inputs | Required calibration | Key hyperparameters | Cost | Evidence quality | Existing implementation | mne-denoise status |
|---|---|---|---|---|---|---|---|---|---|
| Standard ASR | Kothe slides, Mullen 2015, clean_rawdata, Chang validation | Euclidean covariance/PCA; robust calibration; per-direction thresholds; reconstruct bad local PCs. | Multichannel EEG, continuous, high-pass, full rank. | Clean baseline or auto-selected clean windows. | `cutoff`, window length, overlap/stepsize, `max_dims`, `min_clean_fraction`, `max_dropout_fraction`, filter, `max_mem`. | O(windows * C^3) plus covariance O(C^2 T). | High. | clean_rawdata, ASRpy, MEEGkit, eegprep. | Phase 1. |
| Offline ASR | clean_rawdata / ASRpy | Uses recorded data to select calibration and process entire recording; may add final window rejection. | Raw/array continuous; annotations useful. | Explicit baseline or selected windows from same recording. | Same as standard plus calibration strategy and annotations. | High unless chunked. | High. | clean_rawdata, ASRpy. | Phase 1. |
| Online ASR | BCILAB / asr_process | Same core but stateful chunks with lookahead/carry/IIR/cov/last_R. | Streaming chunks same montage/sfreq. | Initial clean baseline, possibly adaptive refresh. | `lookahead`, `stepsize`, `max_mem`, state. | Bounded per chunk; latency = lookahead. | Medium/high. | BCILAB, clean_rawdata core, Timeflux rASR for rASR. | Phase 2 after offline parity. |
| EEGLAB clean_rawdata ASR | clean_rawdata docs/code | Full pipeline includes flatline/drift/channel cleaning, ASR-C/ASR-R, final `clean_windows`. | EEGLAB EEG dataset. | Auto or explicit. | Many wrapper criteria. | High but memory controlled by `MaxMem`. | High. | MATLAB. | Match ASR core and document pipeline differences; do not replicate all channel-cleaning in Phase 1. |
| BCILAB ASR | BCILAB paper / online framework | Online BCI integration around ASR state and LSL-style streaming. | Streaming EEG. | Initial artifact-free sample. | Similar core plus online framework params. | Online bounded. | Medium. | MATLAB BCILAB. | Design-compatible, not Phase 1. |
| rASR / Riemannian ASR | Blum et al. 2019 | Covariance processing on SPD manifold; AIRM, Karcher mean, PGA/tangent-space operations. | Multichannel EEG; SPD covariances after regularization. | Similar clean baseline; robust to some covariance outliers. | Metric, mean iterations, covariance regularization, segment length, cutoff. | Per SPD operation O(C^3); fewer decompositions may be faster. | Good single major validation, fewer replications. | rASRMatlab, MEEGkit, Timeflux rASR. | Phase 2 experimental via `ASR(method="riemannian")`. |
| Adaptive ASR / AASR / PSW-ASR / PSP-ASR | Tsai et al.; AASR repo | Updates subspace/thresholds online via Hebbian/anti-Hebbian learning; PSW whitening. | Streaming or segmented BCI EEG. | Initial calibration plus adaptive updates. | Adapt rate, cutoff range, update window. | Online iterative; validation needed. | Medium for BCI, limited general EEG. | MATLAB AASR. | Phase 3 / experimental. |
| Juggler's ASR | Kim et al. 2025 | Improved calibration-data selection for extreme motion; core ASR unchanged. | High-motion MoBI, high-density EEG. | DBSCAN/GEV-selected clean points/windows. | DBSCAN eps/min samples; GEV tail; amplitude features. | Added clustering/fit cost before ASR. | Promising but narrow and validation evidence is single-paper. | `thiagorroque/asrpy` Python fork (WIP); paywalled paper provides spec. | Phase 3 — calibration-strategy plugin point on existing ASR. |
| ASRDBSCAN | Juggler's ASR | DBSCAN on amplitude-derived features to identify clean calibration cluster. | Same as Juggler. | Cluster-derived calibration segments. | Feature scaling, `eps`, `min_samples`, metric. | Moderate. | Spec in Kim et al. 2025; reference impl in `thiagorroque/asrpy`. | `thiagorroque/asrpy` (WIP). | Phase 3. |
| ASRGEV | Juggler's ASR | Generalized Extreme Value model for amplitude outliers used to select calibration data. | Same as Juggler. | GEV-thresholded calibration segments. | Tail fraction/threshold, fit method. | Moderate. | Spec in Kim et al. 2025; reference impl in `thiagorroque/asrpy`. | `thiagorroque/asrpy` (WIP). | Phase 3. |
| ASR-C vs ASR-R | NEAR/clean_rawdata | Correction reconstructs data; rejection removes modified windows/samples. | Continuous EEG. | Standard. | Rejection flag, sample mask thresholds. | Rejection wrapper cost low. | High as behavior mode; parameter effects vary. | clean_rawdata, NEAR. | ASR-C Phase 1; ASR-R Phase 2. |
| NEAR-style infant/newborn ASR | Kumaravel et al. 2022 | Parameter calibration and LOF bad-channel detection for infant EEG; ASR math unchanged. | Newborn/infant EEG, often short/noisy. | Population-specific simulation/real calibration. | k/mode sweep, LOF parameters. | Wrapper/benchmark cost high. | Good for domain. | MATLAB NEAR. | Phase 3 profile/docs, not core variant. |
| Dusk2Dawn sleep ASR | Somervail et al. 2023 | Sleep-specific workflow to avoid slow-wave loss; ASR math unchanged. | Whole-night sleep EEG. | Sleep-stage/epoch-aware calibration. | Sleep-specific thresholds and segmentation. | Long-recording chunking critical. | Medium. | MATLAB plugin. | Phase 3 profile/docs. |
| Single-channel ASR via decomposition | Kaongoen & Jo 2023; E-ASR/dynamical embedding | Creates pseudo-channels from EEMD/WT/SSA or embedding, then applies ASR. | Single EEG channel. | Decomposition-specific. | Decomposition parameters plus ASR. | Potentially high. | Early. | Research code unclear. | Do not implement yet. |
| IMU-assisted ASR | IMU-enhanced papers/docx notes | Not clearly specified as ASR; often ASR+ICA baseline or adaptive filter beside ASR. | EEG plus IMU/motion. | Motion-context calibration. | Motion thresholds/coupling. | Unknown. | Insufficient. | No reproducible ASR fusion found. | Do not implement yet. |
| Mobile/walking-specific ASR adaptations | MoBI studies, running comparison, EEG-cleanse | Mostly parameter choices, baseline selection, ASR before ICA, motion/gait metrics. | Walking/running EEG, possibly IMU/EMG. | Standing/eyes-open baseline or robust auto selection. | Conservative k, gait-aware QC. | Dataset-dependent. | Good applied evidence. | Pipelines, not standalone variants. | Examples and benchmark profiles. |

Recommended implementation order:

1. Standard ASR array core and `ASR(method="standard")`.
2. MNE Raw wrapper with picks, annotations, copy behavior, and diagnostics.
3. `clean_windows`-compatible auto calibration.
4. Chunked transform and parity tests vs MATLAB.
5. NumPy/Epochs support and ASR-R masks.
6. rASR backend behind `method="riemannian"` with experimental label.
7. Calibration strategy extension points for Juggler/NEAR/sleep profiles.
8. Adaptive/online API once static/chunked state is validated.

## 6. Standard ASR Algorithm Specification

### Inputs and shapes

Use one canonical low-level shape:

- `X`: `float64 ndarray`, shape `(n_channels, n_times)`.
- `sfreq`: sampling frequency in Hz.
- Calibration data `X_cal`: same shape convention, same channel order, same
  reference, same filtering, same units as processing data.
- MNE Raw: extract selected channels into `(n_picks, n_times)`.
- MNE Epochs: internally choose either `(n_epochs, n_channels, n_times)` with
  independent per-epoch processing or concatenate epochs into
  `(n_channels, n_epochs * n_times)` for calibration. This choice must be
  explicit and tested.
- MNE Evoked: do not fit. Transform only if fitted elsewhere and if user
  explicitly accepts that ASR statistics were not estimated from the Evoked.

### MNE object handling

- Default `picks="eeg"`; reject empty picks.
- Exclude `info["bads"]` from ASR by default unless `include_bads=True` is
  added later.
- Preserve all non-picked channels unchanged.
- Preserve `info`, `ch_names`, `sfreq`, measurement date, annotations, events,
  event_id, metadata where applicable.
- Warn if active projections or average reference may reduce rank. Do not apply
  projectors automatically unless the user explicitly requested that in their
  data pipeline.
- For Raw with annotations, calibration and transform should honor
  `reject_by_annotation=True` / `skip_by_annotation=("bad", "bad_acq_skip")`.

### Preprocessing assumptions

ASR assumes:

- High-pass filtered, approximately zero-mean data.
- Same preprocessing for calibration and processing.
- Full-rank channel covariance.
- Bad/flat/noisy channels removed or excluded before ASR.
- Stable channel order and units across fit/transform.

Warnings/errors:

- Warn if `raw.info["highpass"] < 0.25` or unknown high-pass state. Do not
  silently filter in Phase 1; provide helper examples instead.
- Warn if average reference projectors are active or if rank estimate is less
  than number of selected channels.
- Error on non-finite-dominated channels, zero variance channels, mismatched
  channel names, mismatched sampling frequency, or too few samples for
  calibration.

### Calibration data selection

Modes:

- `calibration="manual"`: `fit(calibration_raw_or_array)` uses all supplied
  clean calibration data after annotation exclusion.
- `calibration="auto"`: `fit(raw)` selects clean windows using
  `clean_windows`-style robust channel RMS distributions.
- `calibration={"mask": sample_mask}` or explicit `calibration_intervals`: future
  API for annotations/intervals.

Auto clean-window algorithm:

1. Split data into windows, default 1 s, overlap 0.66 for reference selection.
2. For each channel, compute window RMS or robust log-power.
3. Fit robust clean distribution with `fit_eeg_distribution`.
4. Convert each channel/window to robust z-score.
5. Mark a window clean if no more than `ref_max_bad_channels` fraction of
   channels exceed tolerance bounds, default based on clean_rawdata references
   around `0.075` for ASR reference selection.
6. Concatenate clean windows for calibration.
7. Store `clean_window_mask_`, `clean_window_scores_`, calibration duration, and
   failure reason if not enough clean data.

### Calibration state

The fitted standard ASR state should contain:

| Attribute | Shape | Meaning |
|---|---|---|
| `mixing_` or `M_` | `(C, C)` | Matrix square root of robust calibration covariance. |
| `threshold_matrix_` or `T_` | `(C, C)` | Direction-dependent threshold matrix `diag(mu + cutoff * sigma) @ V.T`. |
| `thresholds_` | `(C,)` | Per-calibration-component RMS thresholds before multiplication by `V.T`. |
| `calibration_patterns_` | `(C, C)` | Calibration eigenvectors or patterns used to build `T_`. |
| `filter_b_`, `filter_a_` | `(order + 1,)` | Statistics-only spectral pre-emphasis filter. |
| `iir_state_` | `(C, order)` | Filter state for online/chunked processing. |
| `cov_state_` | implementation-defined | Moving covariance state for chunked processing. |
| `last_R_` | `(C, C)` | Previous reconstruction matrix for blending. |
| `last_trivial_` | bool | Whether previous `R` was identity. |
| `clean_window_mask_` | `(n_cal_windows,)` | Calibration windows used. |
| `calibration_info_` | dict/dataclass | Parameters, sample counts, rank, warnings, data fingerprint. |

### Robust covariance and threshold fitting

Implementation-level steps:

1. Convert to `float64`, replace isolated non-finite values with zero only in
   the statistics path, and log counts. If a channel has too many non-finite
   values, error.
2. Apply the same statistics-only spectral shaping filter in calibration and
   processing. For parity mode, implement the clean_rawdata Yule-Walker-like
   response or hard-coded coefficients where practical. If exact MATLAB filter
   behavior is not yet reproduced, flag this as a compatibility limitation.
3. Compute vectorized covariance samples/blocks without materializing
   `(n_times, C, C)` for long data.
4. Aggregate covariance samples using block geometric median.
5. Symmetrize and regularize covariance:
   `C = (C + C.T) / 2`; eigenvalue floor
   `eps * trace(C) / C` or configurable `regularization`.
6. Compute `M = sqrt(C)` using `np.linalg.eigh`, not `scipy.linalg.sqrtm`, for
   real symmetric stability.
7. Eigen-decompose calibration covariance or `M` for eigenvectors `V`.
8. Project filtered calibration data into `V`.
9. Compute overlapping short-window RMS for each component.
10. Fit truncated generalized Gaussian distribution with
    `fit_eeg_distribution(rms, min_clean_fraction, max_dropout_fraction,
    fit_quantiles=(0.022, 0.6), beta_grid=1.7:0.15:3.5)`.
11. Build `thresholds = mu + cutoff * sigma`.
12. Build `T = diag(thresholds) @ V.T`.

Parts requiring MATLAB verification:

- Exact Yule-Walker/precomputed filter coefficients.
- `blocksize` indexing and normalization.
- Exact `fit_eeg_distribution` grid, histogram, KL/objective, and fallback
  behavior.
- Whether calibration eigenspace uses covariance or `M`; eigenvectors are
  equivalent for SPD matrices, but numerical ordering must be tested.
- Default values in `clean_rawdata` differ by wrapper/API/GUI version.

### Sliding-window processing

Parameters:

- `window_length`: default 0.5 s for processing statistics.
- `lookahead`: default `window_length / 2`.
- `stepsize`: default 32 samples or derived from overlap; expose explicitly.
- `max_dims`: default 0.66 fraction of channels; cap on removable dimensions.
- `max_mem_mb`: real memory control, not a stub.

For each chunk:

1. Prepend carry/lookahead state.
2. Apply statistics-only filter to lookahead-shifted data.
3. Compute moving average of vectorized outer products over `N` samples.
4. Select update indices at `stepsize`.
5. For each update covariance:
   - Symmetrize/regularize.
   - Eigendecompose covariance: `Cw = V @ diag(D) @ V.T`.
   - Sort eigenvalues/eigenvectors ascending to match MATLAB behavior.
   - Compute directional thresholds:
     `theta2 = np.sum((T @ V) ** 2, axis=0)`.
   - `keep = (D < theta2) | guaranteed_low_variance_keep_mask`.
   - If all kept, `R = I`.
   - Else compute:
     `KVM = keep[:, None] * (V.T @ M)`
     `R = M @ pinv(KVM) @ (keep[:, None] * V.T)`
     The second multiplication by `keep` should be checked against MATLAB's
     `M * pinv(bsxfun(@times, keep', V' * M)) * V'`; the common description
     sometimes omits/varies the final masked `V.T`. Parity tests must settle
     the exact expression.
6. Apply `R` to data between update points with raised-cosine blending from
   `last_R`.
7. Update `last_R`, `last_trivial`, filter state, covariance state, and carry.
8. Return cleaned chunk with the same number of samples in offline mode. In
   online mode, expose the lookahead latency explicitly.

Diagnostics to store:

- `sample_mask_`: samples affected by non-identity reconstruction.
- `window_starts_`, `window_stops_`, `window_times_`.
- `n_components_reconstructed_`: `(n_windows,)`.
- `component_variances_`: compact or optional, `(n_windows, C)`.
- `component_thresholds_`: compact or optional.
- `reconstruction_matrices_`: optional; default do not store full `(n_windows,
  C, C)` for long recordings. Store hashes/summaries unless
  `store_reconstruction_matrices=True`.
- `variance_ratio_` and per-channel summary.
- `rank_`, condition numbers, regularization counts.
- `history_`: parameter dict, warnings, source object type, channel names, data
  fingerprint if cheap.

### Standard ASR pseudocode

```python
def calibrate_asr(X, sfreq, *, cutoff, window_length, window_overlap,
                  min_clean_fraction, max_dropout_fraction, max_mem_mb):
    # X: (C, S), high-pass filtered and full rank
    X = validate_float64_channels_times(X)
    B, A = design_asr_statistics_filter(sfreq)
    Xf, iir_state = apply_iir(B, A, X, axis=1)

    cov_blocks = iter_vectorized_covariance_blocks(
        Xf,
        blocksize=10,
        max_mem_mb=max_mem_mb,
    )
    C_robust = block_geometric_median(cov_blocks)
    C_robust = symmetrize_and_regularize(C_robust)
    M = matrix_sqrt_spd(C_robust)

    evals, V = eigh_sorted(C_robust, ascending=True)
    N = round(window_length * sfreq)
    step = max(1, round(N * (1 - window_overlap)))
    Y = abs(Xf.T @ V)  # (S, C)

    mu = zeros(C)
    sigma = zeros(C)
    for c in range(C):
        rms = sliding_rms(Y[:, c], N, step)
        mu[c], sigma[c] = fit_eeg_distribution(
            rms,
            min_clean_fraction=min_clean_fraction,
            max_dropout_fraction=max_dropout_fraction,
        )

    thresholds = mu + cutoff * sigma
    T = diag(thresholds) @ V.T
    return ASRState(M=M, T=T, B=B, A=A, iir=iir_state, thresholds=thresholds)


def process_asr(X, sfreq, state, *, window_length, lookahead, stepsize,
                max_dims, max_mem_mb):
    # X: (C, S), same channel order/reference as calibration
    X_ext, carry = prepend_lookahead_carry(X, state.carry, lookahead, sfreq)
    X_out = X_ext.copy()

    for chunk in iter_memory_bounded_chunks(X_ext, max_mem_mb=max_mem_mb):
        Xstats, state.iir = apply_iir(
            state.B,
            state.A,
            chunk.lookahead_shifted_data,
            zi=state.iir,
            axis=1,
        )
        cov_stream, state.cov = moving_average_covariance(
            Xstats,
            n_samples=round(window_length * sfreq),
            state=state.cov,
        )
        for update in update_indices(cov_stream, stepsize):
            Cw = symmetrize_and_regularize(cov_stream[update])
            D, V = eigh_sorted(Cw, ascending=True)
            theta2 = sum((state.T @ V) ** 2, axis=0)
            keep = D < theta2
            keep |= force_low_variance_components_to_remain(C=len(D), max_dims=max_dims)

            if all(keep):
                R = eye(C)
                trivial = True
            else:
                KVM = keep[:, None] * (V.T @ state.M)
                R = real(state.M @ pinv(KVM) @ V.T)
                trivial = False

            apply_raised_cosine_blend(
                X_out,
                start=state.last_update,
                stop=update,
                R_old=state.last_R,
                R_new=R,
                skip_if_identity=(state.last_trivial and trivial),
            )
            state.last_R = R
            state.last_trivial = trivial
            log_window_diagnostics(update, D, theta2, keep, R)

    state.carry = update_carry(X_ext, lookahead, sfreq)
    return remove_online_latency_or_restore_offline_length(X_out, state), state
```

## 7. rASR Algorithm Specification

### What changes relative to standard ASR

rASR changes the covariance geometry, not the high-level purpose:

- Standard ASR treats covariance matrices as Euclidean matrices.
- rASR treats covariance matrices as SPD manifold points and uses a
  Riemannian metric, usually the affine-invariant Riemannian metric (AIRM).
- Processing replaces Euclidean covariance averaging/smoothing with a Karcher
  mean over recent covariance matrices.
- PCA-like decomposition can be interpreted as principal geodesic analysis
  (PGA) or tangent-space PCA around a Riemannian mean.

The corpus is somewhat inconsistent about whether rASR changes calibration
substantially or mainly processing. The safest design is to share standard ASR
calibration first, then implement rASR processing geometry as a backend, while
leaving space for a later rASR-specific calibration path once parity with
rASRMatlab/MEEGkit is tested.

### Metric and operations

Use AIRM for Phase 2:

```text
d_R(C1, C2) = || log(C1^{-1/2} C2 C1^{-1/2}) ||_F
```

Required primitives:

- SPD eigenvalue clipping / diagonal loading.
- `sqrtm_spd(C)`, `invsqrtm_spd(C)`.
- `logm_spd(C)`, `expm_sym(S)`.
- AIRM log map:
  `log_C(X) = C^{1/2} log(C^{-1/2} X C^{-1/2}) C^{1/2}`.
- AIRM exp map:
  `exp_C(S) = C^{1/2} exp(C^{-1/2} S C^{-1/2}) C^{1/2}`.
- Karcher mean:
  iterative average of log maps followed by exp-map update.
- Optional tangent-space PCA/PGA.

### Python libraries

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| Custom NumPy/SciPy | No new dependency; full control; easy to test. | More code to maintain; performance tuning needed. | Implement minimal primitives first for tests and fallback. |
| pyRiemann | Mature EEG covariance functions, means, distances, tangent space. | New dependency; API/version stability to manage. | Optional dependency for rASR backend after fallback exists. |
| geomstats | General manifold abstractions. | Heavier dependency and less EEG-specific. | Do not require; research option. |
| pymanopt | Manifold optimization. | Overkill for Phase 2; current ports requiring patches are not attractive. | Avoid as dependency. |

### Regularization

rASR is sensitive to nearly singular covariances because `log(lambda)` diverges
near zero. Every covariance passed to Riemannian operations should be converted
to SPD:

```python
C = (C + C.T) / 2
eps = regularization * np.trace(C) / C.shape[0]
w, V = np.linalg.eigh(C)
C_spd = (V * np.maximum(w, eps)) @ V.T
```

Recommended defaults:

- `regularization=1e-6` relative trace floor for Riemannian operations.
- Enforce minimum samples per covariance window, ideally at least `2*C` and
  preferably `3*C`, or use shrinkage.
- Log condition numbers in diagnostics.

### Computational cost

- SPD log/exp/eigh are O(C^3).
- Karcher mean with `m` covariances and `n_iter` iterations is roughly
  O(m * n_iter * C^3).
- rASR may still be faster than standard ASR if it performs far fewer local
  decompositions, as reported by Blum et al., but this must be benchmarked in
  Python and cannot be assumed.

### API implication and recommendation

Implement rASR as:

```python
ASR(method="riemannian", experimental=True)
```

not as a separate public `RiemannianASR` class initially.

Justification:

- The user-facing lifecycle, fitted state, diagnostics, MNE handling, and
  safety warnings are the same.
- Existing package precedent favors a single estimator with mode flags for
  related behavior (`ZapLine(adaptive=True)`) when the public workflow is
  similar.
- A separate class can be added later as a thin alias if documentation or
  stability demands it.
- Keeping one class reduces duplicated Raw/Epochs/annotation/chunking code.

Tests needed for rASR:

- SPD primitive round-trips: `exp_C(log_C(X)) == X`.
- Distance symmetry and zero distance.
- Karcher mean of identical matrices.
- Regularization avoids NaN/inf on near-singular matrices.
- rASR output preserves shape, metadata, and finite values.
- Synthetic blink/motion artifact suppression does not inflate clean data.
- Regression against MEEGkit/Timeflux/rASRMatlab on tiny fixtures.

## 8. Juggler's ASR / ASRDBSCAN / ASRGEV Specification

### Readiness assessment

Updated 2026-05-06. The Kim et al. 2025 paper PDF and a community Python
implementation (`thiagorroque/asrpy`) are now both available locally under
`refs/asr/papers/` and `refs/asr/repos/asrpy-juggler/`. Phase 3 implementation
is feasible by reading both together, though the paper is paywalled (Elsevier)
and the fork is marked WIP/under test by its author. Original assessment that
the public corpus alone was insufficient is no longer accurate — an
implementation reference now exists.

### What the method changes

The core ASR reconstruction algorithm remains standard ASR. Juggler's ASR
changes calibration-data selection:

- Standard ASR can fail under extreme motion because it cannot find enough
  clean calibration data or selects a poor "ruler".
- Juggler's ASR proposes point-by-point or amplitude-derived evaluation of data
  quality before calibration.
- ASRDBSCAN and ASRGEV select cleaner calibration data, then standard ASR is
  calibrated and applied.

### ASRDBSCAN

What is specified:

- Uses DBSCAN, a density-based clustering algorithm.
- Operates on amplitude-derived features or point/window quality features.
- Identifies dense clusters interpreted as clean data and outliers interpreted
  as artifact.

What is missing:

- Exact feature definition.
- Whether features are per-channel, global, log-power, RMS, robust z-score, or
  timepoint amplitude.
- Scaling/normalization before DBSCAN.
- Exact DBSCAN parameters, distance metric, and parameter search.
- How clustered points are converted to windows or calibration samples.
- Failure/fallback behavior when DBSCAN returns too few clean samples.

### ASRGEV

What is specified:

- Fits a Generalized Extreme Value distribution to amplitude statistics.
- Uses the fitted extreme tail to exclude artifact-laden points/windows from
  calibration.

What is missing:

- Exact statistic modeled by GEV.
- Fit method and bounds.
- Tail probability / threshold choice.
- Channel aggregation rule.
- Conversion from points to contiguous calibration windows.
- Fallback behavior and minimum calibration duration.

### Validation evidence

The corpus reports:

- Simulation cases where ASRDBSCAN/ASRGEV removed motion-like artifacts where
  original ASR failed.
- 205-channel three-ball juggling EEG from 13 participants.
- Calibration usable fraction: ASRDBSCAN about 42%, ASRGEV about 24%, original
  ASR about 9%.
- Downstream ICA variance explained improvements around 30%/29% vs 26% for
  original ASR.

Evidence quality is promising but narrow, and reproducibility is blocked by the
missing implementation details above.

### Public code

- `thiagorroque/asrpy` (`https://github.com/thiagorroque/asrpy`) — fork of
  `DiGyt/asrpy` adding ASRDBSCAN and ASRGEV per Kim et al. 2025. Marked WIP /
  under test by the author. As of this writing the only public Python reference.
- The `code.ornl.gov/fub/juggler` URL previously cited as possibly hosting
  Juggler's ASR was verified to be an unrelated CUDA GPU runtime project. Do
  not pursue.
- `DiGyt/asrpy#15` is the upstream tracking issue and points to the
  thiagorroque fork.
- No public MATLAB implementation found.

### Risks

- Overcleaning if the clean cluster is too narrow.
- Under-cleaning if motion artifacts are included in the clean cluster.
- Strong dependence on channel count and amplitude scaling.
- DBSCAN parameter instability across subjects/tasks.
- Publication claims may not generalize beyond extreme high-density MoBI.

Recommended status:

- `calibration_strategy="standard"` in Phase 1.
- Add a private strategy interface so `dbscan`/`gev` can be prototyped later.
- Do not expose `calibration_strategy="dbscan"` or `"gev"` until there is
  reproducible reference code or enough detail to rederive the algorithm.
- Documentation can mention Juggler's ASR as a research direction and explain
  that it is not implemented.

## 9. Proposed Public API

### Phase 1 user-facing API

```python
from mne_denoise.asr import ASR

asr = ASR(
    sfreq=None,
    cutoff=20.0,
    window_length=0.5,
    window_overlap=0.66,
    max_dropout_fraction=0.1,
    min_clean_fraction=0.25,
    method="standard",
    calibration="auto",
    picks="eeg",
    random_state=None,
    n_jobs=None,
    verbose=None,
)

asr.fit(raw)
raw_clean = asr.transform(raw)
raw_clean = asr.fit_transform(raw)
```

Recommended full constructor:

```python
class ASR(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sfreq: float | None = None,
        *,
        cutoff: float = 20.0,
        window_length: float = 0.5,
        window_overlap: float = 0.66,
        calibration: str = "auto",
        calibration_window_length: float = 1.0,
        calibration_window_overlap: float = 0.66,
        ref_max_bad_channels: float = 0.075,
        ref_tolerances: tuple[float, float] = (-np.inf, 5.5),
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        max_dims: float | int = 0.66,
        method: str = "standard",
        picks: str | list[str] | list[int] = "eeg",
        reject_by_annotation: bool = True,
        skip_by_annotation: tuple[str, ...] = ("bad", "bad_acq_skip"),
        cov_estimator: str = "geometric_median",
        regularization: float = 1e-8,
        filter_kind: str = "asr",
        lookahead: float | None = None,
        stepsize: int | None = None,
        max_mem_mb: int | None = 512,
        copy: bool = True,
        store_reconstruction_matrices: bool = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool | str | int | None = None,
    ):
        ...
```

Methods:

```python
asr.fit(X, y=None, *, calibration=None, calibration_mask=None)
asr.transform(X, y=None, *, copy=None, return_diagnostics=False)
asr.fit_transform(X, y=None, *, calibration=None, return_diagnostics=False)
asr.get_diagnostics()
asr.get_clean_window_mask()
asr.to_annotations(min_components=1, description="ASR_REPAIR")
```

Low-level functions can remain private until stable. If exposed later:

```python
from mne_denoise.asr import calibrate_asr, process_asr

state = calibrate_asr(data, sfreq, cutoff=20.0)
clean, state, diagnostics = process_asr(data, sfreq, state)
```

### Fitted attributes

| Attribute | Meaning |
|---|---|
| `sfreq_` | Sampling frequency used. |
| `ch_names_` | Cleaned channel names in fitted order. |
| `picks_` | Integer picks into original MNE object. |
| `rank_` | Estimated rank used for warnings/regularization. |
| `M_` / `mixing_` | Calibration covariance square root. |
| `T_` / `threshold_matrix_` | Direction-dependent threshold matrix. |
| `thresholds_` | Per-component thresholds. |
| `filters_` | Optional compatibility alias for calibration/component filters, if meaningful. Avoid pretending ASR has one global unmixing like DSS. |
| `patterns_` | Optional compatibility alias for calibration patterns/eigenvectors. |
| `clean_window_mask_` | Calibration windows selected as clean. |
| `sample_mask_` | Samples/windows affected during last transform. |
| `n_components_reconstructed_` | Number of rejected/reconstructed PCs per update window. |
| `diagnostics_` | Structured calibration and transform diagnostics. |
| `calibration_info_` | Calibration metadata and warnings. |
| `history_` | Parameters, source info, warnings, and reproducibility metadata. |

Important naming note: `filters_` and `patterns_` have clear meaning in DSS and
ZapLine. ASR is window-adaptive and does not have one stable global unmixing
matrix. If `filters_`/`patterns_` are exposed for plotting compatibility, their
meaning must be explicitly documented as calibration eigenspace summaries, not
the per-window reconstruction operators.

### Error and warning policy

Errors:

- Unsupported input type or dimensionality.
- Missing `sfreq` for NumPy arrays.
- Channel mismatch between fit and transform.
- Not enough calibration samples/windows.
- Non-finite output from filter/statistics.
- Rank too low unless `allow_rank_deficient=True` is explicitly added later.

Warnings:

- Data do not appear high-pass filtered.
- Average reference or projections may reduce rank.
- More than a configured fraction of windows reconstructed.
- More than `max_dims` cap reached frequently.
- Calibration selected too little clean data.
- rASR or experimental methods used.

## 10. Implementation Roadmap

### Phase 0: design guardrails

- Create ASR module skeleton, docstrings, and tests that fail until
  implemented.
- Decide exact public constructor and diagnostics dataclasses.
- Add licensing note: behavior-compatible clean-room implementation; no copied
  MATLAB code.

### Phase 1: standard ASR MVP

1. Implement array validation and shape utilities.
2. Implement `fit_eeg_distribution` with tests on synthetic clean/outlier RMS.
3. Implement geometric median and matrix square root with SPD regularization.
4. Implement statistics filter design and stateful filtering.
5. Implement calibration on explicit clean array.
6. Implement processing on one 2D array with full-record mode.
7. Implement reconstruction matrix and raised-cosine blending.
8. Add `ASR` estimator for NumPy and MNE Raw.
9. Add diagnostics attributes.
10. Add synthetic artifact tests and no-artifact identity tests.

### Phase 1.5: parity and MNE polish

- Build tiny MATLAB `clean_rawdata` parity fixtures under
  `tests/parity/matlab_reference/`.
- Compare calibration matrices, thresholds, window masks, reconstruction counts,
  and final cleaned arrays within realistic tolerances.
- Add annotation skipping and clean-window auto calibration.
- Add MNE Raw metadata round-trip tests.
- Add examples and docs.

### Phase 2: chunking and ASR-R

- Implement real `max_mem_mb` chunking for calibration and transform.
- Add equivalence tests between full-record and chunked processing.
- Add ASR-R/sample rejection mask functionality without physically deleting
  samples by default.
- Add Epochs support with explicit semantics.
- Add report/QC helper functions.

### Phase 2: rASR experimental backend

- Implement `_riemann.py` primitives and tests.
- Add optional pyRiemann bridge if dependency policy allows.
- Add `ASR(method="riemannian", experimental=True)`.
- Validate against MEEGkit/Timeflux/rASRMatlab tiny fixtures.

### Phase 3: profiles and research variants

- Calibration strategy interface for Juggler-style DBSCAN/GEV prototypes.
- NEAR-style infant profile docs and parameter sweep helpers.
- Sleep/Dusk2Dawn profile docs and slow-wave preservation checks.
- Online streaming API:
  `asr.partial_transform(chunk, state=None)` or a stateful `transform_chunk`.

## 11. Test Plan

Mandatory unit tests:

- Shape validation for arrays, Raw, Epochs, Evoked rejection/transform-only.
- `fit_eeg_distribution` robustness to high-tail artifacts and low dropouts.
- Geometric median on identical, outlier-contaminated, and ill-conditioned
  covariance samples.
- Matrix square root reconstructs covariance within tolerance.
- Statistics filter returns finite output and preserves shape.
- Moving covariance matches direct covariance on small data.
- Reconstruction matrix is identity when all components are kept.
- Reconstruction reduces injected blink/motion bursts and preserves clean
  sinusoid/ERP/SSVEP features within tolerance.
- Raised-cosine blending is continuous and endpoint-correct.
- MNE Raw output preserves type, info, annotations, bads, non-picked channels,
  channel order, and sampling frequency.
- Rank-deficient data warning/error path.
- Memory chunking bound path and chunk equivalence.

Parity tests:

- MATLAB `asr_calibrate`: compare `M`, `T`, thresholds on a small fixed array.
- MATLAB `asr_process`: compare cleaned data and window counts on a small fixed
  array.
- MATLAB `clean_windows`: compare selected windows/sample masks.
- MEEGkit/ASRpy sanity comparison on same parameters, not as the legal source
  of truth.

Validation tests and benchmarks:

- Clean seated synthetic data: ASR should make near-zero changes.
- Blink artifact synthetic data: frontopolar burst variance decreases.
- Muscle/high-frequency burst synthetic data: broadband artifact decreases
  without collapsing alpha.
- ERP synthetic data: amplitude/latency preservation.
- SSVEP synthetic data: stimulus-frequency SNR preserved or improved.
- Mobile/gait mini-slices: reconstructed-window fraction, PSD, gait/IMU
  correlation where metadata exists.

Minimal CI suite:

- Pure synthetic tests only plus one tiny NPZ/FIF fixture if allowed.
- Full OpenNeuro/MoBI benchmarks stay in scripts or optional slow tests.

## 12. Documentation, Examples, and QC

Docs to add:

- `docs/asr.rst`: overview, when to use ASR, prerequisites, calibration, cutoff
  guidance, MNE examples, diagnostics.
- API entries in `docs/api.rst`.
- Changelog fragment when code is implemented.
- Warning box: ASR is not a universal cleaning method; low cutoff can
  overclean; ASR should usually run before ICA and after bad-channel/high-pass
  preparation.

Examples:

| Example | Purpose |
|---|---|
| `examples/asr/plot_01_asr_basics.py` | Fit on Raw with auto calibration and inspect diagnostics. |
| `examples/asr/plot_02_calibration_baseline.py` | Fit on explicit standing/rest baseline, transform task Raw. |
| `examples/asr/plot_03_parameter_tuning.py` | Sweep `cutoff` and show variance/PSD/reconstruction fraction. |
| `examples/asr/plot_04_mne_annotations.py` | Skip bad annotations and add ASR repair annotations. |
| `examples/asr/plot_05_mobile_eeg_qc.py` | MoBI/gait-style QC panels if data are available. |
| `examples/asr/plot_06_riemannian_experimental.py` | rASR comparison after Phase 2. |

QC outputs:

- Summary table: cutoff, method, calibration duration, selected windows,
  reconstructed fraction, median components/window, max components/window,
  variance removed, rank warnings.
- Figures:
  time-series overlay with reconstructed mask, component count trace, variance
  topomap, PSD before/after, spectral distortion, ERP/SSVEP preservation if
  events are provided, Riemannian distance trajectory for rASR.

## 13. Acceptance Criteria

Phase 1 is complete when:

- `ASR(method="standard")` fits and transforms Raw and 2D arrays.
- Clean synthetic data are almost unchanged.
- Injected high-amplitude bursts are reduced.
- MNE metadata preservation tests pass.
- Core primitives are covered by unit tests.
- Chunking is either implemented or explicitly unavailable; no fake `max_mem`
  parameter is accepted without effect.
- Documentation clearly warns about high-pass, rank, bad channels, and cutoff
  risks.

Phase 1.5 is complete when:

- MATLAB parity fixtures exist and are documented.
- Differences from clean_rawdata are quantified and explained.
- Auto calibration selection is tested.

Phase 2 is complete when:

- Chunked processing is memory-bounded and numerically close to full-record
  processing.
- ASR-R/sample masks are exposed.
- rASR has geometry tests and at least one external-reference comparison.

Scientific acceptance thresholds should be empirical, but initial targets are:

- Clean/no-artifact synthetic relative RMS change < 1e-6 to 1e-4 depending on
  filter path.
- MATLAB parity final cleaned array relative RMS error < 1e-2 initially, then
  tightened as exact compatibility improves.
- Reconstructed-window count within 5 percentage points of reference on parity
  fixtures.
- No non-finite output on stress tests.
- Runtime/memory scales approximately linearly with samples for fixed channel
  count when chunking is enabled.

## 14. Open Questions Before Implementation

- Which cutoff default should `mne-denoise` choose? Recommendation: use
  `cutoff=20.0` for a conservative adult EEG default, document that `5` is
  aggressive, and provide tuning examples.
- Should Phase 1 silently high-pass data? Recommendation: no. Warn and document
  preprocessing; filtering changes data and belongs in the user's pipeline.
- Should average-referenced data be supported with regularization? Recommendation:
  error or strong warning by default; add `allow_rank_deficient=True` later only
  after tests.
- Should exact MATLAB Yule-Walker filter behavior be mandatory for MVP?
  Recommendation: implement a compatibility filter path, but make parity
  limitations explicit until verified.
- How to store `reconstruction_matrices_` without memory blow-up? Recommendation:
  default to summaries and allow full matrices only by opt-in.
- Should rASR be a hard dependency on pyRiemann? Recommendation: no; implement
  SciPy fallback and make pyRiemann optional.
- Are Juggler details sufficient? Recommendation: no; postpone public API until
  source code or exact algorithm details are available.

## 15. Immediate Next Engineering Tasks

When implementation starts, do this in order:

1. Add `mne_denoise/asr/` skeleton and public `ASR` stub with docstring only.
2. Add failing tests for public API, MNE Raw round-trip, and synthetic identity.
3. Implement `_stats.fit_eeg_distribution`.
4. Implement `_stats.block_geometric_median`.
5. Implement `_filters.design_asr_filter`.
6. Implement `_calibration.calibrate_asr`.
7. Implement `_process.process_asr` without chunking on small arrays.
8. Wire `ASR.fit/transform`.
9. Add diagnostics and QC helper functions.
10. Add chunking and parity fixtures.

Do not start with rASR or Juggler. They depend on a correct standard ASR core
and would make it harder to debug the foundational algorithm.
