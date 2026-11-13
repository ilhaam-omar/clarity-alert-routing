# Data access and reproduction scope

We analyzed MIMIC-IV v3.1. Obtain the credentialed dataset from https://physionet.org/content/mimiciv/3.1/ and comply with its access requirements and data-use agreement. The release excludes raw tables, cohort identifiers, note text, patient-level states, and per-order decisions. De-identification of MIMIC records does not make these records unrestricted for redistribution.

## Exact frozen-input reproduction

The corrected experiment starts from three frozen, restricted files:

- `01_order_timeline.csv`
- `02_detected_hazards.csv`
- `03_longitudinal_states.csv`

The exact SHA-256 hashes and sizes are recorded in `experiments/experiment_source_guard.json`. We retain these fingerprints so a local run can distinguish the manuscript inputs from another cohort or a later preprocessing revision. Keep these files in authorized storage. The public repository cannot reproduce the full MIMIC results without independently authorized inputs.

## Upstream reconstruction

The `scripts/00a*.py` audit and cohort-construction scripts and Stages 01–03 are included for inspection and authorized reconstruction. Start with the original transplant cohort seed, local MIMIC source tables, and `config/mimic_paths.example.json`; copy the template to `config/mimic_paths.local.json` and set the actual paths. The original seed extraction query is not included in the supplied source archive, so this release does not claim a tested one-command rebuild of cohort membership from a fresh MIMIC download.

The episode scripts identify exact ICD procedure anchors, audit anchor validity, assign admissions to transplant episodes, and construct the first-year inpatient cohort. Their filenames specify their order: `00a1_transplant_anchor_validity.py`, `00a2_transplant_episode_scope.py`, and `00a3_build_longitudinal_cohorts.py`. Each accepts `--paths config/mimic_paths.local.json`. Use the generated first-year cohort configuration for Stages 01–03. Compare resulting cohort counts and input fingerprints with the frozen analysis before treating a reconstruction as equivalent.

The manuscript run used the transplant-coded index admissions and subsequent inpatient admissions through 365 days, assigned to the most recent preceding episode. Medication decision times use linked provider-order times with prescription-start fallback. Laboratory context uses recorded `charttime`; this implementation does not verify that `storetime` preceded each decision. These timestamps support retrospective temporal ordering, not a claim that every feature was available in real time.

## Local outputs

Per-record outputs are written beneath local `private/` directories. The matched runner creates a ZIP using an explicit aggregate allowlist. Review exported aggregate tables for disclosure risk before public release. `.gitignore` prevents common local output directories from being committed accidentally; it does not determine whether a new file is appropriate for release.

The figures and tables in `results/` describe model allocation and structural checks. They do not include observed professional assessments, patient outcomes, or treatment effects.
