# ASR MATLAB Reference Fixtures

This directory contains scripts for generating ASR and rASR parity references
from the local MATLAB checkouts under `refs/asr/repos/`.

The workflow now produces:

- legacy single-case artifacts:
  - `asr_reference_results.mat`
  - `rasr_reference_results.mat`
- expanded case-matrix artifacts:
  - `asr_case_input_<name>.mat`
  - `asr_case_reference_<name>.mat`
  - `rasr_case_reference_<name>.mat`
- adaptive-AASR artifacts:
  - `aasr_case_input_<name>.mat`
  - `aasr_case_reference_<name>_<variant>_<n>updates.mat`

1. From the repository root, create the deterministic input fixture:

   ```bash
   python tests/parity/matlab_reference/generate_asr_input.py
   ```

   This writes the legacy fixture plus the full case matrix.

2. For standard ASR parity, add the local `clean_rawdata` reference checkout
   to the path, then run:

   ```matlab
   addpath(genpath('D:\PhD\mne-denoise\refs\asr\repos\clean_rawdata'))
   cd('D:\PhD\mne-denoise\tests\parity\matlab_reference')
   generate_asr_reference
   ```

3. For experimental rASR parity, run:

   ```matlab
   cd('D:\PhD\mne-denoise\tests\parity\matlab_reference')
   generate_rasr_reference
   ```

   `generate_rasr_reference.m` adds the local `rASRMatlab` checkout and a
   minimal parity-only shim path for the missing Manopt helpers.

4. Commit or otherwise preserve the generated `.mat` artifacts only if the
   project policy allows binary MATLAB reference artifacts.

5. For adaptive AASR parity, first generate the adaptive inputs:

   ```bash
   python tests/parity/matlab_reference/generate_aasr_input.py
   ```

   Then run:

   ```matlab
   cd('D:\PhD\mne-denoise\tests\parity\matlab_reference')
   generate_aasr_reference
   ```

   `generate_aasr_reference.m` adds the local `refs/asr/repos/AASR` checkout
   and emits PSP/PSW references for first-update and repeated-update cases.

The MATLAB script disables ASR's spectral shaping filter by passing `B=1, A=1`.
This isolates calibration covariance, generalized-Gaussian threshold fitting,
and reconstruction behavior before testing the clean_rawdata Yule-Walker
statistics filter.

The case matrix currently covers:

- manual calibration against separate clean reference data
- auto calibration via MATLAB `clean_windows`
- multiple cutoffs
- multiple sampling rates
- multiple `max_dims` settings
- moderate and strong burst regimes

The adaptive matrix currently covers:

- `variant="psp"` and `variant="psw"`
- first-update behavior (`update()` acting like initial `subspace()`)
- repeated adaptive updates across multiple chunks
- moderate and strong synthetic burst regimes
