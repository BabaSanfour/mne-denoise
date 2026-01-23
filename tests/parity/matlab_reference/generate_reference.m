% Generate DSS reference results using NoiseTools
% Run this in MATLAB directly, then compare with Python
%
% Usage:
%   1. Open MATLAB
%   2. cd to d:\PhD\mne-denoise\tests\parity\matlab_reference
%   3. Run this script: generate_reference
%   4. Results saved to dss_reference_results.mat
%   5. Run Python comparison: python compare_with_matlab.py

clear; clc;

%% Setup
fprintf('=== DSS Reference Generation ===\n\n');

% Add NoiseTools to path
noisetools_path = 'C:\Users\s\Documents\MATLAB\NoiseTools';
if ~exist(noisetools_path, 'dir')
    error('NoiseTools not found at: %s', noisetools_path);
end
addpath(genpath(noisetools_path));

% Verify nt_dss0 exists
if ~exist('nt_dss0', 'file')
    error('nt_dss0 not found. Check NoiseTools installation.');
end
fprintf('NoiseTools found: %s\n', which('nt_dss0'));

%% Generate synthetic test data
fprintf('\n1. Generating test data...\n');

rng(42);  % Same seed as Python
n_samples = 1000;
n_channels = 8;
sfreq = 250;

% Base noise
data = randn(n_samples, n_channels) * 0.5;

% Add 10 Hz alpha oscillation
t = (0:n_samples-1)' / sfreq;
alpha_source = sin(2*pi*10*t);
alpha_mixing = randn(1, n_channels);
alpha_mixing = alpha_mixing / norm(alpha_mixing);
data = data + alpha_source * alpha_mixing * 2;

% Add 50 Hz line noise
line_source = sin(2*pi*50*t);
line_mixing = randn(1, n_channels);
line_mixing = line_mixing / norm(line_mixing);
data = data + line_source * line_mixing * 1.5;

fprintf('   Data: %d samples x %d channels\n', n_samples, n_channels);

%% Test 1: nt_dss0 with identity bias
fprintf('\n2. Running nt_dss0 (identity bias)...\n');

% nt_dss0 expects covariance matrices: nt_dss0(C0, C1)
% C0 = baseline covariance, C1 = biased covariance
% For identity bias, C0 = C1

% Compute covariances
C0 = (data' * data) / n_samples;  % (n_channels, n_channels)
C1 = C0;  % Identity bias

[todss_identity, fromdss_identity, pwr_identity] = nt_dss0(C0, C1);

fprintf('   todss shape: %d x %d\n', size(todss_identity));
fprintf('   Top 3 eigenvalues: %.4f, %.4f, %.4f\n', pwr_identity(1:3));

%% Test 2: nt_dss0 with bandpass bias (alpha 8-12 Hz)
fprintf('\n3. Running nt_dss0 (alpha bandpass bias)...\n');

% Bandpass filter the data
[b, a] = butter(4, [8 12]/(sfreq/2), 'bandpass');
data_alpha = filtfilt(b, a, data);

% Compute biased covariance
C1_alpha = (data_alpha' * data_alpha) / n_samples;

[todss_alpha, fromdss_alpha, pwr_alpha] = nt_dss0(C0, C1_alpha);

fprintf('   Top 3 eigenvalues: %.4f, %.4f, %.4f\n', pwr_alpha(1:3));

%% Test 3: Check if top alpha filter correlates with true mixing
top_filter_alpha = todss_alpha(:, 1)';
corr_with_true = abs(corr(top_filter_alpha', alpha_mixing'));
fprintf('   Top filter correlation with true alpha: %.4f\n', corr_with_true);

%% Test 4: nt_zapline (if available)
fprintf('\n4. Checking nt_zapline...\n');
if exist('nt_zapline', 'file')
    % Add line noise for testing
    data_noisy = data + line_source * line_mixing * 3;

    fline = 50 / sfreq;  % Normalized frequency
    [data_cleaned, artifacts] = nt_zapline(data_noisy, fline);

    % Compute PSD reduction at 50 Hz
    [psd_orig, f] = pwelch(data_noisy(:,1), 256, [], [], sfreq);
    [psd_clean, ~] = pwelch(data_cleaned(:,1), 256, [], [], sfreq);

    idx_50 = find(f >= 49 & f <= 51);
    power_orig = mean(psd_orig(idx_50));
    power_clean = mean(psd_clean(idx_50));
    zapline_reduction_db = 10 * log10(power_orig / max(power_clean, 1e-12));

    fprintf('   ZapLine 50 Hz reduction: %.1f dB\n', zapline_reduction_db);

    save_zapline = true;
else
    fprintf('   nt_zapline not found, skipping\n');
    data_cleaned = [];
    zapline_reduction_db = nan;
    save_zapline = false;
end

%% Save results
fprintf('\n5. Saving reference results...\n');

output_file = fullfile(pwd, 'dss_reference_results.mat');

save(output_file, ...
    'data', 'n_samples', 'n_channels', 'sfreq', ...
    'alpha_mixing', 'line_mixing', ...
    'todss_identity', 'fromdss_identity', 'pwr_identity', ...
    'todss_alpha', 'fromdss_alpha', 'pwr_alpha', ...
    'data_cleaned', 'zapline_reduction_db');

fprintf('   Saved to: %s\n', output_file);
fprintf('\n=== Done! ===\n');
fprintf('\nNow run Python comparison:\n');
fprintf('   python compare_with_matlab.py\n');
