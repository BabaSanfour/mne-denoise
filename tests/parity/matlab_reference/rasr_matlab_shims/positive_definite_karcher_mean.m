function Cmean = positive_definite_karcher_mean(A)
% Minimal affine-invariant Karcher mean shim for rASRMatlab parity fixtures.

if ndims(A) ~= 3
    error('A must have shape C x C x N.');
end

[C, ~, N] = size(A);
if N < 1
    error('At least one SPD matrix is required.');
end

Cmean = mean(A, 3);
Cmean = regularize_spd(Cmean);
tol = 1e-7;
max_iter = 32;

for iter = 1:max_iter %#ok<NASGU>
    sqrtC = sqrtm_spd(Cmean);
    invsqrtC = invsqrtm_spd(Cmean);
    tangent = zeros(C, C);
    for k = 1:N
        centered = invsqrtC * regularize_spd(A(:, :, k)) * invsqrtC;
        tangent = tangent + logm_spd(centered) / N;
    end
    update_norm = norm(tangent, 'fro');
    if update_norm <= tol * max(norm(Cmean, 'fro'), 1)
        break;
    end
    Cmean = sqrtC * expm_sym(tangent) * sqrtC;
    Cmean = regularize_spd(Cmean);
end


function C = regularize_spd(C)
    C = (C + C') / 2;
    [V, D] = eig(C);
    d = diag(D);
    floorv = 1e-8 * max([trace(C) / size(C, 1), max(d), 1]);
    d = max(d, floorv);
    C = V * diag(d) * V';
    C = real((C + C') / 2);
end


function S = sqrtm_spd(C)
    C = regularize_spd(C);
    [V, D] = eig(C);
    d = max(diag(D), eps);
    S = V * diag(sqrt(d)) * V';
    S = real((S + S') / 2);
end


function S = invsqrtm_spd(C)
    C = regularize_spd(C);
    [V, D] = eig(C);
    d = max(diag(D), eps);
    S = V * diag(1 ./ sqrt(d)) * V';
    S = real((S + S') / 2);
end


function L = logm_spd(C)
    C = regularize_spd(C);
    [V, D] = eig(C);
    d = max(diag(D), eps);
    L = V * diag(log(d)) * V';
    L = real((L + L') / 2);
end


function E = expm_sym(S)
    S = (S + S') / 2;
    [V, D] = eig(S);
    E = V * diag(exp(diag(D))) * V';
    E = real((E + E') / 2);
end
