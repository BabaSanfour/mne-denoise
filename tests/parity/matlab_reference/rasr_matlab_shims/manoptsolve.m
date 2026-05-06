function Xsol = manoptsolve(problem, X0)
% Minimal shim for rASRMatlab parity generation.
% For the full-basis rASR use case, returning the orthonormalized initial
% point is sufficient to reproduce the deterministic modified eigenproblem.

if nargin < 2 || isempty(X0)
    error('manoptsolve requires an initial point X0.');
end

[U, ~, V] = svd(X0, 'econ');
Xsol = U * V';
