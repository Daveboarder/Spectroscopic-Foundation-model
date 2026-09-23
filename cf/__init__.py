"""
Calibration-free (CF) LIBS quantification: Saha–Boltzmann solver package.

Modules
-------
tables      CFTables — per-element constants (E_ion, atomic mass, LOD),
            partition-function grid U(T) and the curve-of-growth lookup table.
solver_np   Reference numpy implementation for one spectrum
            (``saha_boltzmann_solve_np``).
layer       ``SahaBoltzmannLayer`` — batched, differentiable torch port with
            identical maths (float64, no trainable parameters).
classical   Zero-parameter line weighting (``classical_weights``) and the
            curated 54-line CF-OES list (``select_cf_oes_lines``).

Solver maths (shared by ``solver_np`` and ``layer``)
----------------------------------------------------
For line i of element e in ionisation stage z in {0, 1}:

    y_i = ln(area_i * lambda_i / (g_k A_k)) - z ln F(T),   F(T) = 2 (2 pi m_e k T / h^2)^{3/2}
    y_i = q_e - (E_k,i + z E_ion,e) beta - z eta,          beta = 1/(kT) [1/eV], eta = ln N_e

Unknowns theta = [q_1..q_E, beta, eta] from weighted least squares with prior
rows ``prior_T (beta - beta0)^2`` and ``prior_Ne (eta - eta0)^2`` plus a ridge.
The T-dependence of F(T) is handled by a fixed-point loop (``n_iter``).
Optional self-absorption correction: tau0_i from the current plasma state and
``N l = 10^log10_Nl0``, ``area_i <- area_i / f(tau0_i)`` (curve of growth).
Closure: x_e ∝ U_I,e(T) exp(q_e) (1 + S10_e(T, N_e)); elements without used
lines take their share from the seed ``C0``; number → mass fractions via the
atomic masses; censored below the per-element LOD.
"""

from cf.tables import CFTables, CFTablesTorch, build_cf_tables
from cf.solver_np import CFResult, saha_boltzmann_solve_np
from cf.classical import classical_weights, select_cf_oes_lines, load_cf_oes_lines, CF_OES_54_TSV, CF_MINERAL_TSV

__all__ = [
    "CFTables", "CFTablesTorch", "build_cf_tables",
    "CFResult", "saha_boltzmann_solve_np",
    "classical_weights", "select_cf_oes_lines", "load_cf_oes_lines", "CF_OES_54_TSV", "CF_MINERAL_TSV",
]
