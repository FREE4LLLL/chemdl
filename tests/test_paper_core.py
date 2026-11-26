"""Self-contained unit, mathematical invariant, and artifact checks for the paper release."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_frozen_theory_hash_and_module_imports():
    from chemdl.deploy_gate import FROZEN_THEORY_LF_SHA256

    theory_path = ROOT / "chemdl" / "theory.py"
    lf_sha = hashlib.sha256(theory_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert lf_sha == FROZEN_THEORY_LF_SHA256

    for mod in (
        "chemdl",
        "chemdl.theory",
        "chemdl.deploy_gate",
        "chemdl.transfer",
        "chemdl.diffprep",
        "chemdl.classic",
        "chemdl.select",
        "chemdl.uq",
        "chemdl.corn",
        "chemdl.tablets",
        "chemdl.tecator",
        "chemdl.cordoba",
        "chemdl.diesel",
    ):
        importlib.import_module(mod)


def test_theorem1_and_theorem2_algebraic_invariants():
    from chemdl.theory import (
        closed_form_optimal_ws,
        snv_critical_coupling,
        snv_snr_curve,
        snv_theoretical_gain_pct,
    )

    # Theorem 1: exact sign flip across r_crit
    alpha, beta, sigma_a = 40.0, 2.0, 0.15
    rc = snv_critical_coupling(alpha, beta, sigma_a, clip=False)
    assert 0.0 < rc < 1.0
    assert snv_theoretical_gain_pct(alpha, beta, rc * 0.9, sigma_a) < 0.0
    assert snv_theoretical_gain_pct(alpha, beta, min(rc * 1.1, 0.99), sigma_a) > 0.0

    # Theorem 2: flat degenerate family (beta != 1)
    a_d, b_d, r_d = 0.75, 0.5, float(np.sqrt(1.0 / 3.0))
    wg = np.linspace(0.0, 1.0, 101)
    snrs = snv_snr_curve(wg, a_d, b_d, r_d, sigma_a=0.15, sigma_non=0.0)
    assert np.max(np.abs(snrs - b_d)) < 1e-12
    assert closed_form_optimal_ws(a_d, b_d, r_d, 0.15, sigma_non=0.0) == 0.5


def test_theorem3_and_theorem4_derivative_and_upper_bound():
    from chemdl.theory import snv_snr_curve

    rng = np.random.default_rng(20261125)
    for _ in range(30):
        alpha = float(rng.uniform(5.0, 150.0))
        beta = float(rng.uniform(0.5, 25.0))
        r = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 0.90))
        sa = float(rng.uniform(0.02, 0.20))
        snon = float(rng.uniform(0.02, 0.20))
        A, C = sa ** 2, snon ** 2
        w_wiener = A / (A + C)

        def p_term(w: float) -> float:
            q = 1.0 + alpha * (A * (1.0 - r ** 2) * (1.0 - w) ** 2 + C * (w ** 2))
            return (alpha * (r ** 2) * A * (1.0 - w) ** 2) / q

        def p_prime_exact(w: float) -> float:
            q = 1.0 + alpha * (A * (1.0 - r ** 2) * (1.0 - w) ** 2 + C * (w ** 2))
            return -(2.0 * alpha * (r ** 2) * A * (1.0 - w) * (1.0 + alpha * C * w)) / (q ** 2)

        eps = 1e-7
        for w_chk in (0.15, w_wiener, 0.85):
            fd = (p_term(w_chk + eps) - p_term(w_chk - eps)) / (2.0 * eps)
            ex = p_prime_exact(w_chk)
            assert abs(fd - ex) < 1e-8 and ex <= 0.0
        assert p_prime_exact(w_wiener) < 0.0

        opt_r = minimize_scalar(
            lambda w: -float(snv_snr_curve(w, alpha, beta, r, sa, sigma_non=snon)),
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1e-10},
        )
        assert 0.0 <= float(opt_r.x) < w_wiener - 1e-7


def test_exact_press_loo_and_inductive_zero_leakage():
    from chemdl.deploy_gate import fit_snv_gated_workflow
    from chemdl.theory import diagnose_snv_decision, exact_press_loo_residuals
    from chemdl.transfer import dssnv_apply

    rng = np.random.default_rng(7)
    n, L = 24, 30
    X = rng.normal(2.0, 0.3, size=(n, L))
    y = 1.5 * X[:, 2] - 0.8 * X[:, 9] + rng.normal(0.0, 0.05, size=n)

    # Exact Allen-PRESS vs brute-force LOO ridge
    gamma = 0.5
    e_loo, g_used = exact_press_loo_residuals(X, y, gamma=gamma)
    assert g_used == gamma
    e_bf = np.empty(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        Xtr, ytr = X[m], y[m]
        xm, ym = Xtr.mean(0, keepdims=True), ytr.mean()
        U, s, Vt = np.linalg.svd(Xtr - xm, full_matrices=False)
        W = Vt.T @ ((s / (s ** 2 + gamma)) * (U.T @ (ytr - ym)))
        e_bf[i] = y[i] - (ym + (X[i : i + 1] - xm) @ W)[0]
    assert float(np.max(np.abs(e_loo.ravel() - e_bf))) < 1e-10

    # Strict single-sample inductive zero-leakage
    pred_fn, info = fit_snv_gated_workflow(X[:18], y[:18], abstain_policy="snv")
    assert info["gate_report"]["decision"] in {"IMPROVE", "WORSEN", "ABSTAIN"}
    X_te = X[18:]
    p_batch = pred_fn(X_te)
    p_single = np.array([pred_fn(X_te[i : i + 1])[0] for i in range(len(X_te))])
    assert np.allclose(p_batch, p_single, atol=1e-10)

    X_poison = np.vstack([X_te, np.full((3, L), 100.0)])
    assert np.allclose(p_batch, pred_fn(X_poison)[: len(X_te)], atol=1e-10)

    s_bar = float(np.clip(X[:18].std(axis=1), 1e-12, None).mean())
    assert np.allclose(
        dssnv_apply(X_te, wc=1.0, ws=0.4, s_bar=s_bar),
        dssnv_apply(X_poison, wc=1.0, ws=0.4, s_bar=s_bar)[: len(X_te)],
        atol=1e-12,
    )
    diag = diagnose_snv_decision(X[:18], y[:18])
    assert diag["decision"] == info["gate_report"]["decision"]


def test_r2_shape_synthetic_identifiability_boundary():
    from chemdl.theory import estimate_physical_params_v2

    rng = np.random.default_rng(77)
    n, L = 60, 64
    wl = np.linspace(0.0, 2.0 * np.pi, L)
    lam_tilde = np.linspace(-1.0, 1.0, L)
    c1 = rng.normal(0.0, 1.0, (n, 1))
    c2 = rng.normal(0.0, 1.0, (n, 1))
    X0 = 3.0 + 0.6 * np.sin(wl)[None, :] + 0.3 * c1 * np.cos(wl)[None, :] + 0.2 * c2 * np.sin(2.0 * wl)[None, :]
    y_dum = c1[:, 0]

    eps = rng.normal(0.0, 0.20, n)
    eps -= eps.mean()
    X_scalar = np.exp(eps)[:, None] * X0
    X_tilt = np.exp(eps[:, None] * (1.0 + 1.0 * lam_tilde[None, :])) * X0

    u_scalar = 1.0 - estimate_physical_params_v2(X_scalar, y_dum)["r2_s_shape"]
    u_tilt = 1.0 - estimate_physical_params_v2(X_tilt, y_dum)["r2_s_shape"]
    assert u_scalar > 0.60
    assert u_tilt < 0.25


def test_archived_results_and_release_hygiene():
    p0 = json.loads((ROOT / "_audit" / "res_review_p0_2026_11_23.json").read_text(encoding="utf-8"))
    assert p0["part3_theorem1_confusion"]["overall_v2"]["acc"] == pytest.approx(0.6941, abs=1e-4)

    d18 = json.loads((ROOT / "_audit" / "res_prereg_deploy_holdout_2026_11_24.json").read_text(encoding="utf-8"))
    assert d18["deployment_strategy_eval"]["ALL_18"]["same_pls"]["Gate_Fallback_SNV"]["mean_regret_pct_vs_oracle"] == pytest.approx(1.95, abs=1e-2)

    r3 = json.loads((ROOT / "_audit" / "res_prereg_round3_unseen_2026_11_24.json").read_text(encoding="utf-8"))
    assert r3["n_tasks"] == 13 and r3["n_distinct_labs"] == 6

    wi30 = json.loads((ROOT / "_audit" / "res_wiener_ws_2026_11_25.json").read_text(encoding="utf-8"))
    assert wi30["summary"]["EXTERNAL_30"]["pls"]["methods"]["ws_wiener"]["asymmetric_cost_w10_exact"] == pytest.approx(69.18, abs=0.05)

    id_res = json.loads((ROOT / "_audit" / "res_r2_shape_identifiability_2026_11_25.json").read_text(encoding="utf-8"))
    assert id_res["part_a1_scalar_sweep"]["pooled_metrics_raw_vs_f_adj_pop_lin"]["pearson_r"] > 0.99

    mr = json.loads((ROOT / "_audit" / "res_major_revision_2026_11_25.json").read_text(encoding="utf-8"))
    assert mr["part1_vsn_comparison"]["VALID_17"]["C_10"] == pytest.approx(229.35, abs=0.05)
    assert mr["part2_nested_cv_bias_ablation"]["Pooled_31"]["three_state_agreement"] == "27/31"
    assert mr["part3_noninferiority_and_cluster_cis"]["external_30_noninferiority"]["ridge"]["cluster12_ucb95_wiener_minus_inner_cv"] < 0.03

    from _audit.bench_major_revision_2026_11_25 import verify_vsn_synthetic

    syn = verify_vsn_synthetic()
    assert syn["w_baseline_mean"] > 0.80 and syn["w_chemical_peak_mean"] < 0.25
    assert syn["max_single_sample_diff"] < 1e-12 and syn["max_poison_diff"] < 1e-12

    # Hygiene: no manuscript drafts, no personal paths, no hardcoded tokens
    assert not list(ROOT.rglob("MANUSCRIPT_*.md"))
    pat = re.compile(r"([A-Za-z]:\\Users\\|hf_[A-Za-z0-9]{20,}|tabpfn_sk_[A-Za-z0-9_-]{15,})")
    for p in list(ROOT.glob("*.py")) + list((ROOT / "chemdl").glob("*.py")) + list((ROOT / "tests").glob("*.py")):
        assert not pat.search(p.read_text(encoding="utf-8", errors="ignore")), f"Hygiene hit in {p}"
