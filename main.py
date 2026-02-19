"""Streamlit dashboard for biomechanical simulation and Bayesian optimization."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import streamlit as st

from optimization_pipeline import (
    biomechanical_model,
    chambolle_tv_denoise_1d,
    pain_from_stress,
    total_variation_1d,
)
from models.neural_ode import TORCHDIFFEQ_AVAILABLE

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        from botorch.acquisition import ExpectedImprovement
        from botorch.models import SingleTaskGP
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        try:
            from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
        except ImportError:
            from botorch.fit import fit_gpytorch_model

        BOTORCH_AVAILABLE = True
    except ImportError:
        BOTORCH_AVAILABLE = False
else:
    BOTORCH_AVAILABLE = False


CONVERGENCE_PNG = Path("convergence.png")


def to_numpy(values) -> np.ndarray:
    """Convert torch/numpy values to numpy array."""
    if TORCH_AVAILABLE and isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def rows_to_csv_bytes(rows: list[dict]) -> bytes:
    """Serialize rows to CSV bytes."""
    if not rows:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def run_single_simulation(
    angle: float,
    raideur: float,
    damping: float,
    frequence: float,
    n_steps: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one biomechanical simulation and return time/stress/pain trajectories."""
    params_np = np.array([[angle, raideur, damping, frequence]], dtype=float)
    if TORCH_AVAILABLE:
        params = torch.tensor(params_np, dtype=torch.float32)
    else:
        params = params_np

    stress = to_numpy(biomechanical_model(params, n_steps=n_steps)).reshape(-1)
    time = np.linspace(0.0, 1.0, stress.size)
    pain = to_numpy(pain_from_stress(stress)).reshape(-1)
    return time, stress, pain


def run_neural_ode_simulation(
    position0: float,
    velocity0: float,
    force0: float,
    duration: float,
    n_steps: int,
    hidden_dim: int,
    method: str,
    use_adjoint: bool,
) -> dict[str, np.ndarray | float]:
    """Simulate continuous dynamics with a Neural ODE."""
    if not TORCH_AVAILABLE or not TORCHDIFFEQ_AVAILABLE:
        raise RuntimeError("Neural ODE simulation requires torch and torchdiffeq.")

    from models.neural_ode import KneeDynamicsODE, integrate_knee_dynamics

    model = KneeDynamicsODE(hidden_dim=hidden_dim).to(dtype=torch.double)
    model.eval()

    initial_state = torch.tensor([[position0, velocity0, force0]], dtype=torch.double)
    time_grid = torch.linspace(0.0, duration, n_steps, dtype=torch.double)

    with torch.no_grad():
        trajectory = integrate_knee_dynamics(
            model,
            initial_state,
            time_grid,
            method=method,
            use_adjoint=use_adjoint,
        )

    traj = to_numpy(trajectory[:, 0, :])
    t = to_numpy(time_grid)
    position = traj[:, 0]
    velocity = traj[:, 1]
    force = traj[:, 2]

    accel = np.gradient(velocity, t)
    mass = float(max(model.mass.item(), 1e-3))
    stiffness = float(model.stiffness.item())
    damping = float(model.damping.item())
    torque = mass * accel + damping * velocity + stiffness * position

    return {
        "time": t,
        "position": position,
        "velocity": velocity,
        "force": force,
        "accel": accel,
        "torque": torque,
        "mass": mass,
        "stiffness": stiffness,
        "damping": damping,
    }


def biomechanical_objective(
    x: "torch.Tensor",
    n_steps: int = 200,
    lambda_tv: float = 0.0,
) -> "torch.Tensor":
    """
    Scalar objective for BO.
    x columns: [angle_init, raideur, damping, frequence].
    Returns mean pain per sample with shape [batch, 1].
    """
    stress = biomechanical_model(x, n_steps=n_steps)
    pain = pain_from_stress(stress)
    pain_mean = pain.mean(dim=1, keepdim=True)

    # Mild regularization to avoid unstable high-frequency solutions.
    frequence = x[:, 3:4]
    stability_penalty = 0.02 * (frequence - 2.5) ** 2
    tv_value = total_variation_1d(stress).unsqueeze(-1) / max(n_steps - 1, 1)
    tv_penalty = lambda_tv * tv_value
    return pain_mean + stability_penalty + tv_penalty


def run_bayesian_optimization(
    bounds_np: np.ndarray,
    n_init: int,
    n_iter: int,
    num_restarts: int,
    raw_samples: int,
    lambda_tv: float,
    seed: int,
    progress_bar,
    status_text,
) -> dict:
    """Run GP+EI Bayesian optimization and return detailed traces."""
    torch.manual_seed(seed)
    bounds = torch.tensor(bounds_np, dtype=torch.double)
    dim = bounds.shape[1]
    unit_bounds = torch.stack(
        [torch.zeros(dim, dtype=torch.double), torch.ones(dim, dtype=torch.double)],
        dim=0,
    )

    def to_real(unit_x: "torch.Tensor") -> "torch.Tensor":
        return bounds[0] + unit_x * (bounds[1] - bounds[0])

    train_x = torch.rand(n_init, dim, dtype=torch.double)
    train_real = to_real(train_x)
    train_objective = biomechanical_objective(train_real, lambda_tv=lambda_tv)
    stress_init = biomechanical_model(train_real)
    train_pain = pain_from_stress(stress_init).mean(dim=1, keepdim=True)

    all_rows: list[dict] = []
    for idx in range(n_init):
        x_real = train_real[idx]
        all_rows.append(
            {
                "source": "init",
                "step": float(idx + 1),
                "angle_init": float(x_real[0].item()),
                "raideur": float(x_real[1].item()),
                "damping": float(x_real[2].item()),
                "frequence": float(x_real[3].item()),
                "objective": float(train_objective[idx, 0].item()),
                "pain_mean": float(train_pain[idx, 0].item()),
            }
        )

    iter_rows: list[dict] = []
    for iteration in range(1, n_iter + 1):
        y_std = train_objective.std().clamp_min(1e-9)
        y_norm = (train_objective - train_objective.mean()) / y_std

        gp = SingleTaskGP(train_x, y_norm)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        ei = ExpectedImprovement(model=gp, best_f=y_norm.min().item(), maximize=False)
        candidate_unit, acq_value = optimize_acqf(
            ei,
            bounds=unit_bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        candidate_real = to_real(candidate_unit)
        new_objective = biomechanical_objective(candidate_real, lambda_tv=lambda_tv)
        new_pain = pain_from_stress(biomechanical_model(candidate_real)).mean(dim=1, keepdim=True)
        train_x = torch.cat([train_x, candidate_unit], dim=0)
        train_objective = torch.cat([train_objective, new_objective], dim=0)
        train_pain = torch.cat([train_pain, new_pain], dim=0)

        best_idx = int(torch.argmin(train_objective[:, 0]))
        iter_rows.append(
            {
                "iteration": float(iteration),
                "candidate_angle_init": float(candidate_real[0, 0].item()),
                "candidate_raideur": float(candidate_real[0, 1].item()),
                "candidate_damping": float(candidate_real[0, 2].item()),
                "candidate_frequence": float(candidate_real[0, 3].item()),
                "candidate_objective": float(new_objective[0, 0].item()),
                "candidate_pain_mean": float(new_pain[0, 0].item()),
                "acq_value": float(acq_value.item()),
                "best_objective": float(train_objective[best_idx, 0].item()),
                "best_pain_mean": float(train_pain[best_idx, 0].item()),
            }
        )

        all_rows.append(
            {
                "source": "bo",
                "step": float(n_init + iteration),
                "angle_init": float(candidate_real[0, 0].item()),
                "raideur": float(candidate_real[0, 1].item()),
                "damping": float(candidate_real[0, 2].item()),
                "frequence": float(candidate_real[0, 3].item()),
                "objective": float(new_objective[0, 0].item()),
                "pain_mean": float(new_pain[0, 0].item()),
            }
        )

        progress_bar.progress(iteration / n_iter)
        status_text.text(
            f"Iteration {iteration}/{n_iter} - GP + EI en cours (lambda_TV={lambda_tv:.3f})..."
        )

    objective_values = to_numpy(train_objective[:, 0])
    pain_values = to_numpy(train_pain[:, 0])
    best_global_idx = int(np.argmin(objective_values))
    best_params = to_numpy(to_real(train_x[best_global_idx : best_global_idx + 1])[0])
    best_objective = float(objective_values[best_global_idx])
    best_pain_at_best = float(pain_values[best_global_idx])

    all_objective = np.array([row["objective"] for row in all_rows], dtype=float)
    all_pain_mean = np.array([row["pain_mean"] for row in all_rows], dtype=float)
    best_so_far = np.minimum.accumulate(all_objective)
    steps = np.arange(1, len(all_objective) + 1, dtype=float)

    return {
        "all_rows": all_rows,
        "iter_rows": iter_rows,
        "steps": steps,
        "all_objective": all_objective,
        "all_pain_mean": all_pain_mean,
        "best_so_far": best_so_far,
        "best_params": best_params,
        "best_objective": best_objective,
        "best_pain_at_best": best_pain_at_best,
        "lambda_tv": float(lambda_tv),
    }


def render_series_chart(
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    line_style: str = "-",
    color: str = "b",
    fill: bool = False,
) -> None:
    """Render chart with matplotlib fallback to Streamlit line chart."""
    st.subheader(title)
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots()
        ax.plot(x, y, line_style, color=color, linewidth=2)
        if fill:
            ax.fill_between(x, y, alpha=0.25, color=color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.line_chart({"x": x, y_label: y}, x="x")
        st.info("`matplotlib` non installe: affichage avec chart natif Streamlit.")


def main() -> None:
    st.set_page_config(page_title="Biomech Optim", layout="wide")
    st.title("Optimisation Biomecanique - Genou et Douleur")

    with st.sidebar:
        st.header("Parametres du Modele")
        angle = st.slider("Angle initial (rad)", 0.1, 0.5, 0.3)
        raideur = st.slider("Raideur ligamentaire (N/m)", 5.0, 20.0, 10.0)
        damping = st.slider("Amortissement (N.s/m)", 0.1, 2.0, 0.5)
        frequence = st.slider("Frequence musculaire (Hz)", 1.0, 5.0, 2.0)

        if not TORCH_AVAILABLE:
            st.caption("Torch non installe: simulation executee en mode numpy.")

        launch = st.button("Lancer la simulation", use_container_width=True)
        if launch:
            with st.spinner("Calcul en cours..."):
                time, stress, pain = run_single_simulation(angle, raideur, damping, frequence)
            st.session_state["sim_time"] = time
            st.session_state["sim_stress"] = stress
            st.session_state["sim_pain"] = pain

    col1, col2 = st.columns(2)
    has_sim = "sim_stress" in st.session_state

    with col1:
        if has_sim:
            render_series_chart(
                title="Dynamique Articulaire",
                x=st.session_state["sim_time"],
                y=st.session_state["sim_stress"],
                x_label="Temps (s)",
                y_label="Contrainte mecanique",
                line_style="-",
                color="r",
            )
        else:
            st.subheader("Dynamique Articulaire")
            st.info("Clique sur 'Lancer la simulation' pour voir les resultats.")

    with col2:
        if has_sim:
            render_series_chart(
                title="Module Douleur Parametrique",
                x=st.session_state["sim_time"],
                y=st.session_state["sim_pain"],
                x_label="Temps (s)",
                y_label="Niveau de douleur",
                line_style="--",
                color="b",
                fill=True,
            )
        else:
            st.subheader("Module Douleur Parametrique")
            st.info("Clique sur 'Lancer la simulation' pour voir les resultats.")

    if has_sim:
        pain = st.session_state["sim_pain"]
        time = st.session_state["sim_time"]
        stress = st.session_state["sim_stress"]
        k1, k2, k3 = st.columns(3)
        k1.metric("Douleur moyenne", f"{pain.mean():.2f}")
        k2.metric("Douleur maximale", f"{pain.max():.2f}")
        k3.metric("Integrale douleur", f"{np.trapz(pain, time):.2f}")

        with st.expander("Regularisation BV/TV (Chambolle)", expanded=False):
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                noise_sigma = st.slider(
                    "Bruit sigma",
                    min_value=0.0,
                    max_value=1.5,
                    value=0.15,
                    step=0.01,
                )
            with t2:
                tv_weight = st.slider(
                    "Poids TV lambda",
                    min_value=0.01,
                    max_value=1.5,
                    value=0.15,
                    step=0.01,
                )
            with t3:
                tv_iters = st.slider(
                    "Iterations Chambolle",
                    min_value=20,
                    max_value=400,
                    value=80,
                    step=10,
                )
            with t4:
                noise_seed = st.number_input(
                    "Seed bruit",
                    min_value=0,
                    max_value=100000,
                    value=1234,
                    step=1,
                )

            rng = np.random.default_rng(int(noise_seed))
            noisy_stress = stress + rng.normal(0.0, noise_sigma, size=stress.shape)
            denoised_stress = chambolle_tv_denoise_1d(
                noisy_stress,
                weight=tv_weight,
                n_iter=int(tv_iters),
            )

            pain_noisy = to_numpy(pain_from_stress(noisy_stress)).reshape(-1)
            pain_denoised = to_numpy(pain_from_stress(denoised_stress)).reshape(-1)

            tv_noisy = float(total_variation_1d(noisy_stress))
            tv_denoised = float(total_variation_1d(denoised_stress))
            tv_gain_pct = 100.0 * (tv_noisy - tv_denoised) / max(tv_noisy, 1e-9)

            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots()
                ax.plot(time, stress, label="Stress propre", linewidth=1.8)
                ax.plot(time, noisy_stress, label="Stress bruite", alpha=0.55, linewidth=1.2)
                ax.plot(time, denoised_stress, label="Stress denoise (TV)", linewidth=2.2)
                ax.set_xlabel("Temps (s)")
                ax.set_ylabel("Contrainte")
                ax.set_title("Debruitage TV de Chambolle")
                ax.grid(True, alpha=0.25)
                ax.legend()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.line_chart(
                    {
                        "time": time,
                        "stress_propre": stress,
                        "stress_bruite": noisy_stress,
                        "stress_tv": denoised_stress,
                    },
                    x="time",
                )

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("TV(stress bruite)", f"{tv_noisy:.3f}")
            p2.metric("TV(stress denoise)", f"{tv_denoised:.3f}")
            p3.metric("Reduction TV", f"{tv_gain_pct:.1f}%")
            p4.metric("Gain douleur moyenne", f"{pain_noisy.mean() - pain_denoised.mean():.3f}")

    st.divider()
    st.subheader("Modelisation Continue (Neural ODE)")

    if not TORCH_AVAILABLE:
        st.warning("PyTorch indisponible. Installe `torch` pour activer cette section.")
    elif not TORCHDIFFEQ_AVAILABLE:
        st.warning("`torchdiffeq` indisponible. Installe `torchdiffeq` pour la simulation continue.")
    else:
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            position0 = st.slider("Position initiale", min_value=-1.0, max_value=1.0, value=0.3, step=0.01)
        with n2:
            velocity0 = st.slider("Vitesse initiale", min_value=-2.0, max_value=2.0, value=0.0, step=0.01)
        with n3:
            force0 = st.slider("Force initiale", min_value=-5.0, max_value=5.0, value=2.0, step=0.05)
        with n4:
            hidden_dim = st.slider("Hidden dim", min_value=16, max_value=256, value=64, step=16)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            duration = st.slider("Duree integration (s)", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
        with m2:
            n_steps = st.slider("Nombre de pas", min_value=20, max_value=400, value=120, step=20)
        with m3:
            method = st.selectbox("Solveur ODE", options=("dopri5", "rk4", "euler"), index=0)
        with m4:
            use_adjoint = st.checkbox("Adjoint method", value=True)

        st.caption("Modele non entraine: la correction neurale est initialisee a zero.")

        if st.button("Simuler Neural ODE", use_container_width=True):
            with st.spinner("Integration continue en cours..."):
                result = run_neural_ode_simulation(
                    position0=position0,
                    velocity0=velocity0,
                    force0=force0,
                    duration=duration,
                    n_steps=n_steps,
                    hidden_dim=hidden_dim,
                    method=method,
                    use_adjoint=use_adjoint,
                )
            st.session_state["neural_ode_result"] = result

        if "neural_ode_result" in st.session_state:
            result = st.session_state["neural_ode_result"]

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Max |position|", f"{np.max(np.abs(result['position'])):.3f}")
            r2.metric("Max |vitesse|", f"{np.max(np.abs(result['velocity'])):.3f}")
            r3.metric("Max |couple|", f"{np.max(np.abs(result['torque'])):.3f}")
            r4.metric("Force finale", f"{result['force'][-1]:.3f}")

            render_series_chart(
                title="Position continue",
                x=result["time"],
                y=result["position"],
                x_label="Temps (s)",
                y_label="Position",
                line_style="-",
                color="g",
            )
            render_series_chart(
                title="Vitesse continue",
                x=result["time"],
                y=result["velocity"],
                x_label="Temps (s)",
                y_label="Vitesse",
                line_style="-",
                color="orange",
            )
            render_series_chart(
                title="Couple estime",
                x=result["time"],
                y=result["torque"],
                x_label="Temps (s)",
                y_label="Couple",
                line_style="-",
                color="purple",
            )

    st.divider()
    st.subheader("Optimisation Bayesienne (GP + EI)")

    if not BOTORCH_AVAILABLE:
        st.warning(
            "BoTorch indisponible. Installe `torch`, `gpytorch` et `botorch` pour activer cette section."
        )
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            n_init = st.slider("Points initiaux", min_value=4, max_value=40, value=10, step=1)
        with c2:
            n_iter = st.slider("Iterations BO", min_value=1, max_value=60, value=15, step=1)
        with c3:
            num_restarts = st.slider("EI num_restarts", min_value=2, max_value=30, value=10, step=1)
        with c4:
            raw_samples = st.slider("EI raw_samples", min_value=16, max_value=512, value=128, step=16)
        with c5:
            lambda_tv = st.slider(
                "Penalite TV lambda",
                min_value=0.0,
                max_value=0.2,
                value=0.02,
                step=0.005,
            )
        seed = st.number_input("Seed", min_value=0, max_value=100000, value=1234, step=1)

        if st.button("Demarrer l'optimisation", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            bounds = np.array(
                [
                    [0.1, 5.0, 0.1, 1.0],
                    [0.5, 20.0, 2.0, 5.0],
                ],
                dtype=float,
            )
            result = run_bayesian_optimization(
                bounds_np=bounds,
                n_init=n_init,
                n_iter=n_iter,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                lambda_tv=lambda_tv,
                seed=int(seed),
                progress_bar=progress_bar,
                status_text=status_text,
            )
            st.session_state["bo_result"] = result

        if "bo_result" in st.session_state:
            result = st.session_state["bo_result"]
            best_params = result["best_params"]
            best_objective = result["best_objective"]
            best_pain_at_best = result["best_pain_at_best"]

            st.caption(
                f"Objectif: douleur moyenne + penalite stabilite frequence + "
                f"penalite TV (lambda={result['lambda_tv']:.3f})."
            )

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Angle*", f"{best_params[0]:.3f}")
            m2.metric("Raideur*", f"{best_params[1]:.3f}")
            m3.metric("Damping*", f"{best_params[2]:.3f}")
            m4.metric("Frequence*", f"{best_params[3]:.3f}")
            m5.metric("Objectif min", f"{best_objective:.5f}")
            m6.metric("Douleur moy. au best", f"{best_pain_at_best:.5f}")

            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots()
                ax.plot(result["steps"], result["all_objective"], "o-", linewidth=1.8, label="Objectif")
                ax.plot(result["steps"], result["all_pain_mean"], "--", linewidth=1.5, label="Pain mean")
                ax.plot(result["steps"], result["best_so_far"], "-", linewidth=2.0, label="Best objective")
                ax.set_xlabel("Evaluation")
                ax.set_ylabel("Valeur")
                ax.set_title("Convergence BO (GP + EI)")
                ax.grid(True, alpha=0.25)
                ax.legend()
                st.pyplot(fig, use_container_width=True)
                fig.savefig(CONVERGENCE_PNG, dpi=150, bbox_inches="tight")
                plt.close(fig)
                if CONVERGENCE_PNG.exists():
                    st.download_button(
                        "Telecharger convergence.png",
                        data=CONVERGENCE_PNG.read_bytes(),
                        file_name="convergence.png",
                        mime="image/png",
                    )
            else:
                st.line_chart(
                    {
                        "evaluation": result["steps"],
                        "objective": result["all_objective"],
                        "pain_mean": result["all_pain_mean"],
                        "best_so_far": result["best_so_far"],
                    },
                    x="evaluation",
                )

            st.subheader("Points evalues")
            st.dataframe(result["all_rows"], use_container_width=True)
            st.download_button(
                "Telecharger points BO (CSV)",
                data=rows_to_csv_bytes(result["all_rows"]),
                file_name="points_bo.csv",
                mime="text/csv",
            )

            st.subheader("Historique des iterations BO")
            st.dataframe(result["iter_rows"], use_container_width=True)
            st.download_button(
                "Telecharger historique BO (CSV)",
                data=rows_to_csv_bytes(result["iter_rows"]),
                file_name="historique_bo.csv",
                mime="text/csv",
            )

    st.divider()
    st.caption("Projet Biomecanique | Version scientifique complete")


if __name__ == "__main__":
    main()
