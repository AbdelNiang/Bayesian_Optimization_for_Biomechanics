# Optimisation--genou

Application Streamlit pour:

- simuler la dynamique biomécanique du genou;
- appliquer un denoising BV/TV (algorithme de Chambolle) sur les signaux;
- simuler une dynamique continue via Neural ODE (torchdiffeq);
- estimer un profil de douleur paramétrique;
- optimiser les paramètres par Bayesian Optimization (GP + EI, BoTorch).

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run main.py
```

## Validation rapide

```bash
make test
make docs
```

## Ce qui est exécuté

1. Simulation locale avec paramètres \((angle, raideur, damping, frequence)\).
2. Simulation continue optionnelle (Neural ODE) avec intégration temporelle.
3. Calcul des trajectoires `stress(t)` puis `pain(t)`.
4. Optimisation bayésienne des 4 paramètres avec:
   - `SingleTaskGP`
   - `ExpectedImprovement`
   - `optimize_acqf`

Notes de modelisation:

- le 4e parametre est une frequence effective (Hz), pas une force;
- les entrees BO sont normalisees en \([0,1]^4\) pour stabiliser le GP;
- la douleur est evaluee autour d'un stress cible (non triviale aux bornes).
- l'objectif BO inclut une penalite TV configurable (\(\lambda_{TV}\));
- la section Neural ODE requiert `torchdiffeq`.

## Structure

```text
optimisation--genou/
|-- main.py                    # UI Streamlit + BO (GP+EI)
|-- optimization_pipeline.py   # Stress, douleur, TV/BV et denoising Chambolle
|-- models/neural_ode.py       # Dynamique continue (Neural ODE)
|-- requirements.txt
|-- docs/
|   |-- theorie.md
|   |-- theorie.html
|   `-- theorie.pdf
`-- Makefile
```
