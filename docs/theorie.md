# Partie theorique

Ce document decrit ce qui est execute dans l'application:

1. simulation biomecanique parametrique;
2. regularisation BV/TV (Chambolle) sur signaux 1D;
3. simulation continue optionnelle via Neural ODE;
4. optimisation bayesienne des parametres (GP + EI).

## 1. Variables et bornes

Le vecteur de parametres est:

\[
x = [\theta_0,\; k,\; c,\; f]
\]

avec:

- \(\theta_0\): angle initial;
- \(k\): raideur ligamentaire;
- \(c\): amortissement;
- \(f\): frequence musculaire effective (Hz).

Bornes utilisees:

\[
\theta_0\in[0.1,0.5],\quad
k\in[5,20],\quad
c\in[0.1,2],\quad
f\in[1,5].
\]

## 2. Simulation biomecanique parametrique

Pour \(t_j\in[0,1]\), \(j=1,\dots,N\):

\[
stress_j = k\theta_0 + c\sin(2\pi f t_j).
\]

La douleur est definie autour d'un stress cible \(s^\star\) (\(s^\star=3.0\) dans le code):

\[
\Delta_j = |stress_j - s^\star|
\]

\[
pain_j = 0.6\,\Delta_j + 0.4\,\exp(0.35\,\Delta_j).
\]

Objectif scalaire:

\[
\bar{p}(x)=\frac{1}{N}\sum_{j=1}^{N} pain_j.
\]

Penalisation douce de stabilite:

\[
R_{freq}(x)=0.02\,(f-2.5)^2.
\]

Penalisation BV/TV normalisee:

\[
TV(stress)=\sum_{j=1}^{N-1}|stress_{j+1}-stress_j|,
\]

\[
R_{TV}(x)=\lambda_{TV}\,\frac{TV(stress)}{N-1}.
\]

Objectif optimise:

\[
f_{obj}(x)=\bar{p}(x)+R_{freq}(x)+R_{TV}(x),\qquad
\min_{x\in\mathcal{X}} f_{obj}(x).
\]

## 3. Regularisation TV (algorithme de Chambolle)

Sur un signal bruité \(y\in\mathbb{R}^N\), on estime \(u\) via:

\[
\min_u \frac{1}{2}\|u-y\|_2^2 + \lambda\,TV(u).
\]

Version 1D (difference avant):

\[
(Du)_i=u_{i+1}-u_i,\quad TV(u)=\sum_i |(Du)_i|.
\]

L'algorithme dual de Chambolle met a jour \(p\) puis reconstruit:

\[
u = y - \lambda D^\top p,
\]

\[
p \leftarrow \frac{p + (\tau/\lambda)\,Du}{\max(1,\;|p + (\tau/\lambda)\,Du|)}.
\]

Dans l'application, cette brique est utilisee pour comparer:

- stress propre;
- stress bruité;
- stress denoise TV.

Et mesurer la reduction de variation totale.

## 4. Modele continu (Neural ODE)

La section continue (si `torchdiffeq` est disponible) integre un etat:

\[
z(t)=[p(t),\,v(t),\,u(t)]
\]

ou \(p\) est la position, \(v\) la vitesse, \(u\) une force effective.

Le modele dynamique est:

\[
\dot{p}=v
\]

\[
\dot{v}=\frac{-k_p p-c_p v+u}{m}+\delta_\phi(p,v,u)
\]

\[
\dot{u}=g_\phi(p,v,u)
\]

avec:

- \(m,k_p,c_p\) parametres physiologiques apprenables;
- \(\delta_\phi,g_\phi\) corrections neurales (MLP).

L'integration temporelle utilise `odeint` ou `odeint_adjoint`.

## 5. Optimisation bayesienne (GP + EI)

### 5.1 Espace normalise

La BO est menee en espace unitaire:

\[
u\in[0,1]^4,\qquad x=l+u\odot(h-l).
\]

### 5.2 Surrogate GP

On ajuste `SingleTaskGP` sur:

- entree: \(u_i\in[0,1]^4\);
- sortie normalisee:

\[
\tilde{y}_i=\frac{y_i-\mu_y}{\sigma_y}.
\]

### 5.3 Acquisition EI

Le candidat est choisi par:

\[
u_{n+1}=\arg\max_{u\in[0,1]^4}EI(u),
\]

avec `maximize=False` et:

\[
best\_f=\min_i \tilde{y}_i.
\]

Puis:

1. conversion \(u_{n+1}\to x_{n+1}\);
2. evaluation \(y_{n+1}=f_{obj}(x_{n+1})\);
3. ajout au dataset.

### 5.4 Selection finale

\[
x^\star=\arg\min_i y_i,\qquad
best\_so\_far(n)=\min_{i\le n}y_i.
\]

## 6. Sorties de l'application

- courbes \(stress(t)\), \(pain(t)\) pour une simulation parametrique;
- module BV/TV: denoising de Chambolle + metriques TV;
- courbes continues \(p(t),v(t),\tau(t)\) pour le Neural ODE;
- KPI (moyenne, max, integrale douleur, amplitude dynamique);
- historique BO et meilleur parametre \((\theta_0,k,c,f)\).

## 7. Limites

- modele phenomenologique non calibre patient-specifique;
- Neural ODE non entraine sur donnees cliniques dans cette version;
- BO mono-objectif sans contraintes cliniques explicites;
- \(\lambda_{TV}\) est un hyperparametre (pas encore estime depuis donnees reelles).
