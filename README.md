# spatial-FI-reg
Spatially weighted Fisher-information regularization of Benamou-Brenier action for robust, edge-aware dynamic optimal transport

## Abstract
In Li, Yin, and Osher's formulation, Fisher information introduces smoothness, positivity, and strict convexity, enabling fast and accurate algorithms such as Newton's Method. We propose a heterogeneously weighted Fisher information regularization on dynamic optimal transport. After discretizing the continuous formulation that originated from Benamou and Brenier over a space-time graph with a set of nodes, we multiply the discrete Fisher information term by an anisotropic weight that includes a positive floor to maintain interior iterates and a Perona-Malik diffusion. This increases the weight on regions with lower density variation and decreases it on regions with sharp density changes between nodes, thereby identifying edge sets and reducing blur. This formulation is benchmarked against the previously unweighted regularized transport and the unregularized Benamou-Brenier formulation, and numerical experiments demonstrate improvements in edge preservation and stability. Edge-preserving transport paths like this one are extremely relevant to medical imaging, microscopy, and remote sensing, where boundaries are critical.

## Formulation
$$\min_{m,p}\ \sum_{l=0}^{L} \sum_{i+\frac{e_v}{2}\in E}\; \underbrace{\frac{m_{i+\frac{e_v}2,l}^2}{g_{i+\frac{e_v}2,l}}}_{\textcolor{red}{\text{Dynamic term}}} + \underbrace{\frac{\beta^2 \textcolor{OliveGreen}{w_{i + \frac{e_v}{2}}}}{\Delta x^2} (\log p_{i,l} - \log p_{i+e_v,l})^2g_{i+\frac{e_v}2,l}}_\textcolor{blue}{\text{Fisher information regularization}}$$

subject to

$$p_{i,l} > 0\\
    \frac{p_{i,l+1} - p_{i,l}}{\Delta t} + (\nabla\cdot m_{l})_i = 0, \qquad l=0,\dots,L-1;\\
    p_{i,0} = (p_0)_i,\qquad p_{i,L} = (p_1)_i,\qquad i\in V.$$

where

$$w_{i+\frac{e_v}{2}} = \varepsilon + (1-\varepsilon)\exp\!\left(-\left(\frac{\eta_{i+\frac{e_v}{2}}}{\kappa}\right)^2\right)$$

with the weight floor $\varepsilon$, $\eta_{i+\frac{e_v}{2}}$ as the edge map of the differences between nodes on the grid, and $\kappa$ as the median of nonzero values of $\eta$.
