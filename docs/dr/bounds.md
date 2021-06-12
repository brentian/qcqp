
# Bounds

- we wish to combine small cones and the SDP
- we wish to establish 
- let $I_+$ be indices of positive eigenvalues,

Spectral decomposition, 

$$Q = V\mathsf{diag}(\lambda)V^T$$

where $V$ is the orthonormal basis and $\mathsf{diag}(\lambda)$ is the diagonal matrix constructed from eigenvalues $\lambda = [\lambda_1, ..., \lambda_n]$.


## Many-small-cone (MSC) relaxations


> Shor 

$$\begin{aligned}
   \mathrm{Maximize}\quad & Q\bullet Y + q^Tx                          \\
    \mathrm{s.t.} \quad  & Y \ge xx^T \\
            & \mathsf{diag}(Y) \le x
  \end{aligned}$$

> Basic 

Let optimal value be $v^\mathrm{MSC}$

$$U = V\sqrt{\mathsf{diag}(|\lambda|)}$$

$$\begin{aligned}
   \mathrm{Maximize}\quad & y ^T \mathsf{sign(\lambda)} + q^Tx                 \\
    \mathrm{s.t.} \quad   & U^Tx=z \\
                         & (y_i, z_i) \in SOC              & i = 1, ..., n     \\
                       & y^T\frac{1}{|\lambda|} \le x^Te                         \\
                                
  \end{aligned}$$

The last inequality,

$$y^T\frac{1}{|\lambda|} = z^T\mathsf{diag}(|\lambda|)^{-1}z = x^T x \le x^Te$$

> Diagonalized (DMSC) 

Let optimal value be $v^\mathrm{DMSC}$

$$\begin{aligned}
   \mathrm{Maximize}\quad & y ^T\lambda + q^Tx                          \\
    \mathrm{s.t.} \quad              & Vz = x                          \\
                                     
                                     & (y_i, z_i) \in SOC                & i = 1, ..., n \\
                                     & y^T e \le x^Te
  \end{aligned}$$

> Proposition 1. We show that $v^\mathrm{Shor} \le v^\mathrm{MSC} = v^\mathrm{DMSC}$


We first show the equivalence of MSC and DMSC.

> PF. 

Suppose $(x, z, y)$ is a feasible solution to DMSC. then $(x, \sqrt{\mathsf{diag}(|\lambda|)} z, \mathsf{diag}(|\lambda|)\cdot y)$ is feasible to MSC.


<!-- 
Consider full SDP:


$$
\begin{aligned}
Y \succeq V^Txx^TV\\
VYV^T \succeq xx^T \\
    X \succ 0 \Leftrightarrow A \succ 0, X / A=C-B^{\top} A^{-1} B \succ 0
\end{aligned}
$$ -->