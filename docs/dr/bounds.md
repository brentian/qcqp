
# Bounds

- we wish to combine small cones and the SDP
- we wish to establish 
- let $I_+$ be indices of positive eigenvalues, suppose $|I_+| = p$

Spectral decomposition, 

$$Q = V\mathsf{diag}(\lambda)V^T$$

where $V$ is the orthonormal basis and $\mathsf{diag}(\lambda)$ is the diagonal matrix constructed from eigenvalues $\lambda = [\lambda_1, ..., \lambda_n]$.


# Relaxations

> Shor 

$$\begin{aligned}
   \mathrm{Maximize}\quad & Q\bullet Y + q^Tx                          \\
    \mathrm{s.t.} \quad  & Y \ge xx^T \\
            & \mathsf{diag}(Y) \le x \\
            & 0\le x\le 1
  \end{aligned}$$

> Basic MSC

$$U = V\sqrt{\mathsf{diag}(|\lambda|)}$$

$$\begin{aligned}
   \mathrm{Maximize}\quad & y ^T \mathsf{sign(\lambda)} + q^Tx                 \\
    \mathrm{s.t.} \quad   & U^Tx=z \\
                         & (y_i, z_i) \in SOC              & i = 1, ..., n     \\
                       & l \le z \le u
                                
  \end{aligned}$$

One can derive box constraints for $z, y$ by the definition and bounds for $x$.

> Strengthened MSC

Let optimal value be $v^\mathrm{MSC}$

$$U = V\sqrt{\mathsf{diag}(|\lambda|)}$$

$$\begin{aligned}
   (\mathrm{MSC}) \quad  \mathrm{Maximize}\quad & y ^T \mathsf{sign(\lambda)} + q^Tx                 \\
    \mathrm{s.t.} \quad   & U^Tx=z \\
                         & (y_i, z_i) \in SOC              & i = 1, ..., n     \\
                       & y^T\frac{1}{|\lambda|} \le x^Te                         \\
                                
  \end{aligned}$$

The last inequality,

$$y^T\frac{1}{|\lambda|} = z^T\mathsf{diag}(|\lambda|)^{-1}z = x^T x \le x^Te$$

> Diagonalized (DMSC) 

Let optimal value be $v^\mathrm{DMSC}$

$$\begin{aligned}
   (\mathrm{DMSC}) \quad \mathrm{Maximize}\quad & y ^T\lambda + q^Tx                          \\
    \mathrm{s.t.} \quad              & Vz = x                          \\
                                     
                                     & (y_i, z_i) \in SOC                & i = 1, ..., n \\
                                     & y^T e \le x^Te
  \end{aligned}$$

One can see the equivalence of MSC and DMSC. Suppose $(x_0, z_0, y_0)$ is a feasible solution to DMSC. then $(x_0, \sqrt{\mathsf{diag}(|\lambda|)} z_0, \mathsf{diag}(|\lambda|)\cdot y_0)$ is feasible to MSC. Conversely, if $(x_0, z_0, y_0)$ is feasible to MSC, then $(x_0, \frac{1}{\sqrt{\mathsf{diag}(|\lambda|)}} z_0, \frac{1}{\mathsf{diag}(|\lambda|)}\cdot y_0)$ is feasible to DMSC.

- For DMSC, Strengthened MSC, the bounds for $z, y$ is not needed.

> MSC + Shor

Let optimal value be $v^\mathrm{MSC-Shor}$, let $Y$ be a semi-definite matrix of dimension $p$,

$$\begin{aligned}
   (\mathrm{SMSC}) \quad\mathrm{Maximize}\quad & y ^T\lambda + q^Tx                          \\
    \mathrm{s.t.} \quad              & Vz = x                          \\
                                     & (y_i, z_i) \in SOC                & i \notin I_+ \\
                                     & Y \succeq z_+ z_+^T   \\ 
                                     & \mathsf{diag}(Y) = y \\
                                     & y^T e \le x^Te
  \end{aligned}$$

This is equivalent to DMSC 

Alternatively,

$$\begin{aligned}
   (\mathrm{SMSC2}) \quad\mathrm{Maximize}\quad & y ^T\lambda + q^Tx                          \\
    \mathrm{s.t.} \quad              & Vz = x                          \\
                                     & V_+ z = x_+ \\
                                     & (y_i, z_i) \in SOC                & i \notin I_+ \\
                                     & Y \succeq z_+ z_+^T   \\ 
                                     & \mathsf{diag}(Y) = y \\
                                     & y^T e \le x^Te
  \end{aligned}$$

<!-- 
> Partition MSC

Partition orthonormal basis into into $V_+, V_-$, let $V_+ z = x_+, V_- z = x_-$

since $V_+^TV_- = 0$, we have,

$$x_+^Tx_- = 0, x_+ + x_- = x$$

$$\begin{aligned}

   (\mathrm{PMSC}) \quad\mathrm{Maximize}\quad & y ^T\lambda + q^Tx                          \\
    \mathrm{s.t.} \quad              & x_+ + x_- = x                         \\
                                     & V_+ z = x_+, V_- z = x_- \\
                                     & (y_+, x_+)_i \in SOC                & i = 1, ..., n \\
                                     & (y_-, x_-)_i \in SOC                & i = 1, ..., n \\
                                     & (y, z)_i \in SOC              & i = 1, ..., n     \\
                                     & y_-^Te + y_+ \le x
  \end{aligned}$$ -->


## Strength


#### Proposition 1.
$$v^\mathrm{Shor} \le v^\mathrm{MSC} = v^\mathrm{DMSC} = v^\mathrm{SMSC}$$




# Some unknown use

#### Partition 

Partition orthonormal basis into into $V_+, V_-$, let $V_+ z = x_+, V_- z = x_-$

since $V_+^TV_- = 0$, we have,

$$x_+^Tx_- = 0, x_+ + x_- = x$$

also,

$$
\begin{aligned}
&Q = V_+ \Gamma_+ V_+^T + V_- \Gamma_- V_-^T\\
&x^TQx = (x_+ + x_-)^T (V_+ \Gamma_+ V_+^T + V_- \Gamma_- V_-^T) (x_+ + x_-) \\
=& x_+ ^T V_+ \Gamma_+ V_+^T x_+ + x_- ^T V_- \Gamma_- V_-^T x_-
\end{aligned}$$




<!-- 
Consider full SDP:


$$
\begin{aligned}
Y \succeq V^Txx^TV\\
VYV^T \succeq xx^T \\
    X \succ 0 \Leftrightarrow A \succ 0, X / A=C-B^{\top} A^{-1} B \succ 0
\end{aligned}
$$ -->



