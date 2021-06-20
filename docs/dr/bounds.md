
# Bounds

- we wish to combine small cones and the SDP
- we wish to establish 
- let $I_+$ be indices of positive eigenvalues, suppose $|I_+| = p$

Spectral decomposition, 

$$Q = V\mathrm{diag}(\lambda)V^T$$

where $V$ is the orthonormal basis and $\mathrm{diag}(\lambda)$ is the diagonal matrix constructed from eigenvalues $\lambda = [\lambda_1, ..., \lambda_n]$.


# Relaxations

> Shor 

$$\begin{aligned}
   \mathrm{Maximize}\quad & Q\bullet Y + q^Tx                          \\
    \mathrm{s.t.} \quad  & Y \ge xx^T \\
            & \mathrm{diag}(Y) \le x \\
            & 0\le x\le 1
  \end{aligned}$$

> Basic MSC

$$U = V\sqrt{\mathrm{diag}(|\lambda|)}$$

$$\begin{aligned}
   \mathrm{Maximize}\quad & y ^T \mathrm{sign(\lambda)} + q^Tx                 \\
    \mathrm{s.t.} \quad   & U^Tx=z \\
                         & (y_i, z_i) \in SOC              & i = 1, ..., n     \\
                       & l \le z \le u
                                
  \end{aligned}$$

One can derive box constraints for $z, y$ by the definition and bounds for $x$.

> Strengthened MSC

Let optimal value be $v^\mathrm{MSC}$

$$U = V\sqrt{\mathrm{diag}(|\lambda|)}$$

$$\begin{aligned}
   (\mathrm{MSC}) \quad  \mathrm{Maximize}\quad & y ^T \mathrm{sign(\lambda)} + q^Tx                 \\
    \mathrm{s.t.} \quad   & U^Tx=z \\
                         & (y_i, z_i) \in SOC              & i = 1, ..., n     \\
                       & y^T\frac{1}{|\lambda|} \le x^Te                         \\
                                
  \end{aligned}$$

The last inequality,

$$y^T\frac{1}{|\lambda|} = z^T\mathrm{diag}(|\lambda|)^{-1}z = x^T x \le x^Te$$

> Diagonalized (DMSC) 

Let optimal value be $v^\mathrm{DMSC}$

$$\begin{aligned}
   (\mathrm{DMSC}) \quad \mathrm{Maximize}\quad & y ^T\lambda + q^Tx                          \\
    \mathrm{s.t.} \quad              & Vz = x                          \\
                                     
                                     & (y_i, z_i) \in SOC                & i = 1, ..., n \\
                                     & y^T e \le x^Te
  \end{aligned}$$

One can see the equivalence of MSC and DMSC. Suppose $(x_0, z_0, y_0)$ is a feasible solution to DMSC. then $(x_0, \sqrt{\mathrm{diag}(|\lambda|)} z_0, \mathrm{diag}(|\lambda|)\cdot y_0)$ is feasible to MSC. Conversely, if $(x_0, z_0, y_0)$ is feasible to MSC, then $(x_0, \frac{1}{\sqrt{\mathrm{diag}(|\lambda|)}} z_0, \frac{1}{\mathrm{diag}(|\lambda|)}\cdot y_0)$ is feasible to DMSC.

- For DMSC, Strengthened MSC, the bounds for $z, y$ is not needed.

> MSC + Shor

Let optimal value be $v^\mathrm{SMSC}$, let $Y$ be a semi-definite matrix of dimension $p$,

$$\begin{aligned}
   (\mathrm{SMSC}) \quad\mathrm{Maximize}\quad & y ^T\lambda + q^Tx                          \\
    \mathrm{s.t.} \quad              & Vz = x                          \\
                                     & (y_i, z_i) \in SOC                & i \notin I_+ \\
                                     & Y \succeq z_+ z_+^T   \\ 
                                     & \mathrm{diag}(Y) = y_+ \\
                                     & y^T e \le x^Te
  \end{aligned}$$

This is equivalent to DMSC...

$$\begin{aligned}
&V_+YV_+^T = x_+x_+^T\\
&V_-z_- =x_-
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

One can see the key to improve the relaxation is to find a bound for $\mathrm{diag}(Y)$

# Potential try

#### Partition 

Partition orthonormal basis into into $V_+, V_-$, let $V_+ z = x_+, V_- z = x_-$

since $V_+^TV_- = 0$, we have,

$$x_+^Tx_- = 0,\; x_+ + x_- = x$$

also, suppose $Y \succeq x_+ x_+^T$, then $V_-^Tx_+ = 0,  V_-^T Y=0$

$$\begin{aligned}
   (\mathrm{SMSC2}) \quad\mathrm{Maximize}\quad & y ^T\lambda + q^Tx                          \\
    \mathrm{s.t.} \quad              & x_+ + x_- = Vz \\
                                     & (y_i, z_i) \in SOC                & i = 1, ..., n \\
                                     & Y \succeq x_+ x_+^T   \\ 
                                     & V_-^T Y = 0 \\
                                     & y^T e \le x^Te \\
                                     & \mathrm{diag}(V_+^TYV_+) = y_+
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



