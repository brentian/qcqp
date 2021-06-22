
# Setups

- we wish to combine small cones and the SDP
- we wish to establish 
- let $I_+$ be indices of positive eigenvalues, suppose $|I_+| = p$

Spectral decomposition, 

$$Q = V\Lambda V^T$$

where $V$ is the orthonormal basis and $\Lambda$ is the diagonal matrix constructed from eigenvalues $\lambda = [\lambda_1, ..., \lambda_n]$.

For simplicity, we let:



# Relaxations

> Shor 

$$\begin{aligned}
   \mathrm{Maximize}\quad & Q\bullet Y + q^Tx                          \\
    \mathrm{s.t.} \quad  & Y \ge xx^T \\
            & \mathrm{diag}(Y) \le x \\
            & 0\le x\le 1
  \end{aligned}$$

> Basic MSC

$$U = V\sqrt{|\Lambda|)}$$

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

- The last inequality,
$$y^T\frac{1}{|\lambda|} = z^T|\Lambda|^{-1}z = x^T x \le x^Te$$
- Also one can see that box constraints for $z$ is unnecessary.

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

<!-- > MSC + Simple SDP

Let optimal value be $v^\mathrm{SMSC}$, let $Y$ be a semi-definite matrix of dimension $p$,

$$\begin{aligned}
   (\mathrm{SMSC}) \quad\mathrm{Maximize}\quad & y ^T\lambda + q^Tx                          \\
    \mathrm{s.t.} \quad              & Vz = x                          \\
                                     & (y_i, z_i) \in SOC                & i \notin I_+ \\
                                     & Y \succeq z_+ z_+^T   \\ 
                                     & \mathrm{diag}(Y) = y_+ \\
                                     & y^T e \le x^Te
  \end{aligned}$$

This is equivalent to DMSC... -->


> MSC + Shor


Partition also the orthonormal basis into into $V_+ \in R^{n\times p}, V_-\in R^{n\times (n-p)}$

Note that if we define $V_+ z_+ = x_+, V_- z_- = x_-$, then it reduces to the following partition for $x$,

$$x_+^Tx_- = 0,\; x_+ + x_- = x$$

since $V_+^TV_- = 0$, we have,

$$x_+^TV_- = 0, x_-^TV_+ = 0$$


Consider a SD matrix $Y\in R^{p\times p}$ to estimate $z_+ z_+^T$, we have, 
$$\begin{aligned}
&V_+YV_+^T = x_+x_+^T\\
& \mathrm{diag}(V_+ Y V_+^T) = y_+ \\
\end{aligned}$$

By using SOCP for convex part $(y_-)_i \ge (x_-)^2_i$, the quadratic function, 

$$\begin{aligned}
x^TQx &= (x_- +  x_+)^T (V_+ \Lambda_+ V_+^T + V_- \Lambda_- V_-^T) (x_- +  x_+) \\
      &= x_+^TV_+ \Lambda_+ V_+^Tx_+ + x_-^TV_- \Lambda_- V_-^Tx_- \\
      &= \Lambda_+ \bullet Y + \lambda_-^T y_-
\end{aligned}
$$

This allows the following SOCP + SDP relaxation:

$$\begin{aligned}
   (\mathrm{SMSC}) \quad\mathrm{Maximize}\quad & Y \bullet \Lambda_ + -   + q^Tx                          \\
    \mathrm{s.t.} \quad              & x_+ + x_- = x                         \\
                                     & V_+^T x_- = \bm 0 \\
                                     & (y_-, V_-^Tx_-)_i \in SOC                & i = 1, ..., n \\
                                     & \begin{bmatrix} Y & V_+^Tx_+\\ x_+^TV_+ & 1 \end{bmatrix} \succeq \bm 0\\
                                     & V_-^T Y = \bm 0 \\
                                    % & \mathrm{diag}(V_+ Y V_+^T) + y_- \le x \\ 
  \end{aligned}$$
 
## Strength

#### Proposition 1.
$$v^\mathrm{Shor} \le v^\mathrm{MSC} = v^\mathrm{DMSC} = v^\mathrm{SMSC}$$

One can see the key to improve the relaxation is to find a bound for $\mathrm{diag}(Y)$

We can see now doing diagonalization alone cannot improve the bound.


# Potential try

<!-- 
#### Partition $x$ into $x_+ + x_- =x$

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
  \end{aligned}$$ -->

