
Some ways to retrieve primal solutions

$$\tilde X = \begin{bmatrix}
    X & x \\
    x^T & 1
\end{bmatrix}$$

if $r(\tilde X) = 1$ then $x$ solves the original problem.



## Rank reduction
```
1. Anthony Man-Cho So, Yinyu Ye, Jiawei Zhang, (2008) A Unified Theorem on SDP Rank Reduction. Mathematics of Operations Research
```
- Using $\xi \sim N(0, 1/d)$, and notice $\mathbb E \sum_d \xi_j \xi_j$
- $r(S X) \le \min (r(S), r(X))$, then the resulting matrix is of maximum rank-$d$

$$\hat{\mathbf{X}}=\left(\mathbf{V}^{*}\right)^{T}\left[\sum_{j=1}^{d} \xi^{j}\left(\xi^{j}\right)^{T}\right] \mathbf{V}^{*}$$




steps,


