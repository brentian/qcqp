

- chord and chordal graph: an undirected graph $G$ is called chordal (or **triangulated**, or a rigid circuit) if every cycle of length greater than or equal to 4 has at least one chord.
  - non-chordal graphs can always be chordal extended, i.e., extended to a chordal graph, by adding additional edges to the original graph

- ordering: an ordering $\sigma \{1, 2, ... , n\} \Rightarrow V$ can also be interpreted as a sequence of vertices $\sigma = (\sigma(1), ... , \sigma(n))$. We refer to $\sigma^{−1} (v)$ as the index of vertex $v$ of such ordering.
  - denote a ordered graph as $G_\sigma$


- clique: a clique is a set of vertices $W \subseteq V$ that induces a maximal complete subgraph
  - a vertex $v$ of an undirected graph is simplicial if its neighborhood $adj(v)$ is complete.

> Stable set and clique cover

- stable set: set $S \subseteq V$ is a stable or independent set of an undirected graph $G = (V, E)$ if no two vertices in $S$ are adjacent, i.e., the induced edge set $E(S) = \{(v, w) \in E | v, w \in S\}$ is empty. The size of the largest stable set is called the stable set number of the graph, denoted $\alpha(G)$. 

- clique cover: a clique cover $\mathcal C$ of $G$ is a set of cliques that cover the vertex set $V$. The clique cover number $\bar \chi (G)$ is the minimum number of cliques in a clique cover,

$$\begin{aligned}
& V \subseteq \mathcal C \equiv \bigcup_k C_k \\
& \bar \chi (G) = \min_\mathcal{C} |\mathcal C|
\end{aligned}$$

It's obvious to see,
$$\bar \chi (G)$$
Theorem

