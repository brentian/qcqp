
# Chordal graph

- An undirected graph $G$ is called chordal (or **triangulated**, or a rigid circuit) if every cycle of length greater than or equal to 4 has at least one chord.
  - Chord: 
  - Non-chordal graphs can always be chordal extended, i.e., extended to a chordal graph, by adding additional edges to the original graph

- Ordering: An ordering $\sigma \{1, 2, ... , n\} \Rightarrow V$ can also be interpreted as a sequence of vertices $\sigma = (\sigma(1), ... , \sigma(n))$. We refer to $\sigma^{−1} (v)$ as the index of vertex $v$ of such ordering.
  - denote a ordered graph as $G_\sigma$


- A clique is a set of vertices $W \subseteq V$ that induces a maximal complete subgraph
  - A vertex $v$ of an undirected graph is simplicial if its neighborhood $adj(v)$ is complete.