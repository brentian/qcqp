"""
a script to get a low rank V
"""
import numpy as np

n, m, r = 8, 3, 2
V = np.random.uniform(-5, 5, (m, n, r))
A = - V @ V.transpose((0, 2, 1))

Vi = {}
Ai = {}
g = {}
v = {}
l = {}
s = {}
for i in range(m):
  Ai[i] = A[i]
  Vi[i] = V[i]
  g[i], _v = np.linalg.eigh(A[i])
  s[i] = (np.abs(g[i]) > 1e-4).sum()
  v[i] = _v[:, :s[i]]

for i in range(m):
  l[i] = - g[i].min()

#######
Ud = np.hstack(v.values()) # eigenvectors
Ad = np.hstack(Ai.values())
Vd = np.hstack(Vi.values())

# Q, R = np.linalg.qr(Ad) # [x]
Q, R = np.linalg.qr(Ud)
# Q, R = np.linalg.qr(Vd)

# rank of Q be p
p = (np.abs(R.diagonal()) > 1e-4).sum()
Qr = Q[:, :p]

for i in range(m):
  ei = np.linalg.eigvalsh(A[i] + l[i] * Q @ Q.T).min()
  print(i, f'min eig val {ei}')
  ei = np.linalg.eigvalsh(A[i] + l[i] * Qr @ Qr.T).min()
  print(i, f'min eig val {ei}')
