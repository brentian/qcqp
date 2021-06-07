import sys
import json
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
estimated_coord_style = dict(
  marker='o',
  facecolor='none',
  edgecolor='brown'
)


def read_snl_data():
  pass


def create_random_snl(n, d, k):
  # coordinates
  x = np.random.randint(-10, 10, (n, d))
  
  # compute distances
  d = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      d[i, j] = ((x[i] - x[j]) ** 2).sum()
  
  return d, x[:k], x[k:]


def plot_2d_snl(anchors, x, x_hat):
  fig, axs = plt.subplots()
  axs.scatter(anchors[:, 0], anchors[:, 1], marker="X", facecolor='red', label='anchors')
  axs.scatter(x[:, 0], x[:, 1], marker='x', facecolor='blue', label='ground truth')
  axs.scatter(x_hat[:, 0], x_hat[:, 1], **estimated_coord_style, label=r'$\hat x$')
  axs.legend()
  fig.savefig("/tmp/snl.png", dpi=500)
  return axs, fig


def snl_soc(n, d, k, D, anchors, points):
  try:
    import mosek.fusion as mf
    expr = mf.Expr
    dom = mf.Domain
    mat = mf.Matrix
  except Exception as e:
    import logging
    logging.exception(e)
    raise ValueError("no MOSEK found")
  
  de = D[:k]
  dx = D[k:, k:]
  model = mf.Model('snl_msc_msk')
  x = model.variable("x", [n - k, d])
  e = model.variable("e", [n - k, k], dom.greaterThan(0))
  v = model.variable("v", [n - k, n - k], dom.greaterThan(0))
  
  model.constraint(expr.sub(x, points), dom.equalsTo(0))
  for j in range(k):
    for i in range(n - k):
      eij = e.index([i, j])
      xi = x.pick([[i, idx] for idx in range(d)])
      aj = anchors[j].T.astype(float)
      
      # cross = |x_i|^2 \le eij - aj^Taj + 2aj^T x_i
      cross_expr = expr.add(
        expr.sub(
          eij,
          (aj ** 2).sum()
        ),
        expr.dot(xi, 2 * aj)
      )
      conic_expr = expr.vstack(0.5, cross_expr, xi)
      model.constraint(conic_expr, dom.inRotatedQCone())
      model.constraint(expr.sub(de[j, i + k], eij), dom.greaterThan(0))
  
  for i in range(n - k):
    for j in range(i):
      vij = v.index([i, j])
      cross_expr = expr.sub(
        x.pick([[i, idx] for idx in range(d)]),
        x.pick([[j, idx] for idx in range(d)]),
      )
      conic_expr = expr.vstack(0.5, vij, cross_expr)
      model.constraint(conic_expr, dom.inRotatedQCone())
      model.constraint(expr.sub(dx[i, j], vij), dom.greaterThan(0))
  
  model.objective(
    mf.ObjectiveSense.Minimize,
    expr.add(
      expr.sum(v),
      expr.sum(e))
  )
  
  model.setLogHandler(sys.stdout)
  model.solve()
  model.getProblemStatus()
  return x.level().reshape(x.getShape()), \
         e.level().reshape(e.getShape()), \
         v.level().reshape(v.getShape())


def snl_msc(n, d, k, D, anchors, points):
  try:
    import mosek.fusion as mf
    expr = mf.Expr
    dom = mf.Domain
    mat = mf.Matrix
  except Exception as e:
    import logging
    logging.exception(e)
    raise ValueError("no MOSEK found")
  
  de = D[:k]
  dx = D[k:, k:]
  model = mf.Model('snl_msc_msk')
  x = model.variable("x", [n - k, d])
  z = model.variable("z", [n - k, n - k, d])
  y = model.variable("y", [n - k, n - k, d], dom.greaterThan(0))
  zb = model.variable("zb", [n - k, k, d])
  yb = model.variable("yb", [n - k, k, d], dom.greaterThan(0))
  
  #########
  # vanilla check
  # model.constraint(expr.sub(x, points), dom.equalsTo(0))
  for i in range(n - k):
    for j in range(k):
      aj = anchors[j].T.astype(float)
      zij = zb.pick([[i, j, idx] for idx in range(d)])
      yij = yb.pick([[i, j, idx] for idx in range(d)])
      for idx in range(d):
        conic_expr = expr.vstack(0.5, yb.index([i, j, idx]), zb.index([i, j, idx]))
        model.constraint(conic_expr, dom.inRotatedQCone())
      # affine
      cross_expr = expr.sub(
        x.pick([[i, idx] for idx in range(d)]),
        aj
      )
      model.constraint(expr.sub(cross_expr, zij), dom.equalsTo(0))
      model.constraint(expr.sub(de[j, i + k], expr.sum(yij)), dom.greaterThan(0))
  # for j in range(k):
  #   for i in range(n - k):
  #     eij = e.index([i, j])
  #     xi = x.pick([[i, idx] for idx in range(d)])
  #     aj = anchors[j].T.astype(float)
  #
  #     # cross = |x_i|^2 \le eij - aj^Taj + 2aj^T x_i
  #     cross_expr = expr.add(
  #       expr.sub(
  #         eij,
  #         (aj ** 2).sum()
  #       ),
  #       expr.dot(xi, 2 * aj)
  #     )
  #     conic_expr = expr.vstack(0.5, cross_expr, xi)
  #     model.constraint(conic_expr, dom.inRotatedQCone())
  #     model.constraint(expr.sub(de[j, i + k], eij), dom.greaterThan(0))
  
  for i in range(n - k):
    for j in range(i):
      zij = z.pick([[i, j, idx] for idx in range(d)])
      yij = y.pick([[i, j, idx] for idx in range(d)])
      for idx in range(d):
        conic_expr = expr.vstack(0.5, y.index([i, j, idx]), z.index([i, j, idx]))
        model.constraint(conic_expr, dom.inRotatedQCone())
      # affine
      cross_expr = expr.sub(
        x.pick([[i, idx] for idx in range(d)]),
        x.pick([[j, idx] for idx in range(d)]),
      )
      model.constraint(expr.sub(cross_expr, zij), dom.equalsTo(0))
      model.constraint(expr.sub(dx[i, j], expr.sum(yij)), dom.greaterThan(0))
  
  model.objective(
    mf.ObjectiveSense.Minimize,
    0
  )
  
  model.setLogHandler(sys.stdout)
  model.solve()
  model.getProblemStatus()
  return x.level().reshape(x.getShape()), \
         z.level().reshape(z.getShape()), \
         y.level().reshape(y.getShape())


if __name__ == '__main__':
  n, d, k = 5, 2, 2
  
  D, ac, pts = create_random_snl(n, d, k)
  xx, *_ = snl_msc(n, d, k, D, ac, pts)
  ax, figure = plot_2d_snl(ac, pts, xx)
