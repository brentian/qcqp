import numpy as np
import sys

try:
  import mosek.fusion as mf
  
  expr = mf.Expr
  dom = mf.Domain
  mat = mf.Matrix
except Exception as e:
  import logging
  
  logging.exception(e)


def subp_x(
    xi,
    kappa,
    mu,
    rho=1,
    verbose=True
):
  n, d = xi.shape
  model = mf.Model("kissing-num")
  S = model.variable('s', dom.greaterThan(0))
  X = model.variable('x', [n, d])
  
  # |xi + xj | <= 12
  for i in range(n):
    for j in range(i, n):
      model.constraint(
        expr.vstack(
          0.5,
          S,
          expr.flatten(expr.add(X.slice([i, 0], [i + 1, d]), X.slice([j, 0], [j + 1, d])))
        ),
        dom.inRotatedQCone()
      )
  model.constraint(S, dom.lessThan(12))
  
  obj_expr = 0
  
  # ALM terms
  # the t - s gap
  t = model.variable("t", dom.inRange(0, 4))
  expr_norm_gap = expr.sub(t, 4)
  expr_norm_gap_sqr = model.variable("t_s", dom.greaterThan(0))
  model.constraint(
    expr.vstack(1 / 2, expr_norm_gap_sqr, expr_norm_gap), dom.inRotatedQCone()
  )
  for i in range(n):
    model.constraint(
      expr.vstack(1 / 2, t, expr.flatten(X.slice([i, 0], [i + 1, d]))),
      dom.inRotatedQCone()
    )
  obj_expr = expr.add(obj_expr, expr.mul(kappa, expr_norm_gap))
  obj_expr = expr.add(obj_expr, expr.mul(rho / 2, expr_norm_gap_sqr))
  
  # the <𝜉, x> - gap
  expr_norm_x_gap = model.variable("xi_x", [n])
  expr_norm_x_gap_sqr = model.variable("xi_x_sqr", [n], dom.greaterThan(0))
  for i in range(n):
    model.constraint(
      expr.sub(
        expr.dot(
          expr.flatten(X.slice([i, 0], [i + 1, d])),
          xi[i]
        ),
        expr_norm_x_gap.index(i)),
      dom.equalsTo(4)
    )
    model.constraint(
      expr.vstack(1 / 2, expr_norm_x_gap_sqr.index(i), expr_norm_x_gap),
      dom.inRotatedQCone()
    )
  obj_expr = expr.add(obj_expr, expr.dot(mu, expr_norm_x_gap))
  obj_expr = expr.add(obj_expr, expr.dot(rho / 2 * np.ones(n), expr_norm_x_gap_sqr))
  
  # ALM objective
  model.objective(mf.ObjectiveSense.Minimize, obj_expr)
  if verbose:
    model.setLogHandler(sys.stdout)
  model.solve()
  x = X.level().reshape((n, d))
  xs = expr_norm_x_gap.level()
  xs_sqr = expr_norm_x_gap_sqr.level()
  s = S.level()
  t = t.level()
  return x, xs, xs_sqr, t, s


# %%
def subp_xi(x, mu, rho=1, verbose=True):
  n, d = x.shape
  ###################
  # The above part is unchanged
  ###################
  # norm bounds on y^Te
  model = mf.Model("kissing-num-aux")
  Xi = model.variable("xi", [n, d])
  
  for i in range(n):
    model.constraint(
      expr.vstack(2, expr.flatten(Xi.slice([i, 0], [i + 1, d]))),
      dom.inQCone()
    )
  obj_expr = 0
  
  # ALM terms
  # the <𝜉, x> - gap
  # the <𝜉, x> - gap
  expr_norm_x_gap = model.variable("xi_x", [n])
  expr_norm_x_gap_sqr = model.variable("xi_x_sqr", [n], dom.greaterThan(0))
  
  for i in range(n):
    model.constraint(
      expr.sub(
        expr.dot(
          x[i],
          expr.flatten(Xi.slice([i, 0], [i + 1, d]))
        ),
        expr_norm_x_gap.index(i)),
      dom.equalsTo(4)
    )
    model.constraint(
      expr.vstack(1 / 2, expr_norm_x_gap_sqr.index(i), expr_norm_x_gap),
      dom.inRotatedQCone()
    )
  obj_expr = expr.add(obj_expr, expr.dot(mu, expr_norm_x_gap))
  obj_expr = expr.add(obj_expr, expr.dot(rho / 2 * np.ones(n), expr_norm_x_gap_sqr))
  
  # ALM objective
  model.objective(mf.ObjectiveSense.Minimize, obj_expr)
  if verbose:
    model.setLogHandler(sys.stdout)
  model.solve()
  xi = Xi.level().reshape((n, d))
  xs = expr_norm_x_gap.level()
  xs_sqr = expr_norm_x_gap_sqr.level()
  return xi, xs, xs_sqr


if __name__ == '__main__':
  n, d = 2, 2
  # %%
  xi = x = np.ones((n, d))
  rho = 100
  mu = np.ones(n)
  kappa = 0
  for i in range(10):
    xi, xs_xi, xs_sqr_xi = subp_xi(x, mu, rho, False)
    x, xs_x, xs_sqr_x, t, s = subp_x(xi, kappa, mu, rho, False)
    # t_s  = t - 4
    alm = xs_xi.T @ mu + rho / 2 * xs_sqr_xi.sum() + (t - 4)[0] * kappa + rho / 2 * (t - 4)[0]**2
    mu += xs_xi * rho
    kappa += (t - 4)[0] * rho
    
    if abs(alm) < 1e-2:
      break
    print(f"@{i}, alm: {alm}, bd: {0}, 𝜉x - 4: {np.abs(xs_xi).max()}")
