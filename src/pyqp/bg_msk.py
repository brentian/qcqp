import sys

try:
  import mosek.fusion as mf
  
  expr = mf.Expr
  dom = mf.Domain
  mat = mf.Matrix
except Exception as e:
  import logging
  
  logging.exception(e)

from .classes import Result, qp_obj_func, QP


class MSKResult(Result):
  def __init__(self, problem: mf.Model = None, yval=0, xval=0, tval=0, relax_obj=0, true_obj=0, bound=0, solve_time=0,
               xvar: mf.Variable = None,
               yvar: mf.Variable = None,
               zvar: mf.Variable = None):
    super().__init__(problem, yval, xval, tval, relax_obj, true_obj, bound, solve_time)
    self.xvar = xvar
    self.yvar = yvar
    self.zvar = zvar
    self.solved = False
  
  def solve(self):
    self.problem.solve()
    self.yval = self.yvar.level().reshape(self.yvar.getShape())
    self.xval = self.xvar.level().reshape(self.xvar.getShape())
    self.relax_obj = self.problem.primalObjValue()
    self.solved = True


def shor(
    qp: QP,
    sense="max", verbose=True, solve=True, **kwargs
):
  """
  use a Y along with x in the SDP
      for basic 0 <= x <= e, diag(Y) <= x
  Parameters
  ----------
  Q
  q
  A
  a
  b
  sign
  lb
  ub
  solver
  sense
  kwargs

  Returns
  -------

  """
  _unused = kwargs
  Q, q, A, a, b, sign, lb, ub, ylb, yub, diagx = qp.unpack()
  m, n, d = a.shape
  xshape = (n, d)
  model = mf.Model('shor_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  Z = model.variable("Z", dom.inPSDCone(n + 1))
  Y = Z.slice([0, 0], [n, n])
  x = Z.slice([0, n], [n, n + 1])
  
  # bounds
  model.constraint(expr.sub(x, ub), dom.lessThan(0))
  model.constraint(expr.sub(x, lb), dom.greaterThan(0))
  model.constraint(expr.sub(Y.diag(), x), dom.lessThan(0))
  model.constraint(Z.index(n, n), dom.equalsTo(1.))
  for i in range(m):
    if sign[i] == 0:
      model.constraint(
        expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
        dom.equalsTo(b[i]))
    elif sign[i] == -1:
      model.constraint(
        expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
        dom.greaterThan(b[i]))
    else:
      model.constraint(
        expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
        dom.lessThan(b[i]))
  
  # objectives
  obj_expr = expr.add(expr.sum(expr.mulElm(Q, Y)), expr.dot(x, q))
  model.objective(mf.ObjectiveSense.Minimize
                  if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr)
  
  r = MSKResult()
  r.xvar = x
  r.yvar = Y
  r.zvar = Z
  r.problem = model
  if not solve:
    return r
  
  model.solve()
  xval = x.level().reshape(xshape)
  r.yval = Y.level().reshape((n, n))
  r.xval = xval
  r.relax_obj = model.primalObjValue()
  r.true_obj = qp_obj_func(Q, q, xval)
  r.solved = True
  r.solve_time = model.getSolverDoubleInfo("optimizerTime")
  
  return r



def dshor(
    qp: QP,
    sense="max", verbose=True, solve=True, **kwargs
):
  """
  dual of SDP relaxation
  Parameters
  -------

  """
  _unused = kwargs
  Q, q, A, a, b, sign, lb, ub, ylb, yub, diagx = qp.unpack()
  m, n, d = a.shape
  xshape = (n, d)
  model = mf.Model('shor_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  Z = model.variable("Z", dom.inPSDCone(n + 1))
  Y = Z.slice([0, 0], [n, n])
  x = Z.slice([0, n], [n, n + 1])
  
  # bounds
  model.constraint(expr.sub(x, ub), dom.lessThan(0))
  model.constraint(expr.sub(x, lb), dom.greaterThan(0))
  model.constraint(expr.sub(Y.diag(), x), dom.lessThan(0))
  model.constraint(Z.index(n, n), dom.equalsTo(1.))
  for i in range(m):
    if sign[i] == 0:
      model.constraint(
        expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
        dom.equalsTo(b[i]))
    elif sign[i] == -1:
      model.constraint(
        expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
        dom.greaterThan(b[i]))
    else:
      model.constraint(
        expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
        dom.lessThan(b[i]))
  
  # objectives
  obj_expr = expr.add(expr.sum(expr.mulElm(Q, Y)), expr.dot(x, q))
  model.objective(mf.ObjectiveSense.Minimize
                  if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr)
  
  r = MSKResult()
  r.xvar = x
  r.yvar = Y
  r.zvar = Z
  r.problem = model
  if not solve:
    return r
  
  model.solve()
  xval = x.level().reshape(xshape)
  r.yval = Y.level().reshape((n, n))
  r.xval = xval
  r.relax_obj = model.primalObjValue()
  r.true_obj = qp_obj_func(Q, q, xval)
  r.solved = True
  r.solve_time = model.getSolverDoubleInfo("optimizerTime")
  
  return r


