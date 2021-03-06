import itertools
import numpy as np
import time

from .classes import Result, Params, QP, Bounds


def qp_gurobi(
    qp: QP,
    bounds: Bounds,
    sense="max",
    relax=True,
    verbose=True,
    params: Params = Params(),
    **kwargs
):
  """
  QCQP using Gurobi 9.1 as benchmark
  todo: works only for 1-d case now
      can we use Gurobi Matrix API?
  Parameters
  ----------
  Q
  q
  A
  ax
  b
  sign
  lb
  ub
  sense
  relax
  kwargs

  Returns
  -------

  """
  ######################
  # === extra params
  ######################
  import gurobipy as grb
  st_time = time.time()
  Q, q, A, a, b, sign, al, au = qp.unpack()
  lb, ub, ylb, yub, s = bounds.unpack()
  m, n, d = a.shape
  model = grb.Model()
  model.setParam(grb.GRB.Param.OutputFlag, verbose)
  
  indices = range(q.shape[0])
  vtype = 'C' if relax else 'I'
  x = model.addVars(indices, lb=-grb.GRB.INFINITY, vtype=vtype)
  if lb is not None:
    model.setAttr('lb', x, lb.flatten())
    model.setAttr('ub', x, ub.flatten())
  if s:
    model.addConstr(grb.quicksum(xx * xx for xx in x.values()) <= bounds.sphere ** 2)
  
  obj_expr = grb.quicksum(Q[i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices)) \
             + grb.quicksum(q[i][0] * x[i] for i in indices)
  if sign is not None:
    # unilateral constraints
    for constr_num in range(m):
      if sign[constr_num] == 0:
        model.addConstr(
          grb.quicksum(x[j] * a[constr_num][j][0] for j in indices)
          + grb.quicksum(A[constr_num][i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices))
          == b[constr_num])
      
      elif sign[constr_num] == -1:
        model.addConstr(
          grb.quicksum(x[j] * a[constr_num][j][0] for j in indices)
          + grb.quicksum(A[constr_num][i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices))
          >= b[constr_num])
      
      else:
        model.addConstr(
          grb.quicksum(x[j] * a[constr_num][j][0] for j in indices)
          + grb.quicksum(A[constr_num][i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices))
          <= b[constr_num])
  else:
    for constr_num in range(m):
      model.addConstr(
        grb.quicksum(x[j] * a[constr_num][j][0] for j in indices)
        + grb.quicksum(A[constr_num][i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices))
        <= au[constr_num])
      model.addConstr(
        grb.quicksum(x[j] * a[constr_num][j][0] for j in indices)
        + grb.quicksum(A[constr_num][i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices))
        >= al[constr_num])
  
  model.setParam(grb.GRB.Param.NonConvex, 2)
  model.setParam(grb.GRB.Param.TimeLimit,  params.time_limit)
  model.setParam(grb.GRB.Param.MIPGap, params.opt_eps)
  model.setParam(grb.GRB.Param.Threads, params.threads)
  model.setObjective(obj_expr, sense=(grb.GRB.MAXIMIZE if sense == 'max' else grb.GRB.MINIMIZE))
  model.optimize()
  r = Result()
  r.solve_time = model.Runtime
  if model.status == grb.GRB.OPTIMAL:
    r.relax_obj = r.true_obj = r.bound = model.ObjVal
    r.nodes = model.NodeCount
  else:
    r.bound = model.ObjBoundC
    r.relax_obj = model.ObjBoundC
    r.true_obj = model.ObjVal
    r.nodes = model.NodeCount
  r.xval = np.array([i.x for i in x.values()]).reshape(q.shape).round(6)
  
  return r
