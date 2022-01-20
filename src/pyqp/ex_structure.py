"""
Explore some structural properties for QP
"""
from .classes import CuttingPlane
from . import bg_msk
from .instances import QP
import numpy as np
import numpy.linalg as nl


class Structure(object):
  pass


class Disjunctions(Structure):
  """
  preconditioned disjunctions
  """
  
  def __init__(self, qp: QP):
    self.qp = qp
    self.A = None
    self.bool_cvx = False
    self.bool_created = False
    self.disjunctions = []
  
  def create_disjunctions(self):
    """
    based on rank/eigenvalues of the QP,
      create global disjunctions for the root node,
      ``offline``
    :return:
    """
    
    gamma, u = self.qp.gamma[0], self.qp.U[0]
    # it is the maximization problem thus split on positive eigenvalues
    #  into ax - b < 0 or ax -b > 0
    scope = gamma > 0
    #
    ncvx_size = scope.sum()
    self.bool_cvx = ncvx_size == 0
    if ncvx_size == 1:
      self._create_rank_one(scope)
    else:
      return False
  
  def _create_rank_one(self, scope):
    gamma, u = self.qp.gamma[0], self.qp.U[0]
    a = u.T[scope].flatten()
    R = u.T[~scope]
    self.bool_created = True
    for sign in [1, 0]:  # ax < 0, ax > 0
      self.disjunctions.append(DisjunctionCuttingPlane((a, R, sign)))


class DisjunctionCuttingPlane(CuttingPlane):
  def __init__(self, data):
    super().__init__(data)
  
  def serialize_to_msk(self, xvar, *args):
    expr = bg_msk.expr
    exprs = expr.sub
    exprm = expr.mul
    n = xvar.getShape()[0]
    a, R, sign = self.data
    
    ax = expr.dot(a if sign == 1 else -a, expr.flatten(xvar))
    conic_expr = expr.vstack(
      [ax, expr.flatten(expr.mul(R, xvar))]
    )
    
    # (xi - li)(xj - uj) <= 0
    dom1 = bg_msk.dom.inQCone()
    
    yield conic_expr, dom1


if __name__ == '__main__':
  pass
