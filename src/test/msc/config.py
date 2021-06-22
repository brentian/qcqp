from pyqp import grb, bb, bg_msk
from pyqp import bb_msc, bb_msc2

params = bb_msc.BCParams()
kwargs = dict(
  relax=True,
  sense="max",
  verbose=False,
  params=params,
  bool_use_shor=False,
  rlt=True
)
methods = {
  "grb": grb.qp_gurobi,
  "shor": bg_msk.shor,
  "msc": bg_msk.msc,
  "msc_eig": bg_msk.msc,
  "msc_diag": bg_msk.msc_diag,
  "socp": bg_msk.socp_relaxation
}


class Config(object):
  methods = methods
  kwargs = kwargs
  params = params
