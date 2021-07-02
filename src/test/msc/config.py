import pyqp.bg_msk_msc
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
  "msc": pyqp.bg_msk_msc.msc,
  "msc_eig": pyqp.bg_msk_msc.msc,
  "msc_diag": pyqp.bg_msk_msc.msc_diag,
  "socp": pyqp.bg_msk_msc.socp_relaxation
}


class Config(object):
  methods = methods
  kwargs = kwargs
  params = params
