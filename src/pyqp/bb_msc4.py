from .bb_msc3 import bg_msk, bb_box as bb_box3


def bb_box(qp, backend_name='msk', *args, **kwargs):
  if backend_name == 'msk':
    backend_func = bg_msk.msc_socp_relaxation
  else:
    raise ValueError("not implemented")
  print(kwargs)
  return bb_box3(qp, func=backend_func, *args, **kwargs)
