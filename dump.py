import mosek.fusion as mf
import numpy as np
import sys
expr = mf.Expr
dom = mf.Domain
mat = mf.Matrix
model = mf.Model('shor_msk')
model.setLogHandler(sys.stdout)

n = 10
Q = np.ones((n, n))
Z = model.variable("Z", dom.inPSDCone(n + 1))
Y = Z.slice([0, 0], [n, n])
x = Z.slice([0, n], [n, n + 1])

# bounds
model.constraint(expr.sub(x, 1), dom.lessThan(0))
model.constraint(expr.sub(x, 0), dom.greaterThan(0))
model.constraint(expr.sub(Y.diag(), x), dom.lessThan(0))
model.constraint(Z.index(n, n), dom.equalsTo(1.))


x.setLevel(np.zeros(n).tolist())
model.objective(mf.ObjectiveSense.Minimize,
                expr.dot(Q, Y))

model.writeTask("./dump.task.gz")
model.solve()