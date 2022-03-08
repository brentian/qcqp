//
// Created by C. Zhang on 2021/9/5.
//


#include "qcqp.h"


void check_solution(Result &r, QP &qp, const CutPool &cp) {
    using namespace std;
    if (cp.empty()) {
        check_solution(r, qp);
    } else {
#if DSDP_SDP_DBG
        check_solution(r, qp);
        cout << "check cuts..." << endl;
        int i = 0;
        for (const auto &c: cp) {
            fprintf(stdout, "check for cut: %d, %.3f, %.3f, %.3f\n",
                    i, (r.Xm * c.B).trace(), r.S[i + r.m], c.b);
            i++;
        }
#endif
    }

}

void check_solution(Result &r, QP &qp) {
    using namespace std;
    int i = 0;
    cout << "Comp: X∙Y:" << endl;
    cout << (r.Xm * r.Ym).format(EIGEN_IO_FORMAT) << endl;
    cout << "Res: X - xx.T:" << endl;
    eigen_const_arraymap xm(r.x, r.n);
    cout << (r.Xm.block(0, 0, r.n, r.n) - xm.matrix() * xm.matrix().adjoint()).format(EIGEN_IO_FORMAT) << endl;
    fprintf(stdout,
            "Obj: Q∙X = %.3f, alpha + b∙z = %.3f\n",
            (r.Xm * qp.Qh).trace(),
            qp.b.dot(eigen_const_arraymap(r.y + r.n + 1, r.m)) + r.y[0]);
    cout << "Quad constr..." << endl;
    for (const auto &Ah: qp.Ah) {
        fprintf(stdout, "Constr: %d, %.3f, %.3f, %.3f\n",
                i, (r.Xm * Ah).trace(), r.S[i], qp.b[i]);
        i++;
    }

    fprintf(stdout,
            "Primal: %.3f \n",
            qp.inhomogeneous_obj_val(r.x));

}
