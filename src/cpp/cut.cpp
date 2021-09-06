//
// Created by C. Zhang on 2021/8/2.
//

#include "cut.h"


RLT::RLT() {}

RLT::RLT(int n, int i, int j, double li, double ui, double lj, double uj) :
        n(n), i(i), j(j), li(li), ui(ui), lj(lj), uj(uj) {
    int ij = query_index_lt(i, j);
    int jn = query_index_lt(n, j);
    int in = query_index_lt(n, i);
    if (i != j) {
        size = 3;
        index = new int[3]{ij, jn, in};
        vals = new double[3]{1, -0.5 * ui, -0.5 * lj};
    } else {
        size = 2;
        index = new int[2]{ij, jn};
        vals = new double[2]{1, -0.5 * ui - 0.5 * lj};
    }
    b = -lj * ui;
#if QCQP_CUT_DBG
    // @note, only in dbg mode,
    // compute matrix B then use check_solution to see if
    // it is correct.
    using namespace std;
    double *xx = new double[(n + 1) * (n + 2) / 2]{0.0};
    for (int k = 0; k < size; ++k) {
        xx[index[k]] = vals[k];
    }
    double *xm = new double[(n + 1) * (n + 1)];
    input_lower_triangular(xx, xm, n + 1);
    eigen_matmap xem(xm, n + 1, n + 1);
    B = eigen_matrix::Zero(n + 1, n + 1);
    B += xem;
    delete[] xx;
    delete[] xm;
#endif

}

RLT RLT::create_from_branch(Branch &branch, int orient) {
    int i = branch.i;
    int j = branch.j;
    int n = branch.n;
    if (!orient) {
        //left child
        double li = branch.left_b.xlb[i];
        double lj = branch.left_b.xlb[j];
        double ui = branch.left_b.xub[i];
        double uj = branch.left_b.xub[j];
        return {n, i, j, li, ui, lj, uj};
    }
    double li = branch.right_b.xlb[i];
    double lj = branch.right_b.xlb[j];
    double ui = branch.right_b.xub[i];
    double uj = branch.right_b.xub[j];
    return {n, i, j, li, ui, lj, uj};
}
