//
// Created by C. Zhang on 2021/9/4.
//

#include "bg_dsdp_cut.h"

RLT_DSDP::RLT_DSDP(int n, int i, int j, double li, double ui, double lj, double uj) :
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
//#if dbg
    using namespace std;
    double *xx = new double[(n + 1) * (n + 2) / 2]{0.0};
    for (int k = 0; k < size; ++k) {
        cout << index[k] << endl;
        xx[index[k]] = vals[k];
    }
    double *xm = new double[(n + 1) * (n + 1)];
    input_lower_triangular(xx, xm, n + 1);
    eigen_matmap xem(xm, n + 1, n + 1);
    B = eigen_matrix::Zero(n + 1, n + 1);
    B += xem;
    cout << B << endl;
    delete[] xx;
    delete[] xm;
//#endif

}
