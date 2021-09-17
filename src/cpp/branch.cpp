//
// Created by C. Zhang on 2021/9/5.
//

#include "branch.h"
//
//Bound::Bound(Bound &another) {
//    xlb = std::vector<double>(another.xlb);
//    xub = std::vector<double>(another.xub);
//}

Bound::Bound(int n) {
    xlb = std::vector<double>(n, 0.0);
    xub = std::vector<double>(n, 1.0);
}

Bound::Bound() = default;


void Branch::create_from_result(const Result &r) {
    using namespace std;
    n = r.n;
    eigen_const_arraymap xm(r.x, r.n);


    eigen_matrix::Index maxRow, maxCol;
    auto row_sum = r.Res.rowwise().sum();
    row_sum.maxCoeff(&maxRow, &maxCol);
    i = (int) maxRow;
    r.Res.row(maxRow).maxCoeff(&maxRow, &maxCol);
    j = (int) maxCol;
    xpivot_val = r.x[i];
    xminor_val = r.x[j];
#if QCQP_BRANCH_DBG
    fprintf(stdout, "pivoting on i, j: (%d, %d)@ (%.2f, %.2f) ~ %.2f\n",
            i, j, xpivot_val, xminor_val, r.Res(i, j));
#endif
}

void Branch::imply_bounds(Bound &b) {
    // copy from parent bound
    left_b = Bound(b);
    right_b = Bound(b);
    // left
    left_b.xub[i] = xpivot_val;
    // right
    right_b.xlb[i] = xpivot_val;
}

Branch::Branch(Result &r) {
    create_from_result(r);
}



