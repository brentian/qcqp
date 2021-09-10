//
// Created by C. Zhang on 2021/9/5.
//

#ifndef QCQP_BRANCH_H
#define QCQP_BRANCH_H



#include "result.h"

class Bound {
public:
    std::vector<double> xlb;
    std::vector<double> xub;

    Bound();

    explicit Bound(int n);

    Bound(Bound &another);
};

class Branch {
public:

    int i, j, n;
    double xpivot_val, xminor_val;
    Bound left_b, right_b;

    Branch() = default;

    explicit Branch(Result &r);

    void create_from_result(const Result &r);

    void imply_bounds(Bound &b);
};


#endif //QCQP_BRANCH_H
