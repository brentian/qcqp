//
// Created by C. Zhang on 2021/8/2.
//

#ifndef QCQP_CUT_H
#define QCQP_CUT_H


#include "utils.h"
#include "branch.h"
#define QCQP_CUT_DBG 1
struct Cut {
    eigen_matrix B;
    double b;
    int *index;
    double *vals;
    int size;
};

struct RLT : Cut {
    int i{}, j{}, n{};
    double li{}, ui{}, lj{}, uj{};

    static RLT create_from_branch(Branch &branch, int orient);
    RLT(int n, int i, int j, double li, double ui, double lj, double uj);
    RLT();
};


typedef std::vector<Cut> CutPool;
#endif //QCQP_CUT_H
