//
// Created by C. Zhang on 2021/8/2.
//

#ifndef QCQP_CUT_H
#define QCQP_CUT_H


#include "utils.h"
#include "branch.h"

struct Cut {
    eigen_matrix B;
    double b;
    int *index;
    double *vals;
    int size;
};

template<typename BranchType>
struct RLT : Cut {
    int i{}, j{}, n{};
    double li{}, ui{}, lj{}, uj{};

    static std::vector<RLT> create_from_branch(BranchType &branch, int orient);
    RLT(int n, int i, int j, double li, double ui, double lj, double uj);
    RLT() = default;
};


typedef std::vector<Cut> CutPool;
#endif //QCQP_CUT_H
