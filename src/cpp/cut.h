//
// Created by C. Zhang on 2021/8/2.
//

#ifndef QCQP_CUT_H
#define QCQP_CUT_H


#include "utils.h"

struct Cut {
    eigen_matrix B;
    double b;
    int* index;
    double* vals;
    int size;
};

struct RLT : Cut {


};

struct Bound {

};

typedef std::vector<Cut> CutPool;
#endif //QCQP_CUT_H
