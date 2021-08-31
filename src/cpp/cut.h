//
// Created by C. Zhang on 2021/8/2.
//

#ifndef QCQP_CUT_H
#define QCQP_CUT_H


#include "utils.h"

struct Cut {
    eigen_matrix B;
    double b;
};

struct RLT: Cut{

};

struct Bound{

};


#endif //QCQP_CUT_H
