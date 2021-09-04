//
// Created by C. Zhang on 2021/9/4.
//

#ifndef QCQP_BG_DSDP_CUT_H
#define QCQP_BG_DSDP_CUT_H

#include "qp.h"
#include "utils.h"
#include "cut.h"

struct Bound_DSDP : Bound {

    double *xlb{};
    double *xub{};

    explicit Bound_DSDP(int n) {
        xlb = new double[n]{0.0};
        xub = new double[n]{0.0};
    }
};

struct RLT_DSDP : RLT {
    const int i{}, j{}, n{};
    const double li{}, ui{}, lj{}, uj{};
    RLT_DSDP(int n, int i, int j, double li, double ui, double lj, double uj);
};


#endif //QCQP_BG_DSDP_CUT_H
