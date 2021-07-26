//
// Created by C. Zhang on 2021/7/25.
//

#ifndef QCQP_BG_SDPA_H
#define QCQP_BG_SDPA_H

#include "qp.h"
#include "utils.h"
#include "sdpa_call.h"

class qp_sdpa {

public:
    QP qp;
    SDPA p;
    bool solved = false;


    explicit qp_sdpa(QP &qp) : qp(qp) {};

    ~qp_sdpa() { p.terminate(); }

    void create_sdpa_p(bool solve = false, bool verbose = true);
    void solve_sdpa_p(bool verbose=false);
};


#endif //QCQP_BG_SDPA_H
