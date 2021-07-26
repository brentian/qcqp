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

    qp_sdpa(QP &qp) : qp(qp) {};

    void create_sdpa_p();
};


#endif //QCQP_BG_SDPA_H
