//
// Created by C. Zhang on 2021/9/5.
//

#ifndef QCQP_QCQP_H
#define QCQP_QCQP_H

#include "utils.h"
#include "cut.h"
#include "branch.h"
#include "qp.h"
// backends
#include "bg_dsdp.h"
#include "bg_dsdp_cut.h"
#include "bg_sdpa.h"
// bb
#include "bb_dsdp.h"


void check_solution(Result &r, QP &qp);

void check_solution(Result &r, QP &qp, const CutPool &cp);

#endif //QCQP_QCQP_H
