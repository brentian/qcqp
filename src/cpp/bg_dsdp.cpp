//
// Created by chuwen on 2021/8/30.
//

#include "bg_dsdp.h"

void QP_DSDP::create_problem(bool solve, bool verbose) {
    DSDP dsdp;


//    DSDPCreateSDPCone(dsdp, 1, &sdpcone);
//    SDPConeSetBlockSize(sdpcone, 0, nnodes + 1);
//    SDPConeUsePackedFormat(sdpcone, 0);
//    SDPConeSetSparsity(sdpcone, 0, nedges + nnodes + 1);
//    DSDPSetOptions(dsdp,argv,argc)

    // number of variables
    int n = qp.n;
    m = qp.m;
    m_with_cuts = m + cp.size();
    r.m_with_cuts = m_with_cuts;
    int n_p_square = (n + 1) * (n + 1);
    int n_lower_tr = (n + 1) * (n + 2) / 2;
    //

    int nvar = 1 + n + m_with_cuts;
    // single sdp cone
    SDPCone sdpcone;
    DSDPCreate(nvar, &dsdp);
    DSDPCreateSDPCone(dsdp, 1, &sdpcone);
    // LP cones
    //    LPCone d;
    //    DSDPCreateLPCone(dsdp, &d);
    //    if (m > 0) {
    //        LPCone s;
    //        DSDPCreateLPCone(dsdp, &s);
    //    } else {};
    // todo
    //  SDPConeSetSparsity?
    //  SDPConeSetBlockSize(sdpcone, 0, n + 1);

    // Q: k = 0
    /**
     * SDPCone sdpcone,int64 blockj, int64 vari, int64 n, double alpha, double val[], int64 nnz);
     */
    SDPConeSetADenseVecMat(sdpcone, 0, 0, n + 1, -1, get_lower_triangular(qp.Qh), n_lower_tr);
    if (verbose) { SDPConeViewDataMatrix(sdpcone, 0, 0); }
    // alpha
    auto *val = new double[1]{1.0};
    auto idx = new int[1]{n_lower_tr - 1};
    SDPConeSetASparseVecMat(sdpcone, 0, 1, n + 1, -1, 0, idx, val, 1);
    if (verbose) SDPConeViewDataMatrix(sdpcone, 0, 1);
    DSDPSetDualObjective(dsdp, 1, 1);
    // zi, Ei: k = 2 + (0, ..., n - 1)
    for (int k = 0; k < n; ++k) {
        SDPConeSetADenseVecMat(
                sdpcone, 0, k + 2, n + 1, -1,
                get_lower_triangular(qp.Qd[k].block(0, 0, n + 1, n + 1)),
                n_lower_tr
        );
        if (verbose) {
            std::cout << qp.Qd[k].block(0, 0, n + 1, n + 1) << std::endl;
            SDPConeViewDataMatrix(sdpcone, 0, k + 2);
        }
    }
    // xi, Ai: k = 2 + n + (0, ..., m + #cuts - 1)
//    for (int k = 0; k < m_with_cuts; ++k) {
//        if (k < m) {
//            SDPConeSetADenseVecMat(
//                    sdpcone, 0, k + n + 2, n + 1, - 1.0,
//                    get_lower_triangular(qp.Ah[k]),
//                    n_lower_tr
//            );
//            DSDPSetDualObjective(dsdp, k + n + 2, - qp.b[k]);
//            if (verbose) {
//                std::cout << qp.Ah[k] << std::endl;
//                SDPConeViewDataMatrix(sdpcone, 0, k + n + 2);
//            }
//        } else {
//            // provided by Cutpool
//            Cut c = cp[k - m];
//        };
//    }

    // parameters
    int info;
//    info = DSDPSetGapTolerance(dsdp, 0.001);
//    info = DSDPSetPotentialParameter(dsdp, 5);
//    info = DSDPReuseMatrix(dsdp, 0);
//    info = DSDPSetPNormTolerance(dsdp, 1.0);
    info = DSDPSetup(dsdp);
    DSDPSetStandardMonitor(dsdp, 1);
    info = DSDPSolve(dsdp);
    DSDPSolutionType pdfeasible;
    DSDPGetSolutionType(dsdp, &pdfeasible);
    std::cout << pdfeasible;
//    double *xx, *y = diag;
//    info = DSDPGetY(dsdp, y, nnodes);
    info = DSDPComputeX(dsdp);
//    DSDPCHKERR(info);
    auto *xx = new double [n_p_square]{0.0};
    int xsize;
    info = SDPConeGetXArray(sdpcone, 0, &xx, &xsize);

    auto *x = input_lower_triangular(xx, n +1);
    eigen_matmap X(x, n + 1, n + 1);

    std::cout << X <<std::endl;
    std::cout << pdfeasible;


}
