//
// Created by chuwen on 2021/8/30.
//

#include "bg_dsdp.h"

std::string dsdp_status(DSDPSolutionType st) {
    if (st == 0)return "!<  Not sure whether (D) or (P) is feasible, check y bounds ";
    else if (st == 1) return "!<  Both (D) and (P) are feasible and bounded ";
    else if (st == 3) return "!<  (D) is unbounded and (P) is infeasible  ";
    else return "!<  (D) in infeasible and (P) is unbounded  ";
}

std::string dsdp_stopreason(DSDPTerminationReason st) {
    if (st == 0)return "!< Don't Stop  ";
    else if (st == 1) return "!< Good news: Solution found.";
    else if (st == -6) return "!< The initial points y and r imply that S is not positive  ";
    else if (st == -2) return "!< Short step lengths created by numerical difficulties prevent progress  ";
    else if (st == -8) return "!< Theoretically this matrix is positive definite  ";
    else if (st == -3) return "!< Reached maximum number of iterations  ";
    else if (st == -9) return "!< Another numerical error occurred. Check solution ";
    else if (st == 5) return "!< Objective (DD) big enough to stop  ";
    else if (st == 7) return "!< DSDP didn't stop it, did you?  ";
    else return "!<  Unknown Error  ";
}


void QP_DSDP::create_problem(bool solve, bool verbose, bool use_lp_cone) {
    DSDP dsdp;


    // number of variables (for original)
    int n = qp.n;
    // problem size
    int ndim = n + 1; // homogeneous
    int n_p_square = (n + 1) * (n + 1); // # nnz in the full matrix
    int n_lower_tr = (n + 1) * (n + 2) / 2; // # nnz in the lower triangular block
    // total number of constrs;
    // - 1 for ynn = 1
    // - n diagonal constr: y[i, i] <= x[i]
    // - m_with_cuts
    //      - m quadratic constr
    //      - cp.size() cuts
    int nvar = n + m_with_cuts + 1;

    // dynamic arrays
    double *_tilde_q_data = new double[n_lower_tr]{0.0};
    double val[] = {1.0};
    int idx[] = {n_lower_tr - 1};
    double _ei_val[] = {1.0, -0.5};
    int *_ei_idx = new int[2 * n]{0};
    double *_ah_data = new double[n_lower_tr * m_with_cuts]{0.0};
    // dynamic arrays for solution
    double *slack = new double[nvar]{0.0};
    double *surplus = new double[nvar]{0.0};
    double *x = new double[ndim * ndim]{0.0};
    double *xx = new double[nvar]{0.0};
    double *y = new double[nvar]{0.0};
    int xsize;

    // declare the problem
    SDPCone sdpcone;
    BCone bcone;
    DSDPCreate(nvar, &dsdp);
    DSDPCreateSDPCone(dsdp, 1, &sdpcone);
    DSDPCreateBCone(dsdp, &bcone);
    // todo ? set block size and sparsity
    //    SDPConeSetBlockSize(sdpcone, 0, nvar);
    //    SDPConeSetBlockSize(sdpcone, 1, n);
    //  SDPConeSetSparsity?

    // \tilde Q: homogenized Q
    get_lower_triangular(qp.Qh, _tilde_q_data);
    SDPConeSetADenseVecMat(
            sdpcone, // SDPCone sdpcone,
            0, // int64 blockj,
            0, // int64 vari,
            ndim, // int64 n,
            -1.0, // double alpha
            _tilde_q_data, // double val[],
            n_lower_tr // int64 nnz
    );
    // alpha, Y[n, n] = 1
    SDPConeSetASparseVecMat(
            sdpcone, //SDPCone sdpcone
            0, //int64 blockj,
            1, //int64 vari,
            ndim, //int64 n,
            1.0, //double alpha,
            0, //int64 ishift
            idx, //const int64 ind[],
            val, //const double val[],
            1 // int64 nnz
    );
    if (verbose) SDPConeViewDataMatrix(sdpcone, 0, 1);
    DSDPSetDualObjective(dsdp, 1, 1.0);
    // Y <= xx^T,
    // \tilde Y ∙ E_i + \diag(d) ∙ \diag(e_i)
    // zi, Ei: k = 2 + (0, ..., n - 1)
    for (int k = 0; k < n; ++k) {
        int k_th_diag_idx = (k + 1) * (k + 2) / 2 - 1;
        _ei_idx[k * 2] = k_th_diag_idx;
        _ei_idx[k * 2 + 1] = n_lower_tr - 1 - n + k;
    }
    for (int k = 0; k < n; ++k) {
        int vari = k + 2;
        SDPConeSetSparseVecMat(
                sdpcone, //SDPCone sdpcone
                0, //int64 blockj,
                vari, //int64 vari,
                ndim, //int64 n,
                0, //int64 ishift
                _ei_idx + k * 2, //const int64 ind[],
                _ei_val, //const double val[],
                2 // int64 nnz
        );
        BConeSetUpperBound(bcone, k + 2, 0);
        //  @note: equivalent
        //      BConeSetPSlackVariable(bcone, k + 2);
        DSDPSetDualObjective(dsdp, vari, 0.0);
    }

    // constraints
    // xi, Ai: k = 2 + n + (0, ..., m + #cuts - 1)
    for (int k = 0; k < m_with_cuts; ++k) {
        int vari = k + n + 2;
        int slice = k * n_lower_tr;
        if (k < m) {
            std::cout << qp.Ah[k] << std::endl;
            get_lower_triangular(qp.Ah[k], _ah_data + slice);
            SDPConeSetADenseVecMat(
                    sdpcone,
                    0,
                    vari,
                    ndim,
                    1.0,
                    _ah_data + slice,
                    n_lower_tr
            );
            BConeSetPSlackVariable(bcone, vari);
            DSDPSetDualObjective(dsdp, vari, qp.b[k]);
        } else {
            // provided by Cutpool
            Cut c = cp[k - m];
        };
    }
    if (verbose) {
        for (int k = 0; k < nvar + 1; ++k) {
            fprintf(stdout,
                    "@showing constraint %d, dual-obj: %.3f \n",
                    k,
                    1.0
            );
            SDPConeViewDataMatrix(sdpcone, 0, k);
        }
        BConeView(bcone);
    }

    // set parameters
    int info;
    //    info = DSDPSetGapTolerance(dsdp, 0.001);
    //    info = DSDPSetPotentialParameter(dsdp, 5);
    //    info = DSDPReuseMatrix(dsdp, 0);
    //    info = DSDPSetPNormTolerance(dsdp, 1.0);
    DSDPSetStandardMonitor(dsdp, 1);
    info = DSDPSetPenaltyParameter(dsdp, nvar);
    info = DSDPSetup(dsdp);
    info = DSDPSolve(dsdp);

    // solutions
    DSDPSolutionType pdfeasible;
    DSDPTerminationReason reason;
    DSDPGetSolutionType(dsdp, &pdfeasible);
    DSDPStopReason(dsdp, &reason);
    info = DSDPComputeX(dsdp);

    SDPConeGetXArray(sdpcone, 0, &xx, &xsize);
    DSDPGetY(dsdp, y, nvar);
    BConeCopyX(bcone, surplus, slack, nvar);
    input_lower_triangular(xx, x, ndim);
    eigen_matmap X(x, ndim, ndim);
    eigen_arraymap ye(y, nvar);
    eigen_arraymap sl(slack, nvar);
    eigen_arraymap su(surplus, nvar);
    eigen_matrix S = eigen_matrix::Zero(ndim, ndim);
    // compute dual variable S
    for (int i = 0; i < n; ++i) {
        int vari = i + 2;
        S -= qp.Qd[i].block(0, 0, ndim, ndim) * y[vari];
    }
    for (int i = 0; i < m_with_cuts; ++i) {
        int vari = i + n + 2;
        S -= qp.Ah[i] * y[vari];
    }

    S(ndim - 1, ndim - 1) -= y[0];
    S -= qp.Qh;
    // views.
    SDPConeViewX(sdpcone, 0, ndim, xx, xsize);
    std::cout << X << std::endl;
    std::cout << S << std::endl;
    std::cout << ye.transpose() << std::endl;
    std::cout << "slacks" << std::endl;
    std::cout << sl.transpose() << std::endl;
    std::cout << su.transpose() << std::endl;
    std::cout << "checks" << std::endl;
    fprintf(stdout,
            "check for objective, primal %.3f, dual %.3f\n",
            -(X * qp.Qh).trace(),
            ye(0)
    );
    for (int k = 0; k < n; ++k) {
        fprintf(stdout,
                "check for diag contraint: %d, %.3f, %.3f, %.3f\n",
                k,
                (qp.Qd[k].block(0, 0, ndim, ndim) * X).trace(),
                sl(k + 1),
                su(k + 1)
        );
    }
    for (int k = 0; k < m_with_cuts; ++k) {
        fprintf(stdout,
                "check for quadratic contraint: %d, %.3f, %.3f, %.3f\n",
                k,
                (qp.Ah[k].block(0, 0, ndim, ndim) * X).trace(),
                sl(k + 1 + n),
                su(k + 1 + n)
        );
    }

    DSDPDestroy(dsdp);
    // frees
    delete[] _tilde_q_data;
    delete[] _ei_idx;
    delete[] _ah_data;

}
