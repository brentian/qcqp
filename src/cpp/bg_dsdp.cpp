//
// Created by chuwen on 2021/8/30.
//

#include "bg_dsdp.h"

std::string dsdp_status(DSDPSolutionType st) {
    if (st == 0)return "!< Not sure whether (D) or (P) is feasible, check y bounds ";
    else if (st == 1) return "!< Both (D) and (P) are feasible and bounded ";
    else if (st == 3) return "!< (D) is unbounded and (P) is infeasible  ";
    else return "!< (D) in infeasible and (P) is unbounded  ";
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
    else return "!< Unknown Error  ";
}


void QP_DSDP::create_problem(bool solve, bool verbose, bool use_lp_cone) {

    setup();
    // declare the problem
    DSDPCreate(nvar, &p);
    DSDPCreateSDPCone(p, 1, &sdpcone);
    DSDPCreateBCone(p, &bcone);
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
            _one_idx, //const int64 ind[],
            _one_val, //const double val[],
            1 // int64 nnz
    );
#if DSDP_SDP_DBG
    SDPConeViewDataMatrix(sdpcone, 0, 1);
#endif
    DSDPSetDualObjective(p, 1, 1.0);
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
        DSDPSetDualObjective(p, vari, 0.0);
    }


    // constraints
    // xi, Ai: k = 2 + n + (0, ..., m + #cuts - 1)
    for (int k = 0; k < m_with_cuts; ++k) {
        int vari = k + n + 2;
        int slice = k * n_lower_tr;
        if (k < m) {
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
            DSDPSetDualObjective(p, vari, qp.b[k]);
        } else {
            // provided by Cutpool
            Cut c = cp[k - m];
            SDPConeSetSparseVecMat(
                    sdpcone,
                    0,
                    vari,
                    ndim,
                    0,
                    c.index,
                    c.vals,
                    c.size
            );
            BConeSetPSlackVariable(bcone, vari);
            DSDPSetDualObjective(p, vari, c.b);
        }
    }
#if DSDP_SDP_DBG
    // ck consistency.
    for (int k = 0; k < nvar + 1; ++k) {
        fprintf(stdout,
                "@showing constraint %d, dual-obj: %.3f \n",
                k,
                1.0
        );
        SDPConeViewDataMatrix(sdpcone, 0, k);
    }
    fprintf(stdout, "@showing slacks \n");
    BConeView(bcone);
#endif
    // set parameters
    int info;
    //    info = DSDPSetGapTolerance(dsdp, 0.001);
    //    info = DSDPSetPotentialParameter(dsdp, 5);
    //    info = DSDPReuseMatrix(dsdp, 0);
    //    info = DSDPSetPNormTolerance(dsdp, 1.0);
    //    info = DSDPSetPenaltyParameter(p, nvar);
    DSDPSetPenaltyParameter(p, 1e4);
#if DSDP_SDP_DBG
    DSDPSetStandardMonitor(p, 1);
#else
    DSDPSetStandardMonitor(p, -1);
#endif

    info = DSDPSetup(p);
}


void QP_DSDP::extract_solution() {
    int info;
    if (!bool_solved) throw std::exception();
    __unused int xsize;


    // tempo buffers
    // dynamic arrays for solution
    double *slack = new double[nvar]{0.0};
    double *surplus = new double[nvar]{0.0};
    double *xx;
    double *xmat = new double[ndim * ndim]{0.0};
    double *ymat = new double[ndim * ndim]{0.0};
    r.y = new double[nvar]{0.0};
    r.x = new double[r.n];
    r.D = new double[r.n];
    r.S = new double[m_with_cuts];
    r.ydim = nvar;
    eigen_matmap ymat_s(ymat, ndim, ndim);
    // solutions
    DSDPSolutionType pdfeasible;
    DSDPTerminationReason reason;
    DSDPGetSolutionType(p, &pdfeasible);
    DSDPStopReason(p, &reason);
    info = DSDPComputeX(p);
    SDPConeGetXArray(sdpcone, 0, &xx, &xsize);
    DSDPGetY(p, r.y, nvar);
    BConeCopyX(bcone, surplus, slack, nvar);
    // primal
    input_lower_triangular(xx, xmat, ndim);
    r.save_to_X(xmat);

    for (int i = 0; i < r.n; ++i) {
        r.x[i] = r.Xm(r.n, i);
    }
    for (int i = 0; i < n; ++i) {
        int vari = i + 1;
        ymat_s -= qp.Qd[i].block(0, 0, ndim, ndim) * r.y[vari];
    }
    for (int i = 0; i < m; ++i) {
        int vari = i + n + 1;
        ymat_s -= qp.Ah[i] * r.y[vari];
    }
    for (int i = m; i < m_with_cuts; ++i) {
        int vari = i + n + 1;
        ymat_s -= cp[i - m].B * r.y[vari];
    }
    ymat_s(ndim - 1, ndim - 1) -= r.y[0];
    ymat_s -= qp.Qh;
    r.save_to_Y(ymat);
    for (int i = 1; i <= n; ++i) {
        r.D[i - 1] = slack[i];
    }
    for (int i = n + 1; i <= n + m_with_cuts; ++i) {
        r.S[i - n - 1] = slack[i];
    }
    // compute residual
    eigen_const_arraymap xm(r.x, n);

    r.Res = (r.Xm.block(0, 0, n, n) - xm.matrix() * xm.matrix().adjoint()).cwiseAbs();
    // compute primal dual values
    // objectives
    DSDPGetDObjective(p, &bound);
    bound = -bound; // fix sense
    primal = qp.inhomogeneous_obj_val(r.x);

#if DSDP_SDP_DBG
    std::cout << dsdp_status(pdfeasible) << std::endl;
    std::cout << dsdp_stopreason(reason) << std::endl;
#endif
    delete[] slack;
    delete[] surplus;
}

QP_DSDP::QP_DSDP(QP &qp) : Backend(qp), qp(qp), r(qp.n, qp.m, qp.d), n(qp.n), m(qp.m) {
    // number of variables (for original)
    // problem size
    ndim = n + 1; // homogeneous
    n_p_square = (n + 1) * (n + 1); // # nnz in the full matrix
    n_lower_tr = (n + 1) * (n + 2) / 2; // # nnz in the lower triangular block
}

void QP_DSDP::setup() {
    // total number of constrs;
    // - 1 for ynn = 1
    // - n diagonal constr: y[i, i] <= x[i]
    // - m_with_cuts
    //      - m quadratic constrs
    //      - cp.size() cuts
    m_with_cuts = cp.size() + m;
    nvar = n + m_with_cuts + 1;
    // dynamic arrays
    _tilde_q_data = new double[n_lower_tr]{0.0};
    _one_idx[0] = n_lower_tr - 1;
    _ei_val[0] = 1.0;
    _ei_val[1] = -0.5;
    _ei_idx = new int[2 * n]{0};
    _ah_data = new double[n_lower_tr * m_with_cuts]{0.0};

    bool_setup = true;

}

void QP_DSDP::optimize() {
    int info = DSDPSolve(p);
    bool_solved = true;
}

QP_DSDP::~QP_DSDP() {
    if (bool_setup) {
        try {
            DSDPDestroy(p);
        }
        catch (const char* msg){
            std::cout << msg << std::endl;
        }
        delete[] _tilde_q_data;
        delete[] _ei_idx;
        delete[] _ah_data;
    }// frees
}

void QP_DSDP::assign_initial_point(Result_DSDP &r_another, bool dual_only) const {
    DSDPSetR0(p, r_another.r0);
    if (nvar != r_another.ydim) {
        throw std::exception();
    }
    for (int i = 0; i < r_another.ydim; i++) {
        DSDPSetY0(p, i + 1, r_another.y[i]);
    }
    eigen_arraymap ym(r_another.y, r_another.ydim);
    std::cout << ym << std::endl;
//    DSDPSetPenaltyParameter(p, 1e4);

    // check psd?
//    eigen_matrix ymat = eigen_matrix ::Zero(ndim, ndim);
//    for (int i = 0; i < n; ++i) {
//        int vari = i + 1;
//        ymat -= qp.Qd[i].block(0, 0, ndim, ndim) * r_another.y[vari];
//    }
//    for (int i = 0; i < m; ++i) {
//        int vari = i + n + 1;
//        ymat -= qp.Ah[i] * r_another.y[vari];
//    }
//    for (int i = m; i < m_with_cuts; ++i) {
//        int vari = i + n + 1;
//        ymat -= cp[i - m].B * r_another.y[vari];
//    }
//    ymat(ndim - 1, ndim - 1) -= r_another.y[0];
//    ymat -= qp.Qh;
//    std::cout << ymat.format(EIGEN_IO_FORMAT) << std::endl;
//    std::cout << 1 << std::endl;
}

Result_DSDP QP_DSDP::get_solution() const {
    return Result_DSDP{r};
}


void Result_DSDP::construct_init_point(Result_DSDP &r, double lambda, int pool_size) {
    ydim = n + m + pool_size + 1;
    y = new double[ydim]{0};
    r0 = 1e-2;
    for (int i = 0; i < r.ydim; i++) {
        y[i] = r.y[i];
    }
//    for (int i = r.ydim; i < ydim; i++){
//        y[i] = -0.1;
//    }
}

Result_DSDP::Result_DSDP(int n, int m, int d) :
        Result(n, m, d) {
    r0 = 0;
}

