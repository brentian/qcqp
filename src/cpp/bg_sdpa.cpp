//
// Created by C. Zhang on 2021/7/25.
//

#include "bg_sdpa.h"


void input_block(int k, int l, int size, SDPA &p, eigen_matrix &Q) {
    for (int i = 0; i < size; ++i) {
        for (int j = i; j < size; ++j) {
            p.inputElement(k, l, i + 1, j + 1, Q(i, j));
        }
    }
}

void input_x_init(SDPA &p, eigen_const_arraymap &v) {
    for (int i = 0; i < v.size(); ++i) {
        p.inputInitXVec(i + 1, v[i]);
    }
}

void input_X_init(SDPA &p, eigen_const_matmap &v) {
    for (int i = 0; i < v.rows(); ++i) {
        for (int j = 0; j < v.cols(); ++j) {
            p.inputInitXMat(1, i + 1, j + 1, v(i, j));
        }
    }
}

void input_Y_init(SDPA &p, eigen_const_matmap &v) {
    for (int i = 0; i < v.rows(); ++i) {
        for (int j = 0; j < v.cols(); ++j) {
            p.inputInitYMat(1, i + 1, j + 1, v(i, j));
        }
    }
}

void display_solution_dtls(SDPA &p) {

    fprintf(stdout, "\nStop iteration = %d\n",
            p.getIteration());
    char phase_string[30];
    p.getPhaseString(phase_string);
    fprintf(stdout, "Phase          = %s\n", phase_string);
    fprintf(stdout, "objValPrimal   = %+10.6e\n",
            p.getPrimalObj());
    fprintf(stdout, "objValDual     = %+10.6e\n",
            p.getDualObj());
    fprintf(stdout, "p. feas. error = %+10.6e\n",
            p.getPrimalError());
    fprintf(stdout, "d. feas. error = %+10.6e\n\n",
            p.getDualError());


    fprintf(stdout, "xVec = \n");
    // p.printResultXVec();
    printVector(p.getResultXVec(),
                p.getConstraintNumber(), (char *) "%+8.3e",
                stdout);

    fprintf(stdout, "xMat = \n");
    // p.printResultXMat();
    for (int l = 0; l < p.getBlockNumber(); ++l) {
        if (p.getBlockType(l + 1) == SDPA::SDP) {
            printMatrix(p.getResultXMat(l + 1),
                        p.getBlockSize(l + 1), (char *) "%+8.3e",
                        stdout);
        } else if (p.getBlockType(l + 1) == SDPA::SOCP) {
            printf("current version does not support SOCP\n");
        }
        if (p.getBlockType(l + 1) == SDPA::LP) {
            printVector(p.getResultXMat(l + 1),
                        p.getBlockSize(l + 1), (char *) "%+8.3e",
                        stdout);
        }
    }

    fprintf(stdout, "yMat = \n");
    // p.printResultYMat();
    for (int l = 0; l < p.getBlockNumber(); ++l) {
        if (p.getBlockType(l + 1) == SDPA::SDP) {
            printMatrix(p.getResultYMat(l + 1),
                        p.getBlockSize(l + 1), (char *) "%+8.3e",
                        stdout);
        } else if (p.getBlockType(l + 1) == SDPA::SOCP) {
            printf("current version does not support SOCP\n");
        }
        if (p.getBlockType(l + 1) == SDPA::LP) {
            printVector(p.getResultYMat(l + 1),
                        p.getBlockSize(l + 1), (char *) "%+8.3e",
                        stdout);
        }
    }

    double dimacs_error[7];
    p.getDimacsError(dimacs_error);
    printDimacsError(dimacs_error, (char *) "%+8.3e", stdout);
    fprintf(stdout, "total time = \n");
    p.printComputationTime(stdout);

}


Result_SDPA::Result_SDPA(const int i, const int i1, const int i2) : Result(i, i1, i2) {
}

QP_SDPA::QP_SDPA(QP &qp) : qp(qp), r(qp.n, qp.m, qp.d) {
}

void QP_SDPA::solve_sdpa_p(bool verbose) {
    p.initializeUpperTriangle();
    p.initializeSolve();
    p.solve();
    solved = true;
    if (verbose) {
        display_solution_dtls(p);
    }
}

void QP_SDPA::create_sdpa_p(bool solve, bool verbose) {


    // All parameteres are renewed
    p.setParameterType(SDPA::PARAMETER_DEFAULT);

    // If necessary, each parameter can be set independently
    // p.setParameterMaxIteration(100);
    // p.setParameterEpsilonStar(1.0e-7);
    // p.setParameterLambdaStar(1.0e+2);
    // p.setParameterOmegaStar(2.0);
    // p.setParameterLowerBound(-1.0e+5);
    // p.setParameterUppwerBound(1.0e+5);
    // p.setParameterBetaStar(0.1);
    // p.setParameterBetaBar(0.2);
    // p.setParameterGammaStar(0.9);
    // p.setParameterEpsilonDash(1.0e-7);
    // p.setParameterPrintXVec((char*)"%+8.3e" );
    // p.setParameterPrintXMat((char*)"%+8.3e" );
    // p.setParameterPrintYMat((char*)"%+8.3e" );
    // p.setParameterPrintInformation((char*)"%+10.16e");
    if (verbose) {
        SDPA::printSDPAVersion(stdout);
        p.setDisplay(stdout);
        p.printParameters(stdout);
    }
    int m = qp.m;
    int n = qp.n;
    p.inputConstraintNumber(1 + n + m);
    p.inputBlockNumber(2);
    p.inputBlockType(1, SDPA::SDP);
    p.inputBlockType(2, SDPA::LP);
//    p.inputBlockType(3, SDPA::LP);
    p.inputBlockSize(1, n + 1);
    p.inputBlockSize(2, -n);
//    p.inputBlockSize(3, -m);
    p.initializeUpperTriangleSpace();

    // Q
    input_block(0, 1, n + 1, p, qp.Qh);
    // Y[n, n] = 1
    p.inputElement(1, 1, n + 1, n + 1, 1);
    p.inputCVec(1, 1);
    // diagonal Y <= xx^T
    for (int k = 0; k < n; ++k) {
        eigen_matrix Qt = qp.Qdiag[k].block(0, 0, n + 1, n + 1);
        input_block(k + 2, 1, n + 1, p, Qt);
        p.inputElement(k + 2, 2, k + 1, k + 1, 1);
    }

    // finish data matrices
    if (solve) {
        p.initializeUpperTriangle();
        p.initializeSolve();
        p.solve();
        solved = true;
        if (verbose) {
            display_solution_dtls(p);
        }
    }
}


/**
 * Assign double* buffer into SDPA initial solutions.
 * @param X_init primal initial point as of QCQP,
 * @param y_init dual initial point as of QCQP
 * @param Y_init dual initial point as of QCQP
 *  In SDPA (standard SDP form),
 *      the primal problem is the x∙b form, dual is Q∙Y
 *  While for QCQP,
 *      the primal is Q∙Y and the dual is of x∙b
 */
void QP_SDPA::assign_initial_point(double *X_init, const double *y_init, const double *Y_init, bool dual_only) {

    using namespace Eigen;
    eigen_const_arraymap y(y_init, qp.n);
    eigen_const_matmap Y(Y_init, qp.n + 1, qp.n + 1);
    eigen_const_matmap X(X_init, qp.n + 1, qp.n + 1);
    assign_initial_point(X, y, Y, dual_only);
}

void QP_SDPA::assign_initial_point(eigen_const_matmap X, eigen_const_arraymap y, eigen_const_matmap Y, bool dual_only) {

    p.setInitPoint(true);
    input_x_init(p, y);
    input_X_init(p, Y);
    if (!dual_only) {
        input_Y_init(p, X);
    }
}

void QP_SDPA::extract_solution() {
    if (!solved) {
        std::cerr << "has not been solved!" << std::endl;
    }
    r.X = p.getResultYMat(1);
    r.y = p.getResultXVec();
    r.Y = p.getResultXMat(1);
    r.D = p.getResultYMat(2);
//    r.S = p.getResultYMat(2);
}

Result_SDPA Result_SDPA::construct_init_point(double lambda) {
    eigen_matmap X_old(X, n+1, n+1);
    eigen_const_matmap Y_old(Y, n+1, n+1);
    eigen_const_arraymap y_old(y, n+1);
    eigen_matrix E(n+1, n+1);
    E.setIdentity();
    eigen_matrix X_init = eigen_matrix(n+1, n+1);
    X_init << X_old.matrix() * lambda + E * (1-lambda);
    eigen_matrix Y_init = eigen_matrix(n+1, n+1);
    Y_init << Y_old.matrix() * lambda + E * (1-lambda);
    eigen_matrix y_init = eigen_array :: Zero(n);

//    X_init;

    auto r = Result_SDPA(n, m, d);
    r.X = X_init.data();
    r.Y = Y_init.data();
    r.y = y_init.data();
    return r;
}

void Result_SDPA::show() {
    cout << eigen_matmap(X, n + 1, n + 1) << endl;
    cout << eigen_const_arraymap(y, n) << endl;
    cout << eigen_const_matmap(Y, n + 1, n + 1) << endl;
}
