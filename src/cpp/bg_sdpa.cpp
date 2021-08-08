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


//    fprintf(stdout, "xVec = \n");
//    // p.printResultXVec();
//    printVector(p.getResultXVec(),
//                p.getConstraintNumber(), (char *) "%+8.3e",
//                stdout);
//
//    fprintf(stdout, "xMat = \n");
//    // p.printResultXMat();
//    for (int l = 0; l < p.getBlockNumber(); ++l) {
//        if (p.getBlockType(l + 1) == SDPA::SDP) {
//            printMatrix(p.getResultXMat(l + 1),
//                        p.getBlockSize(l + 1), (char *) "%+8.3e",
//                        stdout);
//        } else if (p.getBlockType(l + 1) == SDPA::SOCP) {
//            printf("current version does not support SOCP\n");
//        }
//        if (p.getBlockType(l + 1) == SDPA::LP) {
//            printVector(p.getResultXMat(l + 1),
//                        p.getBlockSize(l + 1), (char *) "%+8.3e",
//                        stdout);
//        }
//    }
//
//    fprintf(stdout, "yMat = \n");
//    // p.printResultYMat();
//    for (int l = 0; l < p.getBlockNumber(); ++l) {
//        if (p.getBlockType(l + 1) == SDPA::SDP) {
//            printMatrix(p.getResultYMat(l + 1),
//                        p.getBlockSize(l + 1), (char *) "%+8.3e",
//                        stdout);
//        } else if (p.getBlockType(l + 1) == SDPA::SOCP) {
//            printf("current version does not support SOCP\n");
//        }
//        if (p.getBlockType(l + 1) == SDPA::LP) {
//            printVector(p.getResultYMat(l + 1),
//                        p.getBlockSize(l + 1), (char *) "%+8.3e",
//                        stdout);
//        }
//    }

    double dimacs_error[7];
    p.getDimacsError(dimacs_error);
    printDimacsError(dimacs_error, (char *) "%+8.3e", stdout);
//    fprintf(stdout, "total time = \n");
//    p.printComputationTime(stdout);

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
//        p.printParameters(stdout);
    }
    int m = qp.m;
    int n = qp.n;
    p.inputConstraintNumber(1 + n + m);
    if (m > 0) {
        p.inputBlockNumber(3);
        p.inputBlockType(1, SDPA::SDP);
        p.inputBlockSize(1, n + 1);
        p.inputBlockType(2, SDPA::LP);
        p.inputBlockSize(2, -n);
        p.inputBlockType(3, SDPA::LP);
        p.inputBlockSize(3, -m);
    } else {
        p.inputBlockNumber(2);
        // unconstrained
        p.inputBlockType(1, SDPA::SDP);
        p.inputBlockSize(1, n + 1);
        p.inputBlockType(2, SDPA::LP);
        p.inputBlockSize(2, -n);
    }

    p.initializeUpperTriangleSpace();

    // Q
    input_block(0, 1, n + 1, p, qp.Qh);
    // Y[n, n] = 1
    p.inputElement(1, 1, n + 1, n + 1, 1, true);
    p.inputCVec(1, 1);
    // diagonal Y <= xx^T
    for (int k = 0; k < n; ++k) {
        eigen_matrix Qd = qp.Qd[k].block(0, 0, n + 1, n + 1);
        input_block(k + 2, 1, n + 1, p, Qd);
        p.inputElement(k + 2, 2, k + 1, k + 1, 1, true);
    } // sum up to n + 2 matrices
    for (int i = 0; i < m; ++i) {
        eigen_matrix A = qp.Ah[i];
        input_block(n + 2 + i, 1, n + 1, p, A);
        p.inputElement(n + 2 + i, 3, i + 1, i + 1, 1, true);
        p.inputCVec(n + 2 + i, qp.b[i]);
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
void QP_SDPA::assign_initial_point(const double *X_init, const double *y_init, const double *Y_init, bool dual_only) {

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
    p.setParameterEpsilonStar(1.0e-8);
}

void QP_SDPA::extract_solution() {
    if (!solved) {
        std::cerr << "has not been solved!" << std::endl;
    }
    auto X_ = p.getResultYMat(1);
    auto y_ = p.getResultXVec();
    auto Y_ = p.getResultXMat(1);
    r.D = p.getResultYMat(2);
    r.S = p.getResultYMat(3);
    r.save_to_X(X_);
    r.save_to_Y(Y_);
    r.y = y_;
}

Result_SDPA QP_SDPA::get_solution() {
    return Result_SDPA(r);
}


void Result_SDPA::construct_init_point(Result_SDPA &r, double lambda) {

    X = new double[(n + 1) * (n + 1)]{0.0};
    Y = new double[(n + 1) * (n + 1)]{0.0};
    y = new double[n + m + 1]{0.0};

    new(&Ym) eigen_const_matmap(Y, n + 1, n + 1);
    new(&Xm) eigen_const_matmap(X, n + 1, n + 1);
    for (int i = 0; i < n + 1; ++i) {
        X[i * (n + 1) + i] += 1 - lambda;
        Y[i * (n + 1) + i] += 1 - lambda;
    }
    for (int i = 0; i < n + m + 1; ++i) {
        y[i] += lambda * r.y[i];
    };
    for (int i = 0; i < n + 1; ++i) {
        for (int j = 0; j < n + 1; ++j) {
            X[i * (n + 1) + j] += lambda * r.Xm(i, j);
            Y[i * (n + 1) + j] += lambda * r.Ym(i, j);
        }
    }
    S = new double[n]{0.0};
    D = new double[n]{0.0};
    x = new double[n]{0.0};
}

void Result_SDPA::show() {
    cout << "X (homo): " << endl;
    cout << Xm.format(EIGEN_IO_FORMAT) << endl;

    try {
        cout << "d: " << endl;
        cout << eigen_const_arraymap(D, n).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
        cout << "s: " << endl;
        cout << eigen_const_arraymap(S, m).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
    }
    catch (std::exception e) {
        cout << "unsolved" << endl;
    }
    cout << "y: " << endl;
    cout << eigen_const_arraymap(y, n + m + 1).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
    cout << "Y (homo): " << endl;
    cout << Ym.format(EIGEN_IO_FORMAT) << endl;
}

void Result_SDPA::check_solution(QP &qp) {
    int i = 0;
    fprintf(stdout,
            "check objectives: Q∙Y = %.3f, alpha + b∙x = %.3f\n",
            (Xm * qp.Qh).trace(),
            qp.b.dot(eigen_const_arraymap(y + n + 1, m)) + y[0]);
    for (auto Ah: qp.Ah) {
        fprintf(stdout, "check for contraint: %d, %.3f, %.3f, %.3f\n",
                i, (Xm * Ah).trace(), S[i], qp.b[i]);
        i++;
    }
}