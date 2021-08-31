//
// Created by C. Zhang on 2021/7/25.
//

#include "bg_sdpa.h"


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

    double dimacs_error[7];
    p.getDimacsError(dimacs_error);
    printDimacsError(dimacs_error, (char *) "%+8.3e", stdout);

}


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
        for (int j = i; j < v.cols(); ++j) {
            p.inputInitXMat(1, i + 1, j + 1, v(i, j));
        }
    }
}

void input_Dd_init(SDPA &p, eigen_const_arraymap &v) {
    for (int i = 0; i < v.rows(); ++i) {
        p.inputInitXMat(2, i + 1, i + 1, v(i));
    }
}

void input_Sd_init(SDPA &p, eigen_const_arraymap &v) {
    for (int i = 0; i < v.rows(); ++i) {
        p.inputInitXMat(3, i + 1, i + 1, v(i));
    }
}

void input_Y_init(SDPA &p, eigen_const_matmap &v) {
    for (int i = 0; i < v.rows(); ++i) {
        for (int j = i; j < v.cols(); ++j) {
            p.inputInitYMat(1, i + 1, j + 1, v(i, j));
        }
    }
}


void QP_SDPA::solve_sdpa_p(bool verbose) {
    p.initializeUpperTriangle();
    p.initializeSolve();
    if (verbose) {
        p.setDisplay(stdout);
    }
    p.solve();
    solved = true;
    if (verbose) {
        display_solution_dtls(p);
//        p.printResultXVec();
//        p.printResultXMat();
//        p.printResultYMat();
    }
}

void QP_SDPA::create_sdpa_p(bool solve, bool verbose) {
    // All parameteres are renewed
    p.setParameterType(SDPA::PARAMETER_DEFAULT);
    // actualize size
    m = qp.m;
    m_with_cuts = m + cp.size();
    r.m_with_cuts = m_with_cuts;
    int n = qp.n;
    p.inputConstraintNumber(1 + n + m_with_cuts);
    if (m > 0) {
        p.inputBlockNumber(3);
        p.inputBlockType(1, SDPA::SDP);
        p.inputBlockSize(1, n + 1);
        p.inputBlockType(2, SDPA::LP);
        p.inputBlockSize(2, -n);
        p.inputBlockType(3, SDPA::LP);
        p.inputBlockSize(3, -m_with_cuts);
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
    for (int i = 0; i < m_with_cuts; ++i) {
        if (i < m) {
            input_block(n + 2 + i, 1, n + 1, p, qp.Ah[i]);
            p.inputElement(n + 2 + i, 3, i + 1, i + 1, 1, true);
            p.inputCVec(n + 2 + i, qp.b[i]);
        } else {
            // provided by Cutpool
            Cut c = cp[i - m];
            input_block(n + 2 + i, 1, n + 1, p, c.B);
            p.inputElement(n + 2 + i, 3, i + 1, i + 1, 1, true);
            p.inputCVec(n + 2 + i, c.b);
        };
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
void QP_SDPA::assign_initial_point(Result_SDPA &r, bool dual_only) {
    //
    int m_size = qp.m + cp.size();
    p.setInitPoint(true);
    eigen_const_arraymap y(r.y, qp.n + m_size + 1);
    eigen_const_arraymap dd(r.Dd, qp.n);
    eigen_const_arraymap sd(r.Sd, m_size);
    input_x_init(p, y);
    input_X_init(p, r.Ym);
    input_Dd_init(p, dd);
    input_Sd_init(p, sd);
    if (!dual_only) {
        // do not set primal for QCQP
        // alias dual for SDPA
        input_Y_init(p, r.Xm);
    }
//    p.setParameterType(SDPA::PARAMETER_STABLE_BUT_SLOW);
//    p.setParameterLambdaStar(1e1);
//    p.setParameterEpsilonStar(1.0e-8);
    p.printParameters(stdout);
}


void QP_SDPA::extract_solution() {
    if (!solved) {
        std::cerr << "has not been solved!" << std::endl;
    }
    auto y_ = p.getResultXVec();
    auto Y_ = p.getResultXMat(1);
    r.Dd = p.getResultXMat(2);
    r.Sd = p.getResultXMat(3);

    auto X_ = p.getResultYMat(1);
    r.D = p.getResultYMat(2);
    r.S = p.getResultYMat(3);
    // dual for D and S
    r.save_to_X(X_);
    r.save_to_Y(Y_);
    r.y = y_;
    r.x = new double[r.n]{0.0};
    for (int i = 0; i < r.n; ++i) {
        r.x[i] = r.Xm(r.n, i);
    }
}

Result_SDPA QP_SDPA::get_solution() {
    return Result_SDPA(r);
}

void QP_SDPA::print_sdpa_formatted_solution() {
    p.printResultXVec();
    p.printResultXMat();
    p.printResultYMat();
}

/**
 * SDPA uses infeasible start, the initial point is:
 *  primal dual pair
 *  X∙Y = 𝜇∙I
 *  X(-1, -1) = alpha
 *  D: y = alpha, Dd, Sd (1 + n + m)
 *  P: _ = _    , D,  S
 *
 * @param r
 * @param lambda
 * @param pool_size
 */
void Result_SDPA::construct_init_point(Result_SDPA &r, double lambda, int pool_size) {
    m_with_cuts = r.m + pool_size; // total size
    int m_with_cuts_old = r.m_with_cuts; // old size
    double mu = 2;
    X = new double[(n + 1) * (n + 1)]{0.0};
    Y = new double[(n + 1) * (n + 1)]{0.0};
    y = new double[n + m_with_cuts + 1]{mu};
    D = new double[n]{mu};
    S = new double[m_with_cuts]{mu};
    Dd = new double[n]{mu};
    Sd = new double[m_with_cuts]{mu};
    // derived original x.
    x = new double[n]{0.0};


    new(&Ym) eigen_const_matmap(Y, n + 1, n + 1);
    new(&Xm) eigen_const_matmap(X, n + 1, n + 1);
    for (int i = 0; i < n + 1; ++i) {
        X[i * (n + 1) + i] += (1 - lambda) * mu;
        Y[i * (n + 1) + i] += (1 - lambda) * mu;
    }
    for (int i = 0; i < n + 1; ++i) {
        for (int j = 0; j < n + 1; ++j) {
            X[i * (n + 1) + j] += lambda * r.Xm(i, j);
            Y[i * (n + 1) + j] += lambda * r.Ym(i, j);
        }
    }
    for (int i = 0; i < n + m_with_cuts_old + 1; ++i) {
        y[i] = lambda * r.y[i] + (1 - lambda) * mu;
    };
    y[n + m_with_cuts] = mu;

    // slack with size n (diagonal)
    for (int i = 0; i < n; ++i) {
        Dd[i] = y[i + 1];
        D[i] = mu * mu / y[i + 1];
    }
    // slack with size (r.m + pool_size)
    for (int i = 0; i < m_with_cuts; ++i) {
        Sd[i] = y[i + n + 1];
        S[i] = mu * mu / y[i + n + 1];
    };

}

void Result_SDPA::show() {
    cout << "X (homo): " << endl;
    cout << Xm.format(EIGEN_IO_FORMAT) << endl;
    cout << "x: " << endl;
    cout << eigen_const_arraymap(x, n).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;

    try {
        cout << "d: " << endl;
        cout << eigen_const_arraymap(D, n).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
        cout << "s: " << endl;
        cout << eigen_const_arraymap(S, m_with_cuts).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
    }
    catch (std::exception e) {
        cout << "unsolved" << endl;
    }
    cout << "y: " << endl;
    cout << eigen_const_arraymap(y, n + m_with_cuts + 1).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
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
    cout << "Residual: Y - xx.T:" << endl;
    eigen_const_arraymap xm(x, n);
    cout << Xm.block(0, 0, n, n) - xm.matrix() * xm.matrix().adjoint() << endl;
    cout << "Comple: X∙S:" << endl;
    cout << Xm * Ym << endl;
}


void RLT_SDPA::create_from_bound(int n, int i, int j, double li, double ui, double lj, double uj) {
    B = eigen_matrix::Zero(n + 1, n + 1);
    B(i, j) = 1;
    B(n, i) = -lj;
    B(j, n) = -uj;
    b = -li * uj;
    cout << B << endl;
}
