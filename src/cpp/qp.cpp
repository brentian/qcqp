//
// Created by C. Zhang on 2021/7/25.
//

#include "qp.h"


QP::QP(int n, int m, int d, double *Q_data, double *q_data)
        : n(n), m(m), d(d), N((n + 1) ^ 2) {
    Q = eigen_matrix(n, n);
    q = eigen_array(n);
    for (int i = 0; i <= n - 1; i++) {
        q(i) = q_data[i];
        for (int j = 0; j <= n - 1; j++)
            Q(i, j) = Q_data[i * n + j];
    }
    Eigen::Map<Eigen::RowVectorXd> vtmp(q.data(), n);
    Qhfull = eigen_matrix::Zero(n + 1 + n + m, n + 1 + n + m);
    Qh = eigen_matrix::Zero(n + 1, n + 1);
    Qh << Q, q / 2, vtmp / 2, eigen_matrix::Zero(1, 1);
    Qhfull.block(0, 0, n + 1, n + 1) = Qh;
    create_diag_Q(n, m);
}

void QP::create_diag_Q(int n, int m) {
    Qd = std::vector<eigen_matrix>(n, eigen_matrix::Zero(n + 1 + n + m, n + 1 + n + m));

    int i = 0;

    for (auto &Qi:Qd) {
        auto *e = new double[n]{0};
        auto *ee = new double[n * n]{0};
        e[i] = -0.5;
        ee[i * n + i] = 1;
        Eigen::Map<Eigen::RowVectorXd> rtmp(e, n);
        Eigen::Map<eigen_matrix> vtmp(e, n, 1);
        Eigen::Map<eigen_matrix> btmp(ee, n, n);
        eigen_matrix Qtmp(n + 1, n + 1);
        Qtmp << btmp, vtmp, rtmp, eigen_matrix::Zero(1, 1);
        Qi.block(0, 0, n + 1, n + 1) = Qtmp;
        Qi.block(n + 1, n + 1, n, n) = btmp;
        i++;
    }
}

QP::QP(int n, int m, int d, double *Q_data, double *q_data, double *A_data, double *a_data, double *b_data)
        : QP(n, m, d, Q_data, q_data) {
    Ah = std::vector<eigen_matrix>(m);
    A = std::vector<eigen_matrix>(m);
    a = std::vector<eigen_array>(m);
    b = eigen_array(m);
    for (int i = 0; i < m; ++i) {
        eigen_matmap A_(A_data + i * n * n, n, n);
        eigen_arraymap a_(a_data + i * n, n, 1);
        auto Ah_ = homogenize_quadratic_form(A_, a_);
        Ah[i] = Ah_;
        A[i] = A_;
        a[i] = a_;
        b[i] = b_data[i];

    }
}

void QP::show() {
    std::cout << INTERVAL_STR;
    std::cout << "visualizing QCQP instance" << std::endl;
    std::cout << "homogenized objective: " << std::endl;
    std::cout << Qh << std::endl;
    if (verbose) {
        for (auto &Qi:Qd) {
            std::cout << "Q\n";
            std::cout << Qi << std::endl;
        }
    }
    std::cout << "homogenized constraints: " << std::endl;
    for (auto &A_:Ah) {
        std::cout << A_ << std::endl;
        std::cout << "------------------------" << std::endl;
    }
    std::cout << "homogenized constraints: (RHS)" << std::endl;
    std::cout << eigen_array(b).matrix().adjoint() << std::endl;
    std::cout << INTERVAL_STR;
}

eigen_matrix homogenize_quadratic_form(eigen_matmap A, eigen_arraymap a) {

    eigen_matrix Ah(A.cols() + 1, A.cols() + 1);
    Ah << A, a.matrix() / 2, a.matrix().adjoint() / 2, eigen_matrix::Zero(1, 1);
    return Ah;
}

Result::Result(int n, int m, int d) :
        n(n), m(m), d(d),
        Xm(nullptr, n + 1, n + 1),
        Ym(nullptr, n + 1, n + 1) {
    ydim = m;
}


Result::~Result() {
}

void Result::save_to_X(double *X_) {
    X = X_;
    new(&Xm) eigen_const_matmap(X_, n + 1, n + 1);
}


void Result::save_to_Y(double *Y_) {
    Y = Y_;
    new(&Ym) eigen_const_matmap(Y_, n + 1, n + 1);
}


void Result::show() {
    using namespace std;
    cout << "X (homo): " << endl;
    cout << Xm.format(EIGEN_IO_FORMAT) << endl;
    cout << "x: " << endl;
    cout << eigen_const_arraymap(x, n).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;

    try {
        cout << "d: (slack for diag) " << endl;
        cout << eigen_const_arraymap(D, n).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
        cout << "s: (slack for quad c) " << endl;
        cout << eigen_const_arraymap(S, ydim - n - 1).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
    }
    catch (std::exception e) {
        cout << "unsolved" << endl;
    }
    cout << "y: " << endl;
    cout << eigen_const_arraymap(y, ydim).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
    cout << "Y (homo): " << endl;
    cout << Ym.format(EIGEN_IO_FORMAT) << endl;
}

void Result::check_solution(QP &qp) {
    using namespace std;
    int i = 0;
    cout << "Comple: X∙Y:" << endl;
    cout << (Xm * Ym).format(EIGEN_IO_FORMAT) << endl;
    cout << "Residual: X - xx.T:" << endl;
    eigen_const_arraymap xm(x, n);
    cout << (Xm.block(0, 0, n, n) - xm.matrix() * xm.matrix().adjoint()).format(EIGEN_IO_FORMAT) << endl;
    fprintf(stdout,
            "check objectives: Q∙X = %.3f, alpha + b∙z = %.3f\n",
            (Xm * qp.Qh).trace(),
            qp.b.dot(eigen_const_arraymap(y + n + 1, m)) + y[0]);
    cout << "check quad constraints..." << endl;
    for (const auto &Ah: qp.Ah) {
        fprintf(stdout, "check for constraint: %d, %.3f, %.3f, %.3f\n",
                i, (Xm * Ah).trace(), S[i], qp.b[i]);
        i++;
    }

}

void Result::check_solution(QP &qp, const CutPool &cp) {
    using namespace std;
    if (cp.empty()) {
        Result::check_solution(qp);
    } else {
        Result::check_solution(qp);
        cout << "check cuts..." << endl;
        int i = 0;
        for (const auto &c: cp) {
            fprintf(stdout, "check for cut: %d, %.3f, %.3f, %.3f\n",
                    i, (Xm * c.B).trace(), S[i + m], c.b);
            i++;
        }

    }

}
