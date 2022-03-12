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

  Qh = eigen_matrix::Zero(n + 1, n + 1);
  Qh << Q, q / 2, vtmp / 2, eigen_matrix::Zero(1, 1);

  create_diag_Q(n, m);
}

void QP::create_diag_Q(int n, int m) {
  Qd = std::vector<eigen_matrix>(n, eigen_matrix::Zero(n + 1 + n + m, n + 1 + n + m));

  int i = 0;

  for (auto &Qi: Qd) {
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
  Ar = std::vector<eigen_matrix>(m);
  ar = std::vector<eigen_array>(m);
  br = eigen_array(m);
  for (int i = 0; i < m; ++i) {
    eigen_matmap A_(A_data + i * n * n, n, n);
    eigen_arraymap a_(a_data + i * n, n, 1);
    auto Ah_ = homogenize_quadratic_form(A_, a_);
    Ah[i] = Ah_;
    Ar[i] = A_;
    ar[i] = a_;
    br[i] = b_data[i];
  }
}

void QP::show() {
  std::cout << INTERVAL_STR;
  std::cout << "visualizing QCQP instance" << std::endl;
  std::cout << "objective: " << std::endl;

  std::cout << Q << std::endl;
  std::cout << "homogenized objective: " << std::endl;
  std::cout << Qh << std::endl;
  if (verbose) {
    for (auto &Qi: Qd) {
      std::cout << "Q\n";
      std::cout << Qi << std::endl;
    }
  }
  std::cout << "homogenized constraints: " << std::endl;
  for (auto &A_: Ah) {
    std::cout << A_ << std::endl;
    std::cout << "------------------------" << std::endl;
  }
  std::cout << "homogenized constraints: (RHS)" << std::endl;
  std::cout << eigen_array(br).matrix().adjoint() << std::endl;
  std::cout << INTERVAL_STR;
}

double QP::inhomogeneous_obj_val(double *x) const {
  eigen_arraymap xm(x, n);
  auto obj = xm.dot(q) + xm.adjoint() * Q * xm;
  return obj;
}

// compute decompositions, do basic setup
void QP::setup() {
  using namespace std;
  Eigen::SelfAdjointEigenSolver<eigen_matrix> es(-Q);
  vec_es.emplace_back(es);
  A.emplace_back(Q * (-1.0));
  a.emplace_back(q * (-1.0));
  b.push_back(0.0);
  for (int i = 0; i < m; ++i) {
    Eigen::SelfAdjointEigenSolver<eigen_matrix> _es(Ar[i]);
    vec_es.push_back(_es);
    A.push_back(Ar[i]);
    a.push_back(ar[i]);
    b.push_back(br[i]);
  }
}

void QP::convexify(int method) {
  switch (method) {
    case (0): {
      // convexify via eigenvectors of Q
      //  i.e., the objective function
      V = vec_es[0].eigenvectors();
      Dc.emplace_back((-1.0) * vec_es[0].eigenvalues());
      Ac.emplace_back(eigen_matrix::Zero(n, n));
      for (int i = 1; i < m + 1; ++i) {
        // todo, finish for qcqp cases.
        eigen_matrix Ai = A[i];
        auto sigma = vec_es[i].eigenvalues().minCoeff();
        eigen_matrix cAi;
        eigen_array cDi;
        if (sigma < 0) {
          cDi = (1e-5 - sigma) * eigen_array::Ones(n);
          eigen_matrix mcDi = cDi.asDiagonal();
          cAi = Ai + mcDi;
        } else {
          cAi = Ai;
          cDi = eigen_matrix::Zero(n, n);
        }
        Ac.emplace_back(cAi);
        Dc.emplace_back(cDi);
      }
      break;
    }
    case (1): {
      break;
    }
  }
  for (int i = 0; i < m + 1; ++i) {
    eigen_sparse As = Ac[i].sparseView();
    std::vector<int> ar;
    std::vector<int> ac;
    std::vector<double> av;

    for (int i = 0; i < As.outerSize(); i++)
      for (typename eigen_sparse::InnerIterator it(As, i); it; ++it) {
        ar.emplace_back(it.row());
        ac.emplace_back(it.col());
        av.emplace_back(it.value());
      }
    Ac_rows.emplace_back(ar);
    Ac_cols.emplace_back(ac);
    Ac_vals.emplace_back(av);
  }
}

eigen_matrix homogenize_quadratic_form(eigen_matmap A, eigen_arraymap a) {

  eigen_matrix Ah(A.cols() + 1, A.cols() + 1);
  Ah << A, a.matrix() / 2, a.matrix().adjoint() / 2, eigen_matrix::Zero(1, 1);
  return Ah;
}

eigen_matrix homogenize_quadratic_form(eigen_matmap A, eigen_arraymap a, double b) {

  eigen_matrix Ah(A.cols() + 1, A.cols() + 1);
  Ah << A, a.matrix() / 2, a.matrix().adjoint() / 2, -b * eigen_matrix::Ones(1, 1);
  return Ah;
}


Backend::Backend(QP &qp) {

}
