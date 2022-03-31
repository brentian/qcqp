//
// Created by C. Zhang on 2021/7/25.
// a script for IPM-trust region method.
//


#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <iostream>
#include "qcqp.h"
#include "io.h"
#include "bb_copt.h"

int trs(const Bg_COPT &p, const Bound &bound, Result_COPT &r, copt_prob *prob) {
  int errcode;
  // [x,d, r, f]
  int ncol = p.qp.n + 2 * p.ydim + 1;
  std::vector<char *> names(ncol);
  // add variables
  {
    // x
    errcode = COPT_AddCols(
        prob,
        p.qp.n,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        bound.xlb.data(),
        bound.xub.data(),
        names.data()
    );
    // d
    // zlb - z <= d <= zub - z
    eigen_const_arraymap eigen_zlb(bound.zlb.data(), p.ydim);
    eigen_const_arraymap eigen_zub(bound.zub.data(), p.ydim);
    eigen_array zlb = (eigen_zlb - r.z);
    eigen_array zub = (eigen_zub - r.z);
    // bound of z fed by bound
    errcode = COPT_AddCols(
        prob,
        p.ydim,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        zlb.data(),
        zub.data(),
        names.data()
    );
    // r
    std::vector<double> ylb = std::vector<double>(p.ydim, -COPT_INFINITY);
    std::vector<double> yub = std::vector<double>(p.ydim, COPT_INFINITY);
    errcode = COPT_AddCols(
        prob,
        p.ydim,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        ylb.data(),
        yub.data(),
        names.data()
    );
    // add objective f
    errcode = COPT_AddCol(
        prob,
        1.0,
        0,
        nullptr,
        nullptr,
        'C',
        -COPT_INFINITY,
        COPT_INFINITY,
        "f"
    );
  }

  // Add linear constraints
  // [V' -I] [x d] = V'x0
  {
    eigen_sparse Vs(p.ydim, p.qp.n + p.ydim);
    eigen_matrix VsAblk(p.ydim, p.qp.n + p.ydim);
    VsAblk << p.qp.V.adjoint(), -1 * eigen_array::Ones(p.ydim).asDiagonal().toDenseMatrix();
    Vs = VsAblk.sparseView();

    std::vector<char> senses(p.ydim, COPT_EQUAL);
    std::vector<char *> lmapnames(p.ydim);

    errcode = COPT_AddRows(
        prob,
        p.ydim,
        Vs.outerIndexPtr(),
        Vs.innerNonZeroPtr(),
        Vs.innerIndexPtr(),
        Vs.valuePtr(),
        senses.data(),
        r.z.data(),
        nullptr,
        lmapnames.data()
    );
  }

  // Add the quadratic objective constraint
  {
    // construct
    // Gamma = [0 0 0;
    //          0 sigma * G - mu * D 0;
    //          0 0 -mu * D]
    eigen_matrix Gamma = eigen_matrix::Zero(ncol, ncol);
    eigen_matrix D = eigen_matrix::Identity(p.ydim, p.ydim);
    eigen_matrix G = eigen_matrix::Identity(p.ydim, p.ydim);
    double mu = 2.0;
    double sigma = 1.0;
    Gamma.block(p.qp.n + 1, p.qp.n + 1, p.ydim, p.ydim) = sigma * G - mu * D;
    Gamma.block(p.qp.n + p.ydim + 1, p.qp.n + p.ydim + 1, p.ydim, p.ydim) = -mu * D;
    eigen_sparse Gamma_sp = Gamma.sparseView();
    std::vector<int> ar;
    std::vector<int> ac;
    std::vector<double> av;

    for (int i = 0; i < Gamma_sp.outerSize(); i++)
      for (typename eigen_sparse::InnerIterator it(Gamma_sp, i); it; ++it) {
        ar.emplace_back(it.row());
        ac.emplace_back(it.col());
        av.emplace_back(it.value());
      }

    eigen_array cc = eigen_array::Zero(ncol);
    cc(Eigen::seqN(0, p.qp.n)) = - p.qp.a[0];
    cc(Eigen::seqN(p.qp.n, p.ydim)) = 2 * sigma * G * r.z;
    cc(Eigen::seqN(p.qp.n + p.ydim, p.ydim)) = - sigma * G.diagonal();
    cc[ncol - 1] = -1.0;
    // std::cout << Gamma << std::endl;
    std::cout << cc.adjoint() << std::endl;

    std::vector<int> index(ncol, 0);
    for (int i = 0; i < ncol; ++i) {
      index[i] = i;
    }
    COPT_AddQConstr(
        prob,
        ncol,
        index.data(),
        cc.data(),
        ar.size(),
        ar.data(),
        ac.data(),
        av.data(),
        COPT_GREATER_EQUAL,
        0.0,
        names[0]
    );
  }
  // Add squares of [d, r]
  {
    for (int i = 0; i < p.ydim; i++) {
      int zlidx[] = {p.qp.n + i, p.qp.n + p.ydim + i};
      int zidx[] = {p.qp.n + i};
      double zlval[] = {2 * r.z[i], -1.0};
      double zval[] = {1.0};
      COPT_AddQConstr(
          prob,
          2,
          zlidx,
          zlval,
          1,
          zidx,
          zidx,
          zval,
          COPT_LESS_EQUAL,
          r.Res(i, 0),
          names[0]
      );
      // int yidx[] = {p.qp.n + p.ydim + i};
      // COPT_AddQConstr(
      //     prob,
      //     0,
      //     zlidx,
      //     zlval,
      //     1,
      //     yidx,
      //     yidx,
      //     zval,
      //     COPT_LESS_EQUAL,
      //     0.01,
      //     names[0]
      // );
    }
  }
  COPT_SetObjSense(prob, COPT_MAXIMIZE);
  std::cout << INTERVAL_STR << std::endl;
#if COPT_TRS_DBG
  COPT_WriteLp(prob, "/tmp/1.lp");
#endif
  // Solve the problem
  COPT_Solve(prob);

  // Analyze solution
  int iLpStatus = COPT_LPSTATUS_UNSTARTED;
  errcode = COPT_GetIntAttr(prob, COPT_INTATTR_LPSTATUS, &iLpStatus);

  eigen_array xall(ncol);
  // Retrieve optimal solution
  if (iLpStatus == COPT_LPSTATUS_OPTIMAL
      or iLpStatus == COPT_LPSTATUS_NUMERICAL) {
    double dLpObjVal;


    errcode = COPT_GetDblAttr(prob, COPT_DBLATTR_LPOBJVAL, &dLpObjVal);

    errcode = COPT_GetColInfo(prob, COPT_DBLINFO_VALUE, ncol, nullptr, xall.data());
  }
  eigen_array x = xall(Eigen::seqN(0, p.qp.n));
  eigen_array z = xall(Eigen::seqN(p.qp.n, p.ydim));
  eigen_array y = xall(Eigen::seqN(p.qp.n + p.ydim, p.ydim));


  // compute residual
  r.y += y;
  r.z += z;
  r.x = x;
  r.Res = r.y - r.z.cwiseProduct(r.z);
  return 1;
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    return -1;
  }
  json test = parse_json(argv[1]);
  std::string fp = std::string(argv[1]);
  std::cout << fp << std::endl;
  int m = test["m"].get<int>();
  int n = test["n"].get<int>();
  int d = test["d"].get<int>();

  double *Q = new double[n * n];
  double *q = new double[n];
  double *A = new double[m * n * n];
  double *a = new double[m * n];
  double *b = new double[m];
  get_arr(test, "Q", Q);
  get_arr(test, "q", q);
  get_arr(test, "A", A);
  get_arr(test, "a", a);
  get_arr(test, "b", b);
  QP qp = QP(n, m, d, Q, q, A, a, b);
  qp.setup();
  qp.convexify();
  Bound root_b(n, qp.V);
  qp.show();


  using namespace std;
  std::cout << INTERVAL_STR;
  std::cout << "first solve\n";
  std::cout << INTERVAL_STR;
  copt_env *env = nullptr;
  int errcode = COPT_CreateEnv(&env);
  if (errcode) return -1;
  Bg_COPT p(qp);
  p.create_problem(env, root_b, false, true);
  p.optimize();
  auto r = p.get_solution();

  // we start from relaxation.
  while (true) {
    copt_prob *prob = nullptr;
    errcode = COPT_CreateProb(env, &prob);
    if (errcode) return 1;
    std::cout << qp.inhomogeneous_obj_val(r.x.data()) << endl;
    // std::cout << r.Res.adjoint() << std::endl;
    // std::cout << r.x.adjoint() << std::endl;
    // std::cout << r.z.adjoint() << std::endl;
    // std::cout << r.y.adjoint() << std::endl;
    std::cout << r.Res.maxCoeff() << std::endl;
    if (r.Res.maxCoeff() < 1e-4){
      break;
    }
    int trs_info = trs(p, root_b, r, prob);
  }
  delete[] Q;
  delete[] q;
  delete[] A;
  delete[] a;
  delete[] b;
  return 1;
}
