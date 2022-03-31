////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2022.                                                                             /
////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Created by C. Zhang on 2022/3/8.
//

#include "bg_copt.h"

// todo the only difference in the child models
// are actually:
// 1. variable bounds,
// 2. RLT cuts
// 3. other cuts
void Bg_COPT::create_problem(copt_env *env, Bound &bound, bool solve, bool verbose) {
  int errcode;
  errcode = COPT_CreateProb(env, &prob);
  if (errcode) return;

  // [x, z, y, f]
  //  x in Rn, (z, y) in Rr, f in R
  //  where r is the number of columns used in V,
  //  to make the linear transformation:
  //  z = V'x
  std::vector<char *> names(ncol);
  // add variables
  {
    // x
    errcode = COPT_AddCols(
        prob,
        qp.n,
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
    if (errcode) return;
    // z = V'x & y = z^2
    eigen_array zylb(2 * ydim);
    eigen_array zyub(2 * ydim);
    eigen_array infty = eigen_array::Ones(ydim) * COPT_INFINITY;
    zylb << bound.zlb, -infty;
    zyub << bound.zub, infty;
    // bound of z fed by bound
    errcode = COPT_AddCols(
        prob,
        ydim * 2,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        zylb.data(),
        zyub.data(),
        names.data()
    );
    if (errcode) return;
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
    if (errcode) return;
  }

  // add linear constraints
  {
    // linear transformation :)
    // [V' -I] [x z] = 0
    eigen_sparse Vs(ydim, qp.n + ydim);
    eigen_matrix VsAblk(ydim, qp.n + ydim);
    VsAblk << qp.V.adjoint(), -1 * eigen_array::Ones(ydim).asDiagonal().toDenseMatrix();
    Vs = VsAblk.sparseView();

    std::vector<char> senses(ydim, COPT_EQUAL);
    std::vector<char *> lmapnames(ydim);
    std::vector<double> lmapb(ydim, 0.0);

    errcode = COPT_AddRows(
        prob,
        ydim,
        Vs.outerIndexPtr(),
        Vs.innerNonZeroPtr(),
        Vs.innerIndexPtr(),
        Vs.valuePtr(),
        senses.data(),
        lmapb.data(),
        nullptr,
        lmapnames.data()
    );
    if (errcode) return;
  }
  // add squares of [z, y]
  //  which are hadamard product
  //  modeled as n 2-d cones.
  {
    for (int i = 0; i < ydim; i++) {
      int yidx[] = {qp.n + ydim + i};
      int zidx[] = {qp.n + i};
      double yval[] = {-1.0};
      double zval[] = {1.0};
      COPT_AddQConstr(
          prob,
          1,
          yidx,
          yval,
          1,
          zidx,
          zidx,
          zval,
          COPT_LESS_EQUAL,
          0.0,
          names[0]
      );
      // RLT
      // also add linear constraints for y
      //  to avoid unboundedness.
      int indx[] = {qp.n + i, qp.n + ydim + i};
      double lpu[] = {-bound.zlb[i] - bound.zub[i], 1.0};
      double ltu = -bound.zlb[i] * bound.zub[i];
      COPT_AddRow(
          prob,
          2,
          indx,
          lpu,
          COPT_LESS_EQUAL,
          ltu,
          0,
          names[0]
      );
    }
  }

  // add the quadratic objective function as a constraint
  {
    eigen_array cc = eigen_array::Zero(ncol);
    cc(Eigen::seqN(0, qp.n)) = qp.a[0];
    cc(Eigen::seqN(qp.n + ydim, ydim)) = -qp.Dc[0];
    cc(ncol - 1) = 1.0;

    std::vector<int> index(ncol, 0);
    for (int i = 0; i < ncol; ++i) {
      index[i] = i;
    }
    errcode = COPT_AddQConstr(
        prob,
        ncol,
        index.data(),
        cc.data(),
        qp.Ac_rows[0].size(),
        qp.Ac_rows[0].data(),
        qp.Ac_cols[0].data(),
        qp.Ac_vals[0].data(),
        COPT_LESS_EQUAL,
        0.0,
        names[0]
    );
    if (errcode) return;
  }


  // add linear / quadratic constraints
  {
    // add other quadratic constraints
    for (int i = 1; i <= qp.m; i++) {
      // todo, add for other.
    }
  }


  // set parameters and attributes
  // note the model sense is `max` in our setting.
  {
    errcode = COPT_SetIntParam(prob, COPT_INTPARAM_LOGGING, 0);
    if (errcode) return;

    errcode = COPT_SetObjSense(prob, COPT_MAXIMIZE);
    if (errcode) return;
  }
}

void Bg_COPT::optimize() {
#if COPT_REL_DBG
  COPT_WriteLp(prob, "/tmp/1.lp");
#endif
  // Solve the problem
  int errcode = COPT_Solve(prob);

  // Analyze solution
  int iLpStatus = COPT_LPSTATUS_UNSTARTED;
  errcode = COPT_GetIntAttr(prob, COPT_INTATTR_LPSTATUS, &iLpStatus);
  if (errcode) return;

  // Retrieve optimal solution
  if (iLpStatus == COPT_LPSTATUS_OPTIMAL
      or iLpStatus == COPT_LPSTATUS_NUMERICAL) {
    double dLpObjVal;
    r.xall = eigen_array(ncol);

    errcode = COPT_GetDblAttr(prob, COPT_DBLATTR_LPOBJVAL, &dLpObjVal);
    if (errcode) return;

    errcode = COPT_GetColInfo(prob, COPT_DBLINFO_VALUE, ncol, nullptr, r.xall.data());
    if (errcode) return;

#if COPT_REL_DBG
    printf("\nOptimal objective value: %.9e\n", dLpObjVal);

    printf("Variable solution: \n");
    char colName[COPT_BUFFSIZE];
    for (int iCol = 0; iCol < ncol; ++iCol) {
      errcode = COPT_GetColName(prob, iCol, colName, COPT_BUFFSIZE, nullptr);
      if (errcode) return;
      printf("  %s = %.9e\n", colName, colVal[iCol]);
    }
#endif

    // Free memory
    r.x = r.xall(Eigen::seqN(0, qp.n));
    r.z = r.xall(Eigen::seqN(qp.n, ydim));
    r.y = r.xall(Eigen::seqN(qp.n + ydim, ydim));
    // compute residual
    r.Res = r.y - r.z.cwiseProduct(r.z);
    r.primal = qp.inhomogeneous_obj_val(r.x.data());
    r.relax = dLpObjVal;
  } else {
#if QCQP_BRANCH_DBG
    COPT_WriteLp(prob, "/tmp/1.lp");
#endif
    r.relax = -1e6;
    r.Res = eigen_matrix::Zero(1, 1);
  }
}

Bg_COPT::Bg_COPT(QP &qp) : Bg(qp), qp(qp), r(qp.n, qp.m, qp.d) {
  ydim = qp.V.cols();
  ncol = qp.n + ydim * 2 + 1;
}

Result_COPT Bg_COPT::get_solution() const {
  return Result_COPT{r};
}

void Bg_COPT::assign_initial_point(Result_COPT &r_another, bool dual_only) const {

}

void Bg_COPT::extract_solution() {

}

void Bg_COPT::setup() {

}

Result_COPT::Result_COPT(int n, int m, int d) :
    Result(n, m, d) {
}

int Pr_COPT::create_trs_copt(copt_env *env, const Bg_COPT &p, const Bound &bound) {
  return create_trs_copt(env, p, bound, r);
}

int Pr_COPT::create_trs_copt(copt_env *env, const Bg_COPT &p, const Bound &bound, Result_COPT &otherr) {
  int errcode;
  errcode = COPT_CreateProb(env, &prob);
  // [x,d, r, f]
  ncol = p.ncol;
  ydim = p.ydim;
  r.y = eigen_array(otherr.y);
  r.z = eigen_array(otherr.z);
  r.x = eigen_array(otherr.x);
  r.Res = eigen_matrix(otherr.Res);
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
    eigen_array zlb = (eigen_zlb - otherr.z);
    eigen_array zub = (eigen_zub - otherr.z);
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
        otherr.z.data(),
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
    cc(Eigen::seqN(0, p.qp.n)) = -p.qp.a[0];
    cc(Eigen::seqN(p.qp.n, p.ydim)) = 2 * sigma * G * otherr.z;
    cc(Eigen::seqN(p.qp.n + p.ydim, p.ydim)) = p.qp.Dc[0] -sigma * G.diagonal();
    cc[ncol - 1] = -1.0;

#if COPT_TRS_DBG
    std::cout << Gamma.adjoint() << std::endl;
    std::cout << cc.adjoint() << std::endl;
#endif

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
  // add squares of [d, r]
  {
    for (int i = 0; i < p.ydim; i++) {
      int zlidx[] = {p.qp.n + i, p.qp.n + p.ydim + i};
      int zidx[] = {p.qp.n + i};
      double zlval[] = {2 * otherr.z[i], -1.0};
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
          otherr.Res(i, 0),
          names[0]
      );
      // todo: limit r = dy?
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
  {
    COPT_SetIntParam(prob, COPT_INTPARAM_LOGGING, 0);
    COPT_SetObjSense(prob, COPT_MAXIMIZE);
  }
#if COPT_TRS_DBG
  std::cout << INTERVAL_STR << std::endl;
  COPT_WriteLp(prob, "/tmp/1.lp");
#endif
  return 1;
}

int Pr_COPT::iter(QP &qp, Result_COPT &otherr) {

  // Solve the problem
  COPT_Solve(prob);
  int errcode;
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
  eigen_array x = xall(Eigen::seqN(0, qp.n));
  eigen_array z = xall(Eigen::seqN(qp.n, ydim));
  eigen_array y = xall(Eigen::seqN(qp.n + ydim, ydim));


  // compute residual
  r.y += y;
  r.z += z;
  r.x = x;
  r.Res = r.y - r.z.cwiseProduct(r.z);

  return 1;
}

int Pr_COPT::optimize(copt_env *env, QP &qp, const Bg_COPT &p, const Bound &bound) {
  while (true) {
    std::cout << r.Res.maxCoeff() << "," << qp.inhomogeneous_obj_val(r.x.data()) << std::endl;
    if (r.Res.maxCoeff() < 1e-6) {
      break;
    }
    iter(qp, r);
    COPT_DeleteProb(&prob);
    prob = nullptr;

    create_trs_copt(env, p, bound);

  }
  return 1;
}

