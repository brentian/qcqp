////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2022.                                                                             /
////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Created by C. Zhang on 2022/3/8.
//

#include "bg_copt.h"


void QP_COPT::create_problem(copt_env *env, Bound &bound, bool solve, bool verbose) {
  int errcode;
  errcode = COPT_CreateProb(env, &prob);
  if (errcode) return;

  // [x,z,y,f]
  std::vector<char *> names(ncol);
  // add variables
  std::vector<double> lb(qp.n, 0.0);
  std::vector<double> ub(qp.n, 1.0);

  errcode = COPT_AddCols(
      prob,
      qp.n,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      lb.data(),
      ub.data(),
      names.data()
  );
  //
  std::vector<double> zylb(ncol, -COPT_INFINITY);
  std::vector<double> zyub(ncol, COPT_INFINITY);
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
  // add objective
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

  // Add linear constraints
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

  // Add the quadratic objective constraint
  eigen_array cc = eigen_array::Zero(ncol);
  cc(Eigen::seqN(0, qp.n)) = qp.a[0];
  cc(Eigen::seqN(qp.n + ydim, ydim)) = -qp.Dc[0].diagonal();
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
  // Add squares of [z, y]
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
  // Add an extra simple RLT cut
  //  y^Te <= x'x


  // Add quadratic constraints
  // Add other quadratic constraints
  for (int i = 1; i <= qp.m; i++) {
    // todo, add for other.
  }


  // Set parameters and attributes
  // errcode = COPT_SetDblParam(prob, COPT_DBLPARAM_TIMELIMIT, 60);
  errcode = COPT_SetIntParam(prob, COPT_INTPARAM_LOGGING, 0);
  if (errcode) return;

  errcode = COPT_SetObjSense(prob, COPT_MAXIMIZE);
  if (errcode) return;
}

void QP_COPT::optimize() {
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
  if (iLpStatus == COPT_LPSTATUS_OPTIMAL) {
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
  }
  else{
    COPT_WriteLp(prob, "/tmp/1.lp");
    r.relax = -1e6;
    r.Res = eigen_matrix::Zero(1,1);
  }
}

QP_COPT::QP_COPT(QP &qp) : Backend(qp), qp(qp), r(qp.n, qp.m, qp.d) {
  ydim = qp.V.cols();
  ncol = qp.n + ydim * 2 + 1;
}

Result_COPT QP_COPT::get_solution() const {
  return Result_COPT{r};
}

void QP_COPT::assign_initial_point(Result_COPT &r_another, bool dual_only) const {

}

void QP_COPT::extract_solution() {

}

void QP_COPT::setup() {

}

Result_COPT::Result_COPT(int n, int m, int d) :
    Result(n, m, d) {
}
