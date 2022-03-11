//
// Created by C. Zhang on 2021/7/25.
// test for sdpa backend
//


#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <iostream>
#include "qcqp.h"
#include "io.h"
#include "bb_copt.h"

int main(int argc, char *argv[]) {

  if (argc < 3) {
    return -1;
  }
  auto bool_ws = !std::string(argv[2]).compare("T");
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
  // qp instance.
  QP qp = QP(n, m, d, Q, q, A, a, b);
  qp.setup();
  qp.convexify();
  qp.show();
  // bound instance.
  Bound root_b(n, qp.V);
  std::cout << qp.V << std::endl;
  // initialize copt
  copt_env *env = nullptr;
  int info = COPT_CreateEnv(&env);
  if (info) {
    char errmsg[COPT_BUFFSIZE];
    COPT_GetRetcodeMsg(info, errmsg, COPT_BUFFSIZE);
    fprintf(stdout, "cannot create COPT environment!, error: %s", errmsg);
    exit(info);
  }
  // create tree
  Tree_COPT tree = Tree_COPT();

  //
  Params params = Params();
  if (!bool_ws) {
    params.warmstart = false;
  }
  tree.run(qp, root_b, env, params);


  auto r = tree.best_result.top();
  r.show(true);

  delete[] Q;
  delete[] q;
  delete[] A;
  delete[] a;
  delete[] b;
  return 1;
}
