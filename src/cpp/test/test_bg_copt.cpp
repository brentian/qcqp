//
// Created by C. Zhang on 2021/7/25.
// test for COPT backend
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

  std::cout << r.Res.adjoint() << std::endl;
  std::cout << r.z.adjoint() << std::endl;
  std::cout << r.y.adjoint() << std::endl;
  std::cout << "---" << std::endl;
  std::cout << (r.z - qp.V.adjoint() * r.x).adjoint() << std::endl;


  delete[] Q;
  delete[] q;
  delete[] A;
  delete[] a;
  delete[] b;
  return 1;
}
