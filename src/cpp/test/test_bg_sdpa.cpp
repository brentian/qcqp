//
// Created by C. Zhang on 2021/7/25.
// test for sdpa backend
//

#include "bg_sdpa.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <iostream>

/*
  RUN AN QP INSTANCE
  Q:
     [[ 0., -1.,  0., -3.,  1., -5.],
     [-1.,  0.,  3., -2., -2.,  0.],
     [ 0.,  3.,  0.,  0.,  0.,  0.],
     [-3., -2.,  0.,  0.,  0.,  0.],
     [ 1., -2.,  0.,  0.,  0.,  0.],
     [-5.,  0.,  0.,  0.,  0., -6.]]
  q:
     [0, 1, 1, 1, 1, 0]
 
  PYTHON BENCHMARKS
    prob_num  solve_time  best_bound  best_obj  relax_obj  nodes method
     0    6:0:0        0.00         8.0      8.00       8.00    0.0    grb
     1    6:0:0        0.00         0.0      0.00       8.47    0.0   shor
     2    6:0:0        0.01         0.0      0.00       8.47    0.0  eshor
     3    6:0:0        0.00         0.0      0.00       8.47    0.0  dshor
     4    6:0:0        0.00         0.0      6.69       8.76    0.0   emsc
     5    6:0:0        0.00         0.0      5.76       8.86    0.0   ssdp
  ONE should expect exactly the same results.
  \param argc
  \param argv
  \return
 */
int main(int argc, char *argv[]) {

    double Q[] = {0., -1., 0., -3., 1., -5., -1., 0., 3., -2., -2.,
                  0., 0., 3., 0., 0., 0., 0., -3., -2., 0., 0., 0.,
                  0., 1., -2., 0.0, 0., 0., 0., -5., 0., 0., 0., 0., -6.};
    double q[] = {0, 1, 1, 1.0, 1, 0};
    QP qp = QP(6, 0, 1, Q, q);
    using namespace std;
    std::cout << "########################\n";
    std::cout << "first solve\n";
    QP_SDPA qp_sdpa(qp);
    qp_sdpa.create_sdpa_p(false, false);
    qp_sdpa.solve_sdpa_p(true);
    qp_sdpa.extract_solution();
    std::cout << "########################\n";
    std::cout << "show some properties\n";

    eigen_const_matmap Y(qp_sdpa.r.Y, qp.n + 1, qp.n + 1);
    eigen_const_matmap X(qp_sdpa.r.X, qp.n + 1, qp.n + 1);
    std::cout << (X * Y).trace() << std::endl;

    std::cout << "########################\n";
    std::cout << "test initial solution\n";
    auto r = qp_sdpa.r.construct_init_point(0.99);
    r.show();

    std::cout << "########################\n";
    std::cout << "run with warm-start solution\n";
    QP_SDPA qp_sdpa1(qp);
    qp_sdpa1.create_sdpa_p(false, true);
    qp_sdpa1.assign_initial_point(r.X, r.y, r.Y);
    qp_sdpa1.solve_sdpa_p(true);

    return 0;
}
