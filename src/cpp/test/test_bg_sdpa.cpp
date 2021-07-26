//
// Created by C. Zhang on 2021/7/25.
//

#include "bg_sdpa.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <iostream>

/* RUN AN QP INSTANCE
 * Q:
    [[ 0., -1.,  0., -3.,  1., -5.],
    [-1.,  0.,  3., -2., -2.,  0.],
    [ 0.,  3.,  0.,  0.,  0.,  0.],
    [-3., -2.,  0.,  0.,  0.,  0.],
    [ 1., -2.,  0.,  0.,  0.,  0.],
    [-5.,  0.,  0.,  0.,  0., -6.]]
 * q:
    [0, 1, 1, 1, 1, 0]
 PYTHON BENCHMARKS
   prob_num  solve_time  best_bound  best_obj  relax_obj  nodes method
    0    6:0:0        0.00         8.0      8.00       8.00    0.0    grb
    1    6:0:0        0.00         0.0      0.00       8.47    0.0   shor
    2    6:0:0        0.01         0.0      0.00       8.47    0.0  eshor
    3    6:0:0        0.00         0.0   -104.30       8.47    0.0  dshor
    4    6:0:0        0.00         0.0      6.69       8.76    0.0   emsc
    5    6:0:0        0.00         0.0      5.76       8.86    0.0   ssdp
 * */
int main(int argc, char *argv[]) {

    double Q[] = {0., -1., 0., -3., 1., -5., -1., 0., 3., -2., -2., 0., 0., 3., 0., 0., 0., 0., -3., -2., 0., 0., 0.,
                  0.,
                  1., -2., 0.0, 0., 0., 0., -5., 0., 0., 0., 0., -6.};
    double q[] = {0, 1, 1, 1.0, 1, 0};
    QP qp = QP(6, 0, 1, Q, q);
    using namespace std;

    qp_sdpa qp_sdpa(qp);
    qp_sdpa.create_sdpa_p();
    return 0;
}
