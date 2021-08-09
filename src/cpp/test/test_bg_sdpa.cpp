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


/**
 *RUN AN QP INSTANCE
      Q:
      [[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  2. ,  0. ,  0. ,  0. , -1.5],
       [ 0. ,  0. ,  0. ,  1. ,  0. ,  0. ],
       [ 0. ,  0. ,  1. ,  0. , -1. ,  1.5],
       [ 0. ,  0. ,  0. , -1. ,  2. ,  0. ],
       [ 0. , -1.5,  0. ,  1.5,  0. ,  0. ]])
      q:
         [3, 0, 3, 4, 3, 4]
      A: (2, 6, 6)
        [[[  9. ,  -1. ,   4.5,   6.5,  -1.5,  -1.5],
        [ -1. ,  -5. ,  -3.5,  -1. ,  -4.5,   3.5],
        [  4.5,  -3.5,   8. ,  -1.5,   3.5,  -2.5],
        [  6.5,  -1. ,  -1.5,  -4. ,  -2. ,   2.5],
        [ -1.5,  -4.5,   3.5,  -2. ,   0. ,  -3.5],
        [ -1.5,   3.5,  -2.5,   2.5,  -3.5,  -3. ]],
       [[  6. ,   7. ,   1. ,   0.5,   3. ,   1.5],
        [  7. ,  -1. ,  -2.5,  -4.5,  -4.5,   3.5],
        [  1. ,  -2.5,   0. ,   1. ,  -1. ,   1. ],
        [  0.5,  -4.5,   1. ,   3. ,   0.5,   1. ],
        [  3. ,  -4.5,  -1. ,   0.5, -10. ,   0. ],
        [  1.5,   3.5,   1. ,   1. ,   0. ,  -3. ]]]
      a: [[4, 4, 1, 0, 4, 2],
          [0, 2, 4, 1, 1, 0]]
      b: [18., 18.]
      PYTHON BENCHMARKS
        prob_num  solve_time  best_bound  best_obj  relax_obj  nodes method
        0    6:2:0        0.00         0.0       0.0      21.25    0.0   shor
        1    6:2:0        0.01        21.0      21.0      21.00    3.0    grb
      ONE should expect exactly the same results.
      expected solution:
      Y = [[0.9055, 0.6853, 0.9505, 0.9476, 0.9506, 0.9208],
           [0.6853, 0.6963, 0.7004, 0.681 , 0.7146, 0.6119],
           [0.9505, 0.7004, 0.9999, 0.9986, 0.9985, 0.9758],
           [0.9476, 0.681 , 0.9986, 0.999 , 0.9958, 0.981 ],
           [0.9506, 0.7146, 0.9985, 0.9958, 0.9982, 0.9691],
           [0.9208, 0.6119, 0.9758, 0.981 , 0.9691, 0.9772]];
      x = [0.9502, 0.6963, 0.9999, 0.999 , 0.9982, 0.9772]
  \param argc
  \param argv
  \return
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[]) {

    double Q[] = {0., 0., 0., 0., 0., 0., 0., 2.,
                  0., 0., 0., -1.5, 0., 0., 0., 1.,
                  0., 0., 0., 0., 1., 0., -1., 1.5,
                  0., 0., 0., -1., 2., 0., 0., -1.5,
                  0., 1.5, 0., 0.};
    double q[] = {3, 0, 3, 4, 3, 4};
    double A[] = {9., -1., 4.5, 6.5, -1.5, -1.5, -1., -5., -3.5, -1.,
                  -4.5, 3.5, 4.5, -3.5, 8., -1.5, 3.5, -2.5, 6.5,
                  -1., -1.5, -4., -2., 2.5, -1.5, -4.5, 3.5,
                  -2., 0., -3.5, -1.5, 3.5, -2.5, 2.5, -3.5, -3.,
                  6., 7., 1., 0.5, 3., 1.5, 7., -1., -2.5, -4.5, -4.5,
                  3.5, 1., -2.5, 0., 1., -1., 1.,
                  0.5, -4.5, 1., 3., 0.5, 1., 3., -4.5, -1., 0.5, -10.,
                  0., 1.5, 3.5, 1., 1., 0., -3.};
    double a[] = {4, 4, 1, 0, 4, 2, 0, 2, 4, 1, 1, 0};
    double b[] = {18, 18};
    QP qp = QP(6, 2, 1, Q, q, A, a, b);
    qp.show();
    using namespace std;
    std::cout << INTERVAL_STR;
    std::cout << "first solve\n";
    std::cout << INTERVAL_STR;
    QP_SDPA qp_sdpa(qp);
    qp_sdpa.create_sdpa_p(false, true);
    qp_sdpa.solve_sdpa_p(true);
    qp_sdpa.extract_solution();
    qp_sdpa.print_sdpa_formatted_solution();
//    qp_sdpa.p.copyCurrentToInit();
//    qp_sdpa.p.setParameterEpsilonStar(1.0e-8);
//
//    qp_sdpa.p.setInitPoint(true);
//    qp_sdpa.p.copyCurrentToInit();
//    qp_sdpa.p.solve();
    qp_sdpa.p.printParameters(stdout);

    auto r1 = qp_sdpa.get_solution();
    std::cout << INTERVAL_STR;
    std::cout << "check solution\n";
    r1.check_solution(qp);
    std::cout << INTERVAL_STR;
    std::cout << "print solution: \n";
    r1.show();
    std::cout << INTERVAL_STR;


    std::cout << INTERVAL_STR;
    std::cout << "second solve with cut\n";
    std::cout << "and run with warm-start solution\n";
    QP_SDPA qp_sdpa1(qp);
    // generate cut
    std::cout << "generate cut\n";
    std::cout << INTERVAL_STR;
    auto ct = RLT_SDPA();
    ct.create_from_bound(qp.n, 1, 1, 0, 0.6964, 0, 0.6964);
    qp_sdpa1.cp.push_back(ct);
    // generate cut finished
    qp_sdpa1.create_sdpa_p(false, false);
    std::cout << "generate initial solution\n";
    auto r = Result_SDPA(r1.n, r1.m, r1.d);
    r.construct_init_point(r1, 0.90, qp_sdpa1.cp.size());
    r.show();
    qp_sdpa1.assign_initial_point(r, true);
    qp_sdpa1.print_sdpa_formatted_solution();
    qp_sdpa1.solve_sdpa_p(true);
    qp_sdpa1.extract_solution();
    auto r2 = qp_sdpa1.get_solution();
    r2.show();
    std::cout << "finished\n";
    return 1;
}
