//
// Created by C. Zhang on 2021/7/25.
// test for sdpa backend
//

#include "bg_dsdp.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <iostream>

int main(int argc, char *argv[]) {
//    double Q[] = {1., 0., 0., 0., 0., 0.,
//                  0., 1., 0., 0., 0., 0.,
//                  0., 0., 1., 0., 0., 0.,
//                  0., 0., 0., 1., 0., 0.,
//                  0., 0., 0., 0., 1., 0.,
//                  0., 0., 0., 0., 0., 1.};
//
//    double q[] = {0, 0, 0, 0, 0, 0};
    double Q[] = {0., 0., 0., 0., 0., 0.,
                  0., 2., 0., 0., 0., -1.5,
                  0., 0., 0., 1., 0., 0.,
                  0., 0., 1., 0., -1., 1.5,
                  0., 0., 0., -1., 2., 0.,
                  0., -1.5, 0., 1.5, 0., 0.};
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
    QP_DSDP qp_dsdp(qp);
    qp_dsdp.create_problem();
    return 1;
}
