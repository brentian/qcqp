////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2022.                                                                             /
////////////////////////////////////////////////////////////////////////////////////////////////////

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
    Bound root_b(n);
//    qp.show();


    using namespace std;
    std::cout << INTERVAL_STR;
    std::cout << "first solve\n";
    std::cout << INTERVAL_STR;
    QP_DSDP p(qp);
    p.create_problem(false, true);
    p.optimize();
    p.extract_solution();
    auto r = p.get_solution();

    delete[] Q;
    delete[] q;
    delete[] A;
    delete[] a;
    delete[] b;
    return 1;
}
