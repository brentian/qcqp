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

    if (argc < 3) {
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
    Tree_DSDP tree;
    tree.run(qp);
    auto r = tree.best_r.top();
    r.show();
//
//    p1.create_problem();
//    auto init = std::string(argv[2]);
//    if (init.compare("F")) {
//        r1.construct_init_point(r, 0.99, p1.cp.size());
//        p1.assign_initial_point(r1, true);
//    }
//    p1.optimize();
//    p1.extract_solution();
//    check_solution(p1.r, qp, p1.cp);
//    p1.r.show();

    delete[] Q;
    delete[] q;
    delete[] A;
    delete[] a;
    delete[] b;
    return 1;
}
