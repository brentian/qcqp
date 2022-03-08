//
// Created by C. Zhang on 2021/7/25.
//

#include "qp.h"

int main(int argc, char *argv[]) {
    double Q[] = {0., -1., 0., -3., 1., -5., -1., 0., 3., -2., -2., 0., 0., 3., 0., 0., 0., 0., -3., -2., 0., 0., 0.,
                  0.,
                  1., -2., 0.0, 0., 0., 0., -5., 0., 0., 0., 0., -6.};
    double q[] = {0, 1, 1, 1.0, 1, 0};
    QP qp = QP(6, 0, 1, Q, q);
    using namespace std;
    cout << "Q\n";
    cout << qp.Q;
    cout << "\nq\n";
    cout << qp.q;
    cout << "\nQ (homo)\n";
    cout << qp.Qh;

    qp.setup();
    return 1;
}