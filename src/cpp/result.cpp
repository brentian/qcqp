//
// Created by C. Zhang on 2021/9/5.
//

#include "result.h"

Result::Result(int n, int m, int d) :
        n(n), m(m), d(d),
        Xm(nullptr, n + 1, n + 1),
        Ym(nullptr, n + 1, n + 1) {
    ydim = n + m;
}


Result::~Result() {
}

void Result::save_to_X(double *X_) {
    X = X_;
    new(&Xm) eigen_const_matmap(X_, n + 1, n + 1);
}


void Result::save_to_Y(double *Y_) {
    Y = Y_;
    new(&Ym) eigen_const_matmap(Y_, n + 1, n + 1);
}


int Result::show(bool x_only) {
    using namespace std;

    cout << "x: " << endl;
    cout << eigen_const_arraymap(x, n).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
    if (x_only) {
        return 1;
    }
    cout << "X (homo): " << endl;
    cout << Xm.format(EIGEN_IO_FORMAT) << endl;

    try {
        cout << "d: (slack for diag) " << endl;
        cout << eigen_const_arraymap(D, n).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
        cout << "s: (slack for quad c) " << endl;
        cout << eigen_const_arraymap(S, ydim - n - 1).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
    }
    catch (std::exception e) {
        cout << "unsolved" << endl;
    }
    cout << "y: " << endl;
    cout << eigen_const_arraymap(y, ydim).matrix().adjoint().format(EIGEN_IO_FORMAT) << endl;
    cout << "Y (homo): " << endl;
    cout << Ym.format(EIGEN_IO_FORMAT) << endl;
    return 1;
}

