//
// Created by C. Zhang on 2021/9/5.
//

#include "branch.h"


Bound::Bound(int n) {
  xlb = std::vector<double>(n, 0.0);
  xub = std::vector<double>(n, 1.0);
}

Bound::Bound(int n, eigen_matrix V) : Bound(n) {
  compz(n, V);
}

void Bound::compz(int n, eigen_matrix V) {
  int cols = V.cols();
  zlb = std::vector<double>(cols, 0.0);
  zub = std::vector<double>(cols, 0.0);
  eigen_arraymap _l(xlb.data(), n);
  eigen_arraymap _u(xub.data(), n);

  for (int j = 0; j < cols; ++j) {
    eigen_matrix zv(n, 2);
    zv.col(0) = V.col(j).cwiseProduct(_l);
    zv.col(1) = V.col(j).cwiseProduct(_u);
    auto z1 = zv.rowwise().maxCoeff();
    zub[j] = zv.rowwise().maxCoeff().sum();
    zlb[j] = zv.rowwise().minCoeff().sum();
  }
}

Bound::Bound(Bound &b, eigen_matrix V) : Bound(b) {

}

Bound::Bound() = default;

