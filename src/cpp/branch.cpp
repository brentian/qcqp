//
// Created by C. Zhang on 2021/9/5.
//

#include "branch.h"


Bound::Bound(int n) {
  xlb = eigen_array::Zero(n);
  xub = eigen_array::Ones(n);
}

Bound::Bound(int n, eigen_matrix &V) : Bound(n) {
  compute_rotation(n, V);
}

void Bound::compute_rotation(int n, eigen_matrix &V) {
  int cols = V.cols();
  zlb = eigen_array::Zero(n);
  zub = eigen_array::Zero(n);
  eigen_arraymap _l(xlb.data(), n);
  eigen_arraymap _u(xub.data(), n);

  // todo: can we do this more elegantly?
  for (int j = 0; j < cols; ++j) {
    eigen_matrix zv(n, 2);
    zv.col(0) = V.col(j).cwiseProduct(_l);
    zv.col(1) = V.col(j).cwiseProduct(_u);
    auto z1 = zv.rowwise().maxCoeff();
    zub[j] = zv.rowwise().maxCoeff().sum();
    zlb[j] = zv.rowwise().minCoeff().sum();
  }
}

Bound::Bound(Bound &b, eigen_matrix &V) : Bound(b) {

}

Bound::Bound() = default;

