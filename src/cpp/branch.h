//
// Created by C. Zhang on 2021/9/5.
//

#ifndef QCQP_BRANCH_H
#define QCQP_BRANCH_H


#include "result.h"

class Bound {
public:
    std::vector<double> xlb;
    std::vector<double> xub;
    std::vector<double> zlb;
    std::vector<double> zub;

    Bound();

    explicit Bound(int n);

    explicit Bound(Bound &b, eigen_matrix V);

    explicit Bound(int n, eigen_matrix V);

    void compz(int n, eigen_matrix V);
};

template<typename ResultType>
class Branch {
public:

    int i{}, j{}, n{};
    double xpivot_val{}, xminor_val{};
    Bound left_b, right_b;

    Branch<ResultType>() = default;

    explicit Branch<ResultType>(ResultType &r, char target = 'x') {
      create_from_result(r, target);
    }

    void create_from_result(const ResultType &r, char target = 'x') {
      using namespace std;
      n = r.n;
      eigen_matrix::Index maxRow, maxCol;
      auto row_sum = r.Res.rowwise().sum();
      row_sum.maxCoeff(&maxRow, &maxCol);
      i = (int) maxRow;
      r.Res.row(maxRow).maxCoeff(&maxRow, &maxCol);
      j = (int) maxCol;
#if QCQP_BRANCH_DBG
      cout << row_sum.adjoint() << endl;
      fprintf(stdout, "pivoting on i, j: (%d, %d) ~ %.2f\n",
              i, j, r.Res(i, j));
#endif
      switch (target) {
        case 'x': {
          xpivot_val = r.x[i];
          xminor_val = r.x[j];
          break;
        }
        case 'z': {
          xpivot_val = r.z[i];
          break;
        }
      }
    }

    void imply_bounds(Bound &b, char target = 'x') {
      // copy from parent bound
      left_b = Bound(b);
      right_b = Bound(b);
      // left
      switch (target) {
        case 'x': {
          left_b.xub[i] = xpivot_val;
          // right
          right_b.xlb[i] = xpivot_val;
          break;
        }
        case 'z': {
          left_b.zub[i] = xpivot_val;
          // right
          right_b.zlb[i] = xpivot_val;
          break;
        }
      }
    }

    void imply_bounds(Bound &b, QP &qp, char target = 'x') {
      // copy from parent bound
      left_b = Bound(b, qp.V);
      right_b = Bound(b, qp.V);
      // left
      switch (target) {
        case 'x': {
          left_b.xub[i] = xpivot_val;
          // right
          right_b.xlb[i] = xpivot_val;
          break;
        }
        case 'z': {
          left_b.zub[i] = xpivot_val;
          // right
          right_b.zlb[i] = xpivot_val;
          break;
        }
      }
    }
};


#endif //QCQP_BRANCH_H
