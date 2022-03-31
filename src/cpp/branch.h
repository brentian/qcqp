//
// Created by C. Zhang on 2021/9/5.
//

#ifndef QCQP_BRANCH_H
#define QCQP_BRANCH_H


#include "result.h"

class Bound {
public:
    eigen_array xlb;
    eigen_array xub;
    eigen_array zlb;
    eigen_array zub;

    Bound();

    explicit Bound(int n);

    explicit Bound(Bound &b, eigen_matrix & V);

    explicit Bound(int n, eigen_matrix & V);

    void compute_rotation(int n, eigen_matrix & V);
};

// branching rules
#ifndef QCQP_BRANCH_AND_BOUND_RULES
#define BR_MAX_VIOLATION 0
#define BR_MAX_PRIORITY 1
#define BB_PIVOT_REL 0
#define BB_PIVOT_SPLIT 1
#endif


template<typename ResultType>
class Branch {
public:

    int i{}, j{}, n{};
    double x_pivot{}, x_minor{};
    Bound b_left, b_right;
    int type_branch{};
    int type_bound{};

    Branch<ResultType>() = default;

    explicit Branch<ResultType>(
        ResultType &r,
        char target = 'x',
        int rule = BR_MAX_VIOLATION,
        std::vector<double> priority = std::vector<double>()
    ) {
      type_branch = rule;
      switch (rule) {
        case BR_MAX_VIOLATION: {
          maximum_violation(r, target);
          break;
        }
        case BR_MAX_PRIORITY: {
          maximum_priority(r, priority, target);
          break;
        }
      }
    }

    void maximum_violation(
        const ResultType &r,
        char target = 'x'
    ) {
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
          x_pivot = r.x[i];
          x_minor = r.x[j];
          break;
        }
        case 'z': {
          x_pivot = r.z[i];
          break;
        }
      }
    }

    void maximum_priority(
        const ResultType &r,
        std::vector<double> priority,
        char target = 'x'
    ) {
      if (priority.empty()) return;
      using namespace std;
      n = r.n;
      eigen_matrix::Index maxCol;
      eigen_arraymap row_prior(priority.data(), priority.size());
      eigen_array row_sum = row_prior.cwiseProduct(r.Res.rowwise().sum());
      row_sum.maxCoeff(&maxCol);
      i = (int) maxCol;
      j = 0;
#if QCQP_BRANCH_DBG
      cout << row_sum.adjoint() << endl;
      fprintf(stdout, "pivoting on i, j: (%d, %d) ~ %.2f\n",
              i, j, r.Res(i, j));
#endif
      switch (target) {
        case 'x': {
          x_pivot = r.x[i];
          x_minor = r.x[j];
          break;
        }
        case 'z': {
          x_pivot = r.z[i];
          break;
        }
      }
    }

    void imply_bounds(Bound &b, char target = 'x') {
      // copy from parent bound
      b_left = Bound(b);
      b_right = Bound(b);
      // left
      switch (target) {
        case 'x': {
          b_left.xub[i] = x_pivot;
          // right
          b_right.xlb[i] = x_pivot;
          break;
        }
        case 'z': {
          b_left.zub[i] = x_pivot;
          // right
          b_right.zlb[i] = x_pivot;
          break;
        }
      }
    }

    void imply_bounds(Bound &b, QP &qp, char target = 'x') {
      // copy from parent bound
      b_left = Bound(b, qp.V);
      b_right = Bound(b, qp.V);
      // left
      switch (target) {
        case 'x': {
          b_left.xub[i] = x_pivot;
          // right
          b_right.xlb[i] = x_pivot;
          break;
        }
        case 'z': {
          b_left.zub[i] = x_pivot;
          // right
          b_right.zlb[i] = x_pivot;
          break;
        }
      }
    }
};


#endif //QCQP_BRANCH_H
