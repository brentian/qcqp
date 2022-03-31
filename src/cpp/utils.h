//
// Created by C. Zhang on 2021/7/25.
//

#ifndef QCQP_UTILS_H
#define QCQP_UTILS_H

#include <iostream>
#include <fstream>
#include <array>
#include <random>
#include <thread>
#include <future>
#include <chrono>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"


using eigen_array = Eigen::VectorXd;
using eigen_matrix = Eigen::MatrixXd;
using eigen_sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using eigen_matmap = Eigen::Map<eigen_matrix>;
using eigen_arraymap = Eigen::Map<eigen_array>;
using eigen_const_arraymap = Eigen::Map<const eigen_array>;
using eigen_const_matmap = Eigen::Map<const eigen_matrix>;

void printVector(double *ele, int dim, char *printFormat,
                 FILE *fpout);

void printMatrix(double *ele, int dim, char *printFormat,
                 FILE *fpout);

void printDimacsError(double dimacs_error[7], char *printFormat,
                      FILE *fpout);

void get_lower_triangular(const eigen_matrix &Q, double *arr);

void input_lower_triangular(const double *lowert, double *full_x, int n);

int query_index_lt(int i, int j);


eigen_sparse construct_sparse(int rowCount, int colCount, int nonZeroCount, double *nonZeroArray, int *rowIndex,
                              int *colIndex);

template<typename KeyType, typename ValueType>
std::pair<KeyType, ValueType> get_max(const std::map<KeyType, ValueType> &x) {
  using pairtype = std::pair<KeyType, ValueType>;
  return *std::max_element(x.begin(), x.end(), [](const pairtype &p1, const pairtype &p2) {
    return p1.second < p2.second;
  });
}

#ifndef QCQP_STRING_UTILS
#define QCQP_STRING_UTILS
namespace QCQPStrUtil {
    const Eigen::IOFormat _IO_FORMAT(
        4, 0, ", ", ",\n",
        "[", "]", "[", "]");
}
#define INTERVAL_STR "#############################\n"
#define EIGEN_IO_FORMAT QCQPStrUtil::_IO_FORMAT
#endif //QCQP_STRING_UTILS

//DEBUGGGING

#define QCQP_CUT_DBG 0
#define QCQP_BRANCH_DBG 0
#define DSDP_REL_DBG 0
#define COPT_REL_DBG 0
#define COPT_TRS_DBG 0

class Params {
public:
    // bool verbose=false;

    double tolgap = 1e-4;
    double tolfeas = 1e-4;
    double lb = -1e6;
    double ub = -1e6;
    double gap = 1e6;
    int interval_logging = 10;
    int interval_primal = 30;
    bool bool_warmstart = false;
    bool bool_use_primal = true;
};


#endif //QCQP_UTILS_H
