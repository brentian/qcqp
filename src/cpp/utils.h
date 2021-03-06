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
#include "Eigen/Dense"
#include "Eigen/Core"


using eigen_array = Eigen::VectorXd;
using eigen_matrix = Eigen::MatrixXd;
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
#define DSDP_SDP_DBG 0
#endif //QCQP_UTILS_H
