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
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using eigen_array = Eigen::VectorXd;
using eigen_matrix = Eigen::MatrixXd;
using eigen_matmap = Eigen::Map<eigen_matrix>;
using eigen_arraymap = Eigen::Map<eigen_array>;
using eigen_const_arraymap = Eigen::Map<const eigen_array>;
using eigen_const_matmap = Eigen::Map<const eigen_matrix>;

json parse_json(char *fp);

json parse_json(const std::string &fp);

void printVector(double *ele, int dim, char *printFormat,
                 FILE *fpout);

void printMatrix(double *ele, int dim, char *printFormat,
                 FILE *fpout);

void printDimacsError(double dimacs_error[7], char *printFormat,
                      FILE *fpout);

#endif //QCQP_UTILS_H
