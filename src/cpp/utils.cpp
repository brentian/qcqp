//
// Created by C. Zhang on 2021/7/25.
//

#include "utils.h"


void printVector(double *ele, int dim, char *printFormat, FILE *fpout) {
  fprintf(fpout, "[ ");
  for (int k = 0; k < dim - 1; ++k) {
    fprintf(fpout, printFormat, ele[k]);
    fprintf(fpout, " ");
  }
  fprintf(fpout, printFormat, ele[dim - 1]);
  fprintf(fpout, "]; \n");
}

void printMatrix(double *ele, int dim, char *printFormat, FILE *fpout) {
  fprintf(fpout, "[\n");
  for (int i = 0; i < dim; ++i) {
    fprintf(fpout, "[ ");
    for (int j = 0; j < dim - 1; ++j) {
      fprintf(fpout, printFormat, ele[i + dim * j]);
      fprintf(fpout, " ");
    }
    fprintf(fpout, printFormat, ele[i + dim * (dim - 1)]);
    fprintf(fpout, "]; \n");
  }
  fprintf(fpout, "]; \n");
}

void printDimacsError(double dimacs_error[7], char *printFormat,
                      FILE *fpout) {
  fprintf(fpout, "\n");
  fprintf(fpout, "* DIMACS_ERRORS * \n");
  fprintf(fpout, "err1 = ");
  fprintf(fpout, printFormat, dimacs_error[1]);
  fprintf(fpout, "  [||Ax-b|| / (1+||b||_1)]\n");
  fprintf(fpout, "err2 = ");
  fprintf(fpout, printFormat, dimacs_error[2]);
  fprintf(fpout, "  [max(0, -lambda(x)/(1+||b||_1))]\n");
  fprintf(fpout, "err3 = ");
  fprintf(fpout, printFormat, dimacs_error[3]);
  fprintf(fpout, "  [||A^Ty + z - c || / (1+||c||_1)]\n");
  fprintf(fpout, "err4 = ");
  fprintf(fpout, printFormat, dimacs_error[4]);
  fprintf(fpout, "  [max(0, -lambda(z)/(1+||c||_1))]\n");
  fprintf(fpout, "err5 = ");
  fprintf(fpout, printFormat, dimacs_error[5]);
  fprintf(fpout, "  [(<c,x> - <b,y>) / (1 + |<c,x>| + |<b,y>|)]\n");
  fprintf(fpout, "err6 = ");
  fprintf(fpout, printFormat, dimacs_error[6]);
  fprintf(fpout, "  [<x,z> / (1 + |<c,x>| + |<b,y>|)]\n");
  fprintf(fpout, "\n");
}

void get_lower_triangular(const eigen_matrix &Q, double *arr) {
  // todo, better
  int n = Q.cols();
  int ct = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      arr[ct] = Q(i, j);
      ct++;
    }
  }
}

void input_lower_triangular(const double *lowert, double *full_x, int n) {
  int ct = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      full_x[i * n + j] = lowert[ct];
      full_x[j * n + i] = lowert[ct];
      ct++;
    }
  }
}


/**
 * get index of (i, j) in n x n matrix
 * (lower triangular formatted symmetric matrix)
 * @param i
 * @param j
 * @param n
 * @return
 */
int query_index_lt(int i, int j) {
  int il, jl;
  if (i >= j) {
    il = i;
    jl = j;
  } else {
    il = j;
    jl = i;
  }
  return (il + 1) * il / 2 + jl;
}

eigen_sparse construct_sparse(
    int rowCount, int colCount, int nonZeroCount, double *nonZeroArray, int *rowIndex, int *colIndex
) {

  Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor>> spMap(rowCount, colCount, nonZeroCount, rowIndex, colIndex,
                                                                 nonZeroArray, 0);
  eigen_sparse matrix = spMap.eval();
  matrix.reserve(nonZeroCount);
  return matrix;

}


