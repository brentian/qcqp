//
// Created by C. Zhang on 2021/7/25.
//

#include "utils.h"

json parse_json(char *fp) {
    using namespace std;
    ifstream ifs(fp);
    json _json = json::parse(ifs);
    return _json;
}

json parse_json(const std::string &fp) {
    using namespace std;
    ifstream ifs(fp);
    json _json = json::parse(ifs);
    return _json;
}

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

double *get_lower_triangular(const eigen_matrix &Q) {
    // todo, better
    int n = Q.cols();
    std::vector<double> arr(n * (n + 1) / 2);
    int ct = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            arr[ct] = Q(i, j);
            ct++;
        }
    }
    return arr.data();
}

double *input_lower_triangular(double *lowert, int n) {
    auto full_x = new double[n * n]{0.0};
    int ct = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            full_x[i * n + j] = lowert[ct];
            full_x[j * n + i] = lowert[ct];
            ct++;
        }
    }
    return full_x;
}

