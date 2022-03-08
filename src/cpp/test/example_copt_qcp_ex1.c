/*
 * This file is part of the Cardinal Optimizer, all rights reserved.
 */

// Minimize
//  obj: 2.1 x1 - 1.2 x2 + 3.2 x3 + x4 + x5 + x6 + 2 x7 + [ x2^2 ] / 2
// Subject To
//  r1: x1 + 2 x2 = 6
//  r2: 2 x1 + x3 >= 5
//  r3: x6 + 2 x7 <= 7
//  r4: -x1 + 1.2 x7 >= -2.3
//  q1: [ -1.8 x1^2 + x2^2 ] <= 0
//  q2: [ 4.25 x3^2 - 2 x3 * x4 + 4.25 x4^2 - 2 x4 * x5 + 4 x5^2  ] + 2 x1 + 3 x3 <= 9.9
//  q3: [ x6^2 - 2.2 x7^2 ] >= 5
// Bounds
//  0.2 <= x1 <= 3.8
//  x2 Free
//  0.1 <= x3 <= 0.7
//  x4 Free
//  x5 Free
//  x7 Free
// End

#include "copt.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
  int errcode = 0;
  copt_env* env = NULL;
  copt_prob* prob = NULL;

  // Create COPT environment and problem
  errcode = COPT_CreateEnv(&env);
  if (errcode) goto COPT_EXIT;
  errcode = COPT_CreateProb(env, &prob);
  if (errcode) goto COPT_EXIT;

  // Add variables
  int nCol = 7;
  double colCost[] = {2.1, -1.2, 3.2, 1.0, 1.0, 1.0, 2.0};
  double colLower[] = {0.2, -COPT_INFINITY, 0.1, -COPT_INFINITY, -COPT_INFINITY,
                       0.0, -COPT_INFINITY};
  double colUpper[] = {3.8, +COPT_INFINITY, 0.7, +COPT_INFINITY, +COPT_INFINITY,
                       +COPT_INFINITY, +COPT_INFINITY};
  char *colNames[] = {"x1", "x2", "x3", "x4", "x5", "x6", "x7"};
  errcode = COPT_AddCols(prob, nCol, colCost, NULL, NULL, NULL, NULL, NULL,
                         colLower, colUpper, colNames);
  if (errcode) goto COPT_EXIT;

  // Add linear constraints
  int nRow = 4;
  int rowBeg[] = {0, 2, 4, 6};
  int rowCnt[] = {2, 2, 2, 2};
  int rowIdx[] = {0, 1, 0, 2, 5, 6, 0, 6};
  double rowElem[] = {1.0, 2.0, 2.0, 1.0, 1.0, 2.0, -1.0, 1.2};
  char rowSense[] = {COPT_EQUAL, COPT_GREATER_EQUAL, COPT_LESS_EQUAL, COPT_GREATER_EQUAL};
  double rowRhs[] = {6.0, 5.0, 7.0, -2.3};
  char *rowNames[] = {"r1", "r2", "r3", "r4"};
  errcode = COPT_AddRows(prob, nRow, rowBeg, rowCnt, rowIdx, rowElem, rowSense,
                         rowRhs, NULL, rowNames);
  if (errcode) goto COPT_EXIT;

  // Add quadratic constraints
  {
    int nRowMatCnt = 0;
    int *rowMatIdx = NULL;
    double *rowMatElem = NULL;
    int nQMatCnt = 2;
    int qMatRow[] = {0, 1};
    int qMatCol[] = {0, 1};
    double qMatElem[] = {-1.8, 1.0};
    char cRowSense = COPT_LESS_EQUAL;
    double dRowBound = 0.0;
    char *name = "q1";
    errcode = COPT_AddQConstr(prob, nRowMatCnt, rowMatIdx, rowMatElem,
                              nQMatCnt, qMatRow, qMatCol, qMatElem,
                              cRowSense, dRowBound, name);
    if (errcode) goto COPT_EXIT;
  }

  {
    int nRowMatCnt = 2;
    int rowMatIdx[] = {0, 2};
    double rowMatElem[] = {2.0, 3.0};
    int nQMatCnt = 5;
    int qMatRow[] = {2, 2, 3, 3, 4};
    int qMatCol[] = {2, 3, 3, 4, 4};
    double qMatElem[] = {4.25, -2.0, 4.25, -2.0, 4.0};
    char cRowSense = COPT_LESS_EQUAL;
    double dRowBound = 9.9;
    char *name = "q2";
    errcode = COPT_AddQConstr(prob, nRowMatCnt, rowMatIdx, rowMatElem,
                              nQMatCnt, qMatRow, qMatCol, qMatElem,
                              cRowSense, dRowBound, name);
    if (errcode) goto COPT_EXIT;
  }

  {
    int nRowMatCnt = 0;
    int *rowMatIdx = NULL;
    double *rowMatElem = NULL;
    int nQMatCnt = 2;
    int qMatRow[] = {5, 6};
    int qMatCol[] = {5, 6};
    double qMatElem[] = {1.0, -2.2};
    char cRowSense = COPT_GREATER_EQUAL;
    double dRowBound = 5.0;
    char *name = "q3";
    errcode = COPT_AddQConstr(prob, nRowMatCnt, rowMatIdx, rowMatElem,
                              nQMatCnt, qMatRow, qMatCol, qMatElem,
                              cRowSense, dRowBound, name);
    if (errcode) goto COPT_EXIT;
  }

  // Set quadratic objective
  int nQElem = 1;
  int qRow[] = {1};
  int qCol[] = {1};
  double qElem[] = {0.5};
  errcode = COPT_SetQuadObj(prob, nQElem, qRow, qCol, qElem);
  if (errcode) goto COPT_EXIT;

  // Set parameters and attributes
  errcode = COPT_SetDblParam(prob, COPT_DBLPARAM_TIMELIMIT, 60);
  if (errcode) goto COPT_EXIT;

  errcode = COPT_SetObjSense(prob, COPT_MINIMIZE);
  if (errcode) goto COPT_EXIT;

  // Solve the problem
  errcode = COPT_Solve(prob);

  // Analyze solution
  int iLpStatus = COPT_LPSTATUS_UNSTARTED;
  errcode = COPT_GetIntAttr(prob, COPT_INTATTR_LPSTATUS, &iLpStatus);
  if (errcode) goto COPT_EXIT;

  // Retrieve optimal solution
  if (iLpStatus == COPT_LPSTATUS_OPTIMAL) {
    double dLpObjVal;
    double *colVal = NULL;

    colVal = (double *) malloc(nCol * sizeof(double));
    if (!colVal) {
      errcode = COPT_RETCODE_MEMORY;
      goto COPT_EXIT;
    }

    errcode = COPT_GetDblAttr(prob, COPT_DBLATTR_LPOBJVAL, &dLpObjVal);
    if (errcode) goto COPT_EXIT;

    errcode = COPT_GetColInfo(prob, COPT_DBLINFO_VALUE, nCol, NULL, colVal);
    if (errcode) goto COPT_EXIT;

    printf("\nOptimal objective value: %.9e\n", dLpObjVal);

    printf("Variable solution: \n");
    char colName[COPT_BUFFSIZE];
    for (int iCol = 0; iCol < nCol; ++iCol) {
      errcode = COPT_GetColName(prob, iCol, colName, COPT_BUFFSIZE, NULL);
      if (errcode) goto COPT_EXIT;
      printf("  %s = %.9e\n", colName, colVal[iCol]);
    }

    // Free memory
    if (colVal != NULL) {
      free(colVal);
    }
  }

  // Error handling
COPT_EXIT:
  if (errcode) {
    char errmsg[COPT_BUFFSIZE];
    COPT_GetRetcodeMsg(errcode, errmsg, COPT_BUFFSIZE);
    printf("ERROR %d: %s\n", errcode, errmsg);
    return 0;
  }

  // Delete problem and environment
  COPT_DeleteProb(&prob);
  COPT_DeleteEnv(&env);

  return 0;
}
