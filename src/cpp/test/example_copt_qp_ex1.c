/*
 * This file is part of the Cardinal Optimizer, all rights reserved.
 */

// Minimize
//  OBJ.FUNC: [ 2 X0 ^2 + 4 X0 * X1 + 4 X1 ^2
//            + 4 X1 * X2 + 4 X2 ^2
//            + 4 X2 * X3 + 4 X3 ^2
//            + 4 X3 * X4 + 4 X4 ^2
//            + 4 X4 * X5 + 4 X5 ^2
//            + 4 X5 * X6 + 4 X6 ^2
//            + 4 X6 * X7 + 4 X7 ^2
//            + 4 X7 * X8 + 4 X8 ^2
//            + 4 X8 * X9 + 2 X9 ^2 ] / 2
// Subject To
//  ROW0: X0 + 2 X1 + 3 X2  = 1
//  ROW1: X1 + 2 X2 + 3 X3  = 1
//  ROW2: X2 + 2 X3 + 3 X4  = 1
//  ROW3: X3 + 2 X4 + 3 X5  = 1
//  ROW4: X4 + 2 X5 + 3 X6  = 1
//  ROW5: X5 + 2 X6 + 3 X7  = 1
//  ROW6: X6 + 2 X7 + 3 X8  = 1
//  ROW7: X7 + 2 X8 + 3 X9  = 1
// Bounds
//       X0 Free
//       X1 Free
//       X2 Free
//       X3 Free
//       X4 Free
//       X5 Free
//       X6 Free
//       X7 Free
//       X8 Free
//       X9 Free
// End

#include "copt.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int errcode = 0;
  copt_env *env = NULL;
  copt_prob *prob = NULL;

  // Create COPT environment and problem
  errcode = COPT_CreateEnv(&env);
  if (errcode) goto COPT_EXIT;
  errcode = COPT_CreateProb(env, &prob);
  if (errcode) goto COPT_EXIT;

  // Add variables
  int nCol = 10;
  double colCost[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double colLower[] = {-COPT_INFINITY, -COPT_INFINITY, -COPT_INFINITY,
                       -COPT_INFINITY, -COPT_INFINITY, -COPT_INFINITY,
                       -COPT_INFINITY, -COPT_INFINITY, -COPT_INFINITY,
                       -COPT_INFINITY};
  double colUpper[] = {+COPT_INFINITY, +COPT_INFINITY, +COPT_INFINITY,
                       +COPT_INFINITY, +COPT_INFINITY, +COPT_INFINITY,
                       +COPT_INFINITY, +COPT_INFINITY, +COPT_INFINITY,
                       +COPT_INFINITY};
  char *colNames[] = {"X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"};
  errcode = COPT_AddCols(prob, nCol, colCost, NULL, NULL, NULL, NULL, NULL,
                         colLower, colUpper, colNames);
  if (errcode) goto COPT_EXIT;

  // Add constraints
  int nRow = 8;
  int rowBeg[] = {0, 3, 6, 9, 12, 15, 18, 21};
  int rowCnt[] = {3, 3, 3, 3, 3, 3, 3, 3};
  int rowIdx[] = {0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5,
                  4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9};
  double rowElem[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
                      1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
                      1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
                      1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
  char rowSense[] = {COPT_EQUAL, COPT_EQUAL, COPT_EQUAL, COPT_EQUAL,
                     COPT_EQUAL, COPT_EQUAL, COPT_EQUAL, COPT_EQUAL};
  double rowRhs[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  char *rowNames[] = {"ROW0", "ROW1", "ROW2", "ROW3",
                      "ROW4", "ROW5", "ROW6", "ROW7"};
  errcode = COPT_AddRows(prob, nRow, rowBeg, rowCnt, rowIdx, rowElem, rowSense,
                         rowRhs, NULL, rowNames);
  if (errcode) goto COPT_EXIT;

  // Set quadratic objective
  int nQElem = 19;
  int qRow[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9};
  int qCol[] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9};
  double qElem[] = {1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1};
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
