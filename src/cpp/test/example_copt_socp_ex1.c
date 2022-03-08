/*
 * This file is part of the Cardinal Optimizer, all rights reserved.
 */

#include "copt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int solve_soc() {
  int errcode = 0;

  copt_env *env = NULL;
  copt_prob *prob = NULL;

  // Create COPT environment
  errcode = COPT_CreateEnv(&env);
  if (errcode) goto COPT_EXIT;

  // Create COPT problem
  errcode = COPT_CreateProb(env, &prob);
  if (errcode) goto COPT_EXIT;

  // Add variables
  //
  //   minimize: z
  //
  //   bnds:
  //     x, y, t free, z non-negative
  //
  int ncol = 4;
  double colcost[] = {0.0, 0.0, 1.0, 0.0};
  double collb[] = {-COPT_INFINITY, -COPT_INFINITY, 0, -COPT_INFINITY};
  double colub[] = {+COPT_INFINITY, +COPT_INFINITY, +COPT_INFINITY, +COPT_INFINITY};

  errcode = COPT_AddCols(prob, ncol, colcost, NULL, NULL, NULL, NULL, NULL, 
                         collb, colub, NULL);
  if (errcode) goto COPT_EXIT;

  // Add constraints
  //
  //   r0: 3*x + y >= 1
  //   c0: z^2 >= x^2 + 2*y^2
  //
  // c0 is converted to:
  //
  //   r1: sqrt(2.0)*y - t = 0
  //   c1: z^2 >= x^2 + t^2
  //
  int nrow = 2;
  int rowbeg[] = {0, 2};
  int rowcnt[] = {2, 2};
  int rowind[] = {0, 1, 1, 3};
  double rowelem[] = {3.0, 1.0, sqrt(2.0), -1.0};
  char rowsen[] = {COPT_GREATER_EQUAL, COPT_EQUAL};
  double rowrhs[] = {1.0, 0.0};

  errcode = COPT_AddRows(prob, nrow, rowbeg, rowcnt, rowind, rowelem, rowsen, rowrhs,
                         NULL, NULL);
  if (errcode) goto COPT_EXIT;

  // Add regular cone
  int ncone = 1;
  int conetype[] = {COPT_CONE_QUAD};
  int conebeg[] = {0};
  int conecnt[] = {3};
  int coneind[] = {2, 0, 3};

  errcode = COPT_AddCones(prob, ncone, conetype, conebeg, conecnt, coneind);
  if (errcode) goto COPT_EXIT;

  // Set parameters and attributes
  errcode = COPT_SetDblParam(prob, COPT_DBLPARAM_TIMELIMIT, 10);
  if (errcode) goto COPT_EXIT;
  errcode = COPT_SetObjSense(prob, COPT_MINIMIZE);
  if (errcode) goto COPT_EXIT;

  // Solve problem
  errcode = COPT_SolveLp(prob);
  if (errcode) goto COPT_EXIT;

  // Analyze solution
  int conestat = COPT_LPSTATUS_UNSTARTED;
  double coneobjval;
  double *conesol = NULL;

  errcode = COPT_GetIntAttr(prob, COPT_INTATTR_LPSTATUS, &conestat);
  if (errcode) goto COPT_EXIT;

  if (conestat == COPT_LPSTATUS_OPTIMAL) {
    conesol = (double *) malloc(ncol * sizeof(double));

    errcode = COPT_GetLpSolution(prob, conesol, NULL, NULL, NULL);
    if (errcode) goto COPT_EXIT;

    errcode = COPT_GetDblAttr(prob, COPT_DBLATTR_LPOBJVAL, &coneobjval);
    if (errcode) goto COPT_EXIT;

    printf("\nObjective value: %.12e", coneobjval);
    printf("\nVariable solution: \n");
    for (int i = 0; i < ncol; ++i) {
      printf("  x[%d] = %.6f\n", i, conesol[i]);
    }
    printf("\n");

    free(conesol);
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

int solve_rsoc() {
  int errcode = 0;

  copt_env *env = NULL;
  copt_prob *prob = NULL;

  // Create COPT environment
  errcode = COPT_CreateEnv(&env);
  if (errcode) goto COPT_EXIT;

  // Create COPT problem
  errcode = COPT_CreateProb(env, &prob);
  if (errcode) goto COPT_EXIT;

  // Add variables
  //
  //   minimize: 1.5*x - 2*y + z
  //
  //   bnds:
  //     0 <= x <= 20
  //     y, z, r >= 0
  //     s, t free
  //
  int ncol = 6;
  double colcost[] = {1.5, -2.0, 1.0, 0.0, 0.0, 0.0};
  double collb[] = {0.0, 0.0, 0.0, 0.0, -COPT_INFINITY, -COPT_INFINITY};
  double colub[] = {20.0, COPT_INFINITY, COPT_INFINITY, COPT_INFINITY, COPT_INFINITY, COPT_INFINITY};

  errcode = COPT_AddCols(prob, ncol, colcost, NULL, NULL, NULL, NULL, NULL, 
                         collb, colub, NULL);
  if (errcode) goto COPT_EXIT;

  // Add constraints
  //
  //   r0: 2*x + y >= 2
  //   r1: -x + 2*y <= 6
  //   r2: r = 1
  //   r3: 2.8284271247 * x + 0.7071067811 * y - s = 0
  //   r4: 3.0822070014 * y - t = 0
  //   c0: 2*z*r >= s^2 + t^2
  //
  int nrow = 5;
  int rowbeg[] = {0, 2, 4, 5, 8};
  int rowcnt[] = {2, 2, 1, 3, 2};
  int rowind[] = {0, 1, 0, 1, 3, 0, 1, 4, 1, 5};
  double rowelem[] = {2.0, 1.0, -1.0, 2.0, 1.0, 2.8284271247, 0.7071067811, -1.0, 3.0822070014, -1.0};
  char rowsen[] = {COPT_GREATER_EQUAL, COPT_LESS_EQUAL, COPT_EQUAL, COPT_EQUAL, COPT_EQUAL};
  double rowrhs[] = {2.0, 6.0, 1.0, 0.0, 0.0};

  errcode = COPT_AddRows(prob, nrow, rowbeg, rowcnt, rowind, rowelem, rowsen, rowrhs,
                         NULL, NULL);
  if (errcode) goto COPT_EXIT;

  // Add rotated cone
  int ncone = 1;
  int conetype[] = {COPT_CONE_RQUAD};
  int conebeg[] = {0};
  int conecnt[] = {4};
  int coneind[] = {2, 3, 4, 5};

  errcode = COPT_AddCones(prob, ncone, conetype, conebeg, conecnt, coneind);
  if (errcode) goto COPT_EXIT;

  // Set parameters and attributes
  errcode = COPT_SetDblParam(prob, COPT_DBLPARAM_TIMELIMIT, 10);
  if (errcode) goto COPT_EXIT;
  errcode = COPT_SetObjSense(prob, COPT_MINIMIZE);
  if (errcode) goto COPT_EXIT;

  // Solve problem
  errcode = COPT_SolveLp(prob);
  if (errcode) goto COPT_EXIT;

  // Analyze solution
  int conestat = COPT_LPSTATUS_UNSTARTED;
  double coneobjval;
  double *conesol = NULL;

  errcode = COPT_GetIntAttr(prob, COPT_INTATTR_LPSTATUS, &conestat);
  if (errcode) goto COPT_EXIT;

  if (conestat == COPT_LPSTATUS_OPTIMAL) {
    conesol = (double *) malloc(ncol * sizeof(double));

    errcode = COPT_GetLpSolution(prob, conesol, NULL, NULL, NULL);
    if (errcode) goto COPT_EXIT;

    errcode = COPT_GetDblAttr(prob, COPT_DBLATTR_LPOBJVAL, &coneobjval);
    if (errcode) goto COPT_EXIT;

    printf("\nObjective value: %.12e", coneobjval);
    printf("\nVariable solution: \n");
    for (int i = 0; i < ncol; ++i) {
      printf("  x[%d] = %.6f\n", i, conesol[i]);
    }
    printf("\n");

    free(conesol);
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

int solve_mps(char *filename) {
  int errcode = 0;

  copt_env *env = NULL;
  copt_prob *prob = NULL;

  // Create COPT environment
  errcode = COPT_CreateEnv(&env);
  if (errcode) goto COPT_EXIT;

  // Create COPT problem
  errcode = COPT_CreateProb(env, &prob);
  if (errcode) goto COPT_EXIT;

  // Read SOCP from MPS file
  errcode = COPT_ReadMps(prob, filename);
  if (errcode) goto COPT_EXIT;

  // Set parameters and attributes
  errcode = COPT_SetDblParam(prob, COPT_DBLPARAM_TIMELIMIT, 10);
  if (errcode) goto COPT_EXIT;

  // Solve problem
  errcode = COPT_SolveLp(prob);
  if (errcode) goto COPT_EXIT;

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

int main(int argc, char *argv[]) {
  int errcode = 0;

  // Solve SOCP with regular cone
  errcode = solve_soc();
  if (errcode) goto COPT_EXIT;

  // Solve SOCP with rotated cone
  errcode = solve_rsoc();
  if (errcode) goto COPT_EXIT;

  // Solve SOCP from MPS file
  if (argc >= 2) {
    errcode = solve_mps(argv[1]);
    if (errcode) goto COPT_EXIT;
  }

COPT_EXIT:
  return 0;
}
