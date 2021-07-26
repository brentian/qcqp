//
// Created by C. Zhang on 2021/7/25.
//

#include "bg_sdpa.h"


void inputBlock(int k, int l, int size, SDPA &p, eigen_matrix &Q) {
    for (int i = 0; i < size; ++i) {
        for (int j = i; j < size; ++j) {
            p.inputElement(k, l, i + 1, j + 1, Q(i, j));
        }
    }
}


void inputBlockL(int k, int i, int n, SDPA &p, eigen_array &q) {

}

void display_solution_dtls(SDPA &p) {

    fprintf(stdout, "\nStop iteration = %d\n",
            p.getIteration());
    char phase_string[30];
    p.getPhaseString(phase_string);
    fprintf(stdout, "Phase          = %s\n", phase_string);
    fprintf(stdout, "objValPrimal   = %+10.6e\n",
            p.getPrimalObj());
    fprintf(stdout, "objValDual     = %+10.6e\n",
            p.getDualObj());
    fprintf(stdout, "p. feas. error = %+10.6e\n",
            p.getPrimalError());
    fprintf(stdout, "d. feas. error = %+10.6e\n\n",
            p.getDualError());


    fprintf(stdout, "xVec = \n");
    // p.printResultXVec();
    printVector(p.getResultXVec(),
                p.getConstraintNumber(), (char *) "%+8.3e",
                stdout);

    fprintf(stdout, "xMat = \n");
    // p.printResultXMat();
    for (int l = 0; l < p.getBlockNumber(); ++l) {
        if (p.getBlockType(l + 1) == SDPA::SDP) {
            printMatrix(p.getResultXMat(l + 1),
                        p.getBlockSize(l + 1), (char *) "%+8.3e",
                        stdout);
        } else if (p.getBlockType(l + 1) == SDPA::SOCP) {
            printf("current version does not support SOCP\n");
        }
        if (p.getBlockType(l + 1) == SDPA::LP) {
            printVector(p.getResultXMat(l + 1),
                        p.getBlockSize(l + 1), (char *) "%+8.3e",
                        stdout);
        }
    }

    fprintf(stdout, "yMat = \n");
    // p.printResultYMat();
    for (int l = 0; l < p.getBlockNumber(); ++l) {
        if (p.getBlockType(l + 1) == SDPA::SDP) {
            printMatrix(p.getResultYMat(l + 1),
                        p.getBlockSize(l + 1), (char *) "%+8.3e",
                        stdout);
        } else if (p.getBlockType(l + 1) == SDPA::SOCP) {
            printf("current version does not support SOCP\n");
        }
        if (p.getBlockType(l + 1) == SDPA::LP) {
            printVector(p.getResultYMat(l + 1),
                        p.getBlockSize(l + 1), (char *) "%+8.3e",
                        stdout);
        }
    }

    double dimacs_error[7];
    p.getDimacsError(dimacs_error);
    printDimacsError(dimacs_error, (char *) "%+8.3e", stdout);
    fprintf(stdout, "total time = \n");
    p.printComputationTime(stdout);

}


void qp_sdpa::create_sdpa_p() {

    SDPA::printSDPAVersion(stdout);
    SDPA p;
    p.setDisplay(stdout);

    // All parameteres are renewed
    p.setParameterType(SDPA::PARAMETER_DEFAULT);

    // If necessary, each parameter can be set independently
    // p.setParameterMaxIteration(100);
    // p.setParameterEpsilonStar(1.0e-7);
    // p.setParameterLambdaStar(1.0e+2);
    // p.setParameterOmegaStar(2.0);
    // p.setParameterLowerBound(-1.0e+5);
    // p.setParameterUppwerBound(1.0e+5);
    // p.setParameterBetaStar(0.1);
    // p.setParameterBetaBar(0.2);
    // p.setParameterGammaStar(0.9);
    // p.setParameterEpsilonDash(1.0e-7);
    // p.setParameterPrintXVec((char*)"%+8.3e" );
    // p.setParameterPrintXMat((char*)"%+8.3e" );
    // p.setParameterPrintYMat((char*)"%+8.3e" );
    // p.setParameterPrintInformation((char*)"%+10.16e");

    p.printParameters(stdout);
    int m = qp.m;
    int n = qp.n;
    p.inputConstraintNumber(1 + n + m);
    p.inputBlockNumber(3);
    p.inputBlockType(1, SDPA::SDP);
    p.inputBlockType(2, SDPA::LP);
    p.inputBlockType(3, SDPA::LP);
    p.inputBlockSize(1, n + 1);
    p.inputBlockSize(2, -n);
    p.inputBlockSize(3, -m);
    p.initializeUpperTriangleSpace();

    // Q
    inputBlock(0, 1, n + 1, p, qp.Qh);
    // Y[n, n] = 1
    p.inputElement(1, 1, n + 1, n + 1, 1);
    p.inputCVec(1, 1);
    // diagonal Y <= xx^T
    for (int k = 0; k < n; ++k) {
        eigen_matrix Qt = qp.Qdiag[k].block(0, 0, n + 1, n + 1);
        inputBlock(k + 2, 1, n + 1, p, Qt);
//        inputBlockL(k + 2, 2, n, p,);
        p.inputElement(k + 2, 2, k + 1, k + 1, 1);
    }
    p.initializeUpperTriangle();
    p.initializeSolve();
    p.solve();

    display_solution_dtls(p);
    p.terminate();
}


