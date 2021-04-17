#  MIT License
#
#  Copyright (c) 2021 Cardinal Operations
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of
#  this software and associated documentation files (the "Software"), to deal in
#  the Software without restriction, including without limitation the rights to
#  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
#  of the Software, and to permit persons to whom the Software is furnished to do
#  so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import cvxpy as cp
import numpy as np


# https://www.cvxpy.org/examples/basic/quadratic_program.html
# requirements: pip install cvxpy


# You would have to change the name of the txt file, and possibly
# the size parameter (if not using size 100)
def qknap():
    # http://cedric.cnam.fr/~soutif/QKP/jeu_100_25_1.txt
    # The data reading code is a bit ugly, so just ignore it. It simply reads
    # the linear and quadratic constraints from the txt file
    size = 100
    raw = np.zeros([size, size])
    capacity = 0
    # This follows the notation from Professor Ye's presentation
    q = np.zeros([size, 1])
    a = np.zeros([size, 1])
    row_counter = -2  # For cleaner indexing
    leq = False
    with open("data/soutif_small/jeu_100_25_1.txt", 'r') as f:
        for line in f:  # Read the following if statements in order. They are essentially doing f.readline()
            if row_counter < 0:
                print("Going past initial data...")
            elif row_counter == 0:  # Read the linear term in optimization function
                temp = line.strip('\n').split(' ')
                counter = 0
                for i in temp:
                    if i != '':
                        q[counter][0] = int(i)
                        counter += 1
            elif row_counter > 0 and row_counter < size:  # Quadratic term in opt
                temp = line.strip('\n').split(' ')
                column_counter = row_counter
                for i in temp:
                    if i != '':
                        raw[row_counter - 1][column_counter] = int(i)
                        column_counter += 1
            elif row_counter == size:
                print("pass blank line")
            elif row_counter == size + 1:  # Define type of knapsack capacity equality. Assume always <= capacity
                leq = 1 - int(line.strip('\n'))
            elif row_counter == size + 2:  # Define knapsack capacity
                capacity = int(line.strip('\n'))
            elif row_counter == size + 3:  # Read linear terms in the constraint (knapsack capacity equation)
                temp = line.strip('\n').split(' ')
                counter = 0
                for i in temp:
                    if i != '':
                        a[counter][0] = int(i)
                        counter += 1
            else:
                print("we are done processing data")
            row_counter += 1
    f.close()
    # ---data reading code ends here---
    # Now we relax the QCQP into an SDP formulation
    # Again, following Professor Ye's presentation
    Y = cp.Variable((size + 1, size + 1), PSD=True)
    augmented_Q = cp.bmat([[raw, q], [q.T, np.zeros([1, 1])]])

    A = []  # We have a total of size+1 constraints, for each variable {0,1} and the capacity
    for i in range(size):
        temp = np.zeros([size, size])
        temp[i][i] = 1
        t = np.zeros([size, 1])
        t[i][0] = -1
        A.append(cp.bmat([[temp, t], [t.T, np.zeros([1, 1])]]))  # Q: core assumption, matrices are symmetric
    A.append(cp.bmat([[np.zeros([size, size]), a], [a.T, np.zeros([1, 1])]]))
    b = np.zeros(size + 1)
    b[-1] = capacity

    # Additional constraint for the auxiliary variable
    aux = np.zeros([size + 1, size + 1])
    aux[-1][-1] = 1

    # Sequentially: SDP constraint, aux constraint, 0-1 variable constraint, and capacity constraint
    constraints = [Y >> 0]
    constraints += [cp.trace(aux @ Y) == 1]
    constraints += [
        cp.trace(A[i] @ Y) == b[i] for i in range(size)
    ]
    if leq:
        constraints += [cp.trace(A[-1] @ Y) <= b[-1]]
    else:
        print("undefined knapsack capacity constraint")

    prob = cp.Problem(cp.Maximize(cp.trace(augmented_Q @ Y)), constraints)
    prob.solve(verbose=True)
    # prob.solve(solver=cp.MOSEK, verbose=True)

    print("The optimal value is", prob.value)
    print("A solution Y is")
    print(Y.value)


if __name__ == '__main__':
    qknap()
