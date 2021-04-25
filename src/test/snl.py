# An attempt that started off as integer programming
# But I noticed that solutions are not integers...

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
# https://www.cvxpy.org/examples/basic/quadratic_program.html
# requirements: pip install cvxpy


# You would have to change the name of the txt file, and possibly
# the size parameter (if not using size 100)
def qkp(path):
    # http://cedric.cnam.fr/~soutif/QKP/jeu_100_25_1.txt
    # The data reading code is a bit ugly, so just ignore it. It simply reads
    # the linear and quadratic constraints from the txt file
    # -----------------------------------IGNORE---------------------------------------
    # -----------------------------------IGNORE---------------------------------------
    # -----------------------------------IGNORE---------------------------------------
    size = 100
    raw = np.zeros([size, size])
    capacity = 0
    # This follows the notation from Professor Ye's presentation
    q = np.zeros([size, 1])
    a = np.zeros([size, 1])
    row_counter = -2  # For cleaner indexing
    leq = False
    with open(path, 'r') as f:
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
                        raw[row_counter - 1][column_counter] = int(i)/2
                        raw[column_counter][row_counter - 1] = int(i) / 2
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
    # ---data reading code ends here---
    # ---data reading code ends here---
    # ---data reading code ends here---
    # ---data reading code ends here---



    # Now we relax the QCQP into an SDP formulation
    # Again, following Professor Ye's presentation
    # Key question: I thought I had 0-1 constraints, but the answer didn't?
    print(raw)
    Y = cp.Variable((size + 1, size + 1), PSD=True)
    # test = cp.Variable()
    augmented_Q = cp.bmat([[raw, q/2], [q.T/2, np.zeros([1, 1])]])

    # The A matrix and b vector is designed to fit the trace(A @ Y) == b constraint
    # Specifically, the last element of b is the capacity, and all others are zero
    # This is because I want these constraints to read xi^2 - xi = 0
    # By assuming the auxiliary variable is always +-1 (its value squared is always 1)
    # trace(A @ Y) = xi^2 - 0.5*xi - 0.5*xi = xi^2 - xi
    A = []  # We have a total of size+1 constraints, for each variable in {0,1} and the capacity constraint
    for i in range(size):
        temp = np.zeros([size, size])
        temp[i][i] = 1
        t = np.zeros([size, 1])
        t[i][0] = -1
        A.append(cp.bmat([[temp, t/2], [t.T/2, np.zeros([1, 1])]]))
    A.append(cp.bmat([[np.zeros([size, size]), a], [a.T, np.zeros([1, 1])]]))
    b = np.zeros(size + 1)
    b[-1] = capacity

    # Additional constraint for the auxiliary variable
    aux = np.zeros([size + 1, size + 1])
    aux[-1][-1] = 1

    # Sequentially: SDP constraint, aux constraint, 0-1 variable constraint, and capacity constraint
    constraints = [Y >> 0]
    constraints += [cp.trace(aux @ Y) == 1]
    constraints += [ # This should enforce xi^2 - xi = 0
        cp.trace(A[i] @ Y) <= b[i] for i in range(size)
    ]
    # constraints += [cp.trace(A[-1] @ Y) <= test]

    if leq:
        constraints += [cp.trace(A[-1] @ Y) <= b[-1]]
    else:
        print("undefined knapsack capacity constraint")

    prob = cp.Problem(cp.Maximize(cp.trace(augmented_Q @ Y)), constraints)
    # prob.solve(verbose=True)
    prob.solve(solver=cp.MOSEK, verbose=True)

    # Key question: Why isn't Y an integer matrix?????
    print(Y.value)
    return prob, Y

# TODO: Don't use random numbers, but the sample problem on https://web.stanford.edu/~yyye/Col.html
# TODO: Add sensor-sensor constraints. Currently there is only anchor-sensor constraints
# TODO: Currently this is the SDP relaxed version. Convert data into Gurobi input format
# Reference original paper: https://web.stanford.edu/~yyye/adhocn4.pdf
def snl():

    # Currently just producing random SNL problems
    dim = 2 #How many dimensions we are in (dim = 2 means Euclidean)
    pts = 50 #How many sensors/variables we are solving for
    anc = 10 #How many anchors we have
    # generate random anchor and ground truth point positions
    anchors = np.random.random((anc, dim)) * 100
    gt = np.random.random((pts, dim)) * 100
    # Not considering sensor-sensor distance for now, trivial to add
    ax_distance = list()
    for row in anchors:
        # Question: Apparently there is a huge difference in result between
        # using and not using the np.square(). Why?
        ax_distance.append(np.square(np.linalg.norm(row - gt, axis=1)))
    ax_distance = np.array(ax_distance)
    print(ax_distance.shape)


    Z = cp.Variable((dim+pts, dim+pts), PSD=True)
    slack_one = cp.Variable((anc, pts))
    slack_two = cp.Variable((anc, pts))

    constraints = [Z >> 0]
    constraints += [slack_one >= 0]
    constraints += [slack_two >= 0]

    # I think the three constraints above are easy to understand
    # We also impose the structure of Z, and the distance constraint below
    # This constraint is to make the first dim x dim block of Z identity matrix

    # Question: In the original paper there seems to be a sum constraint, i.e.
    # diagonals of the identity add up to #dim. Is that really necessary?
    # Not including that here, but could add as a performance test
    for i in range(dim):
        temp = np.zeros((dim+pts, 1))
        temp[i] = 1
        constraints += [cp.trace(temp.T @ Z @ temp) == 1]
    temp = np.zeros((dim+pts, 1))
    for i in range(dim):
        temp[i] = 1
    constraints += [cp.sum(cp.trace(temp.T @ Z @ temp)) == dim]

    # This constraint is for anchor-sensor distances. Again, we could easily include
    # sensor-sensor distances here as well
    for i in range(anc):
        for j in range(pts):
            temp = np.zeros((pts,))
            temp[j] = -1
            t = np.concatenate([anchors[i], temp]).reshape(-1, 1)
            constraints += [cp.trace(t.T @ Z @ t) + slack_one[i][j] - slack_two[i][j] == ax_distance[i][j]]

    # Our optimization is simply summing over the entire matrix
    prob = cp.Problem(cp.Minimize(cp.sum(slack_one) + cp.sum(slack_two)), constraints)
    prob.solve(solver=cp.MOSEK, verbose=True)

    print(prob.value)
    # return prob, Z

    # Let's do some plotting
    gt_points = gt
    plt.scatter(gt_points[:, 0], gt_points[:, 1])
    op_points = np.array(Z.value[0:dim, dim:]).reshape(-1, dim)
    plt.scatter(op_points[:, 0], op_points[:, 1])
    plt.show()


if __name__ == '__main__':
    # qkp("../sbin/soutif_/jeu_100_25_1.txt")
    snl()