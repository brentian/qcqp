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

def qcqp(Q, q, A, a, b, sign, backend="cvx", solver = None):
    """qcqp solver parameterized as follows:
        \max x^TQx + <q, x>
        s.t.
        x^T(A_i)x + <a_i, x> [\le, \eq, \ge] b_i, \forall i
    in homogeneous case, q = 0; a_i = 0 \forall i

    Parameters
    ----------
    Q : np.array, of shape (n, n)
    q : np.array, of shape (n)
    A : np.array, of shape (m, n, n)
    a : np.array, of shape (m, n)
    b : np.array, of shape (m)
    sign : np.array, of shape (m) defining the sign in {\le, \ge, \eq}

    As defined in the above formulation:
        Q, q correspond to objective
        A[j], a[j], b[j], sign[j] correspond to j'th constraint.

    if q is nontrivial, the instance is inhomogeneous.
    """
    pass
