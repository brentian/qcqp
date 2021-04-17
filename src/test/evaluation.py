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
try:
    import gurobipy as grb
except:
    print("no gurobipy found!")

try:
    import cvxpy as cvx
except:
    print("no cvxpy found!")


class Eval(object):
    def __init__(self, prob_num, solve_time, best_bound, best_obj, best_reform_obj=0.0):
        self.prob_num = prob_num
        self.solve_time = round(solve_time, 2)
        self.best_bound = best_bound if best_bound == "-" else round(
            best_bound, 2)
        self.best_obj = round(best_obj, 2)
        self.best_reform_obj = round(best_reform_obj, 2)


def evaluate(prob_num, model, *variables):
    try:
        import gurobipy as grb
    except:
        print("no gurobipy found!")
    try:
        import cvxpy as cvx
    except:
        print("no cvxpy found!")
    best_reform_obj = 0.0
    if isinstance(model, grb.Model):
        solve_time = model.Runtime
        best_bound = model.ObjBoundC
        best_obj = model.ObjVal
    elif isinstance(model, cvx.Problem):
        stats = model.solver_stats
        solve_time = stats.solve_time
        best_bound = "-"  # todo, add this
        best_reform_obj = model.value
        best_obj = model.true_obj

    else:
        raise ValueError("not implemented")
    return Eval(prob_num, solve_time, best_bound, best_obj, best_reform_obj)
