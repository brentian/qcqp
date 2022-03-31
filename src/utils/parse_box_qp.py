"""
The script to parse S. Burer's QP instance.
 see, https://github.com/sburer/BoxQP_instances
Problem is

  max  0.5*x'*Q*x + c'*x
  s.t. 0 <= x <= e

File format is

  n
  c'
  Q
"""
import json
import sys
import os
import numpy as np


def parseline(l):
  return list(map(lambda x: float(x), l.split(" ")[:-1]))


def read_instance(name, fpath):
  content = {
    'd': 1,
    'm': 0,
    "A": [],
    "a": [],
    "b": []
  }
  Q = []
  with open(fpath, 'r') as f:
    linect = 0
    for l in f:
      linect += 1
      if linect == 1:
        content['n'] = int(l[:-1])
        continue
      if linect == 2:
        content['q'] = parseline(l)
        continue
      Q.append(parseline(l))
  
  Qa = np.array(Q) * 0.5
  Q = Qa.flatten().tolist()
  content['Q'] = Q
  content['name'] = name
  return content


if __name__ == '__main__':
  
  input_dir, output_dir = sys.argv[1:]
  
  if os.path.isdir(input_dir):
    in_files = os.listdir(input_dir)
    
    print(f"running {len(in_files)} models")
    
    for i in in_files:
      content = read_instance(i, f"{input_dir}/{i}")
      with open(f"{output_dir}/{i}.json", 'w') as fo:
        json.dump(content, fo)
# "Basic" instances come from
#
#     @Article{VanNem05b,
#       author        = {D. Vandenbussche and G. Nemhauser},
#       title         = {A branch-and-cut algorithm for nonconvex quadratic
#                       programs with box constraints},
#       journal       = {Mathematical Programming},
#       volume        = {102},
#       number        = {3},
#       year          = {2005},
#       pages         = {559--575}
#     }
#
# "Extended" instances come from
#
#     @Article{BurVan09,
#       author        = {Samuel Burer and Dieter Vandenbussche},
#       title         = {Globally Solving Box-Constrained Nonconvex Quadratic
#                       Programs with Semidefinite-Based Finite Branch-and-Bound},
#       journal       = {Comput. Optim. Appl.},
#       fjournal      = {Computational Optimization and Applications. An
#                       International Journal},
#       volume        = {43},
#       year          = {2009},
#       number        = {2},
#       pages         = {181--195}
#     }
#
# "Extended2" instances come from
#
#     @Article{Burer10,
#       author  = {Samuel Burer},
#       title   = {Optimizing a Polyhedral-Semidefinite Relaxation of
#                 Completely Positive Programs},
#       journal = {Mathematical Programming Computation},
#       year    = {2010},
#       volume  = {2},
#       pages   = {1-19},
#       number  = {1}
#     }
