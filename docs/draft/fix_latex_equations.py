# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: notes
# @file: /fix_latex_equations.py
# @created: Wednesday, 29th July 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Wednesday, 29th July 2020 9:38:39 pm
# @description:
#   fix latex issues on \[\begin{...} \end{...}\]
#   remove redundant \[ \] chars

import re
import sys
import os


def parse_line_redundant_rec(line):
  l1 = re.sub(r"(\\\[)(\s){0,}(\\begin)", r'\3', line)
  return re.sub(r"(\\end\{\S{0,}\})(\s){0,}(\\\])", r'\1', l1)

def parse_line(line):
  l1 = re.sub(r"\\boldsymbol", r'\\mathbfit', line)
  return l1


def main(fname):

  def process():
    with open(fname) as f:
      yield parse_line(f.read())

  f_out_name = f"{fname}.tmp"
  with open(f_out_name, 'w') as f_out:
    for l in process():
      f_out.write(l)

  os.rename(fname, f"{fname}.raw")
  os.rename(f_out_name, fname)


if __name__ == "__main__":
  fname = sys.argv[1]
  main(fname)
