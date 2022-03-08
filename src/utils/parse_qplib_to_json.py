import argparse
import numpy as np
import json

parser = argparse.ArgumentParser("qplib to json translator")
parser.add_argument("--i", type=str, help='input .qplib file')
parser.add_argument("--o", type=str, help='output .json directory')

DEFAULT_SPLIT = " "
HASH_SENTENCE_MAPPING = {
  "number of variables": "n",  # qp.n
  "number of constraints": "m",  # qp.m
  "number of quadratic terms in objective": "nQ",  # =>  qp.Q
  "default value for linear coefficients in objective": "q",  # =>  qp.q
  "number of non-default linear coefficients in objective": "nq",  # =>  qp.q
  "objective constant": "conc",
  "number of quadratic terms in all constraints": "nA",  # =>  qp.A
  "number of linear terms in all constraints": "na",  # =>  qp.a
  "default left-hand-side value": "al",
  "number of non-default left-hand-sides": "nal",
  "default right-hand-side value": "au",
  "number of non-default right-hand-sides": "nau",
  "default variable lower bound value": "vl",
  "number of non-default variable lower bounds": "nvl",
  "default variable upper bound value": "vu",
  "number of non-default variable upper bounds": "nvu",
}
CONTEXT_ORDER = [
  "number of variables", "number of constraints",
  "number of quadratic terms in objective",
  "default value for linear coefficients in objective",
  "number of non-default linear coefficients in objective",
  "objective constant", "number of quadratic terms in all constraints",
  "number of linear terms in all constraints", "default left-hand-side value",
  "number of non-default left-hand-sides", "default right-hand-side value",
  "number of non-default right-hand-sides",
  "default variable lower bound value",
  "number of non-default variable lower bounds",
  "default variable upper bound value",
  "number of non-default variable upper bounds"
]
CONTEXT_MAPPING = {
  "nQ": "Q",
  "nq": "q",
  "nA": "A",
  "na": "a",
  "nal": "al",
  "nau": "au",
  "nvl": "vl",
  "nvu": "vu"
}
SHAPE_MAPPINGS = {
  "nQ": ("n", "n"),
  "nq": ("n"),
  "nA": ("m", "n", "n"),
  "na": ("m", "n"),
  "nal": ("m"),
  "nau": ("m"),
  "nvl": ("n"),
  "nvu": ("n")
}
CONTEXT_MULTIPLIER = {
  "Q": 0.5,
  "A": 0.5
}


def parse_case(fp=None):
  lines = open(fp, 'r').readlines()
  # PREPROCESS
  name = lines[0][:-1]
  ptype = lines[1][:-1]
  sense = lines[2][:-1]
  # DETECT HASH BLOCKS
  hash_pivots = []
  hash_type = []
  hash_lines = {
    l[:-1].split("#")[-1][1:]: idx for idx, l in enumerate(lines) if '#' in l
  }
  for k in CONTEXT_ORDER:
    if k in hash_lines:
      hash_type.append(HASH_SENTENCE_MAPPING[k])
      hash_pivots.append(hash_lines[k])
  
  output = {"name": name, "ptype": ptype, "sense": sense}
  for idx, t in enumerate(hash_type):
    if t in {"n", "m"}:
      lc = hash_pivots[idx]
      value = int(lines[lc].split(DEFAULT_SPLIT)[0])
      output[t] = value
      continue
    if t in {"al", "au", "vu", "vl", "q"}:
      # the default values
      lc = hash_pivots[idx]
      value = float(lines[lc].split(DEFAULT_SPLIT)[0])
      output[t] = max(-1e8, min(value, 1e8))
      continue
    elif t in {"nal", "nau", "nvl", "nvu", "na", "nA", "nq", "nQ"}:
      lc = hash_pivots[idx]
      value = int(lines[lc].split(DEFAULT_SPLIT)[0])
      output[t] = value
      # change the context
      block = lines[lc + 1:lc + value + 1]
      context = CONTEXT_MAPPING[t]
      # multiplier, for qplib, the quad forms is 1/2
      shape = tuple(output[k] for k in SHAPE_MAPPINGS[t])
      array = np.ones(shape) * output.get(context, 0)  # query default values
      for l in block:
        *index, v = l[:-1].split(DEFAULT_SPLIT)
        index = tuple(int(i) - 1 for i in index)
        array[index] = float(v) * CONTEXT_MULTIPLIER.get(context, 1)
      output[context] = array.flatten().tolist()
    
    else:
      print(f"no need to proceed at {t}")
  
  
  # axis 1 for the problem
  output["d"] = 1
  return output


if __name__ == '__main__':
  args = parser.parse_args()
  instance = parse_case(fp=args.i)
  json.dump(instance, open(args.o, 'w'))
