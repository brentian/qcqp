---
bibliography: ['./qcqp.bib']
titlepage: true
title: QCQP Project Plan
mainfont: Latin Modern Sans
author: Chuwen Zhang
date: \today
---


# Intro

## Scope of our work
Develop a QCQP solver that uses SDP-relaxation-and-refinement approach. The QCQP solver should be problem-independent that works for any QCQP instance.

### Modeling interface
The modeling interface is domain specific language, a simple tool for user to define a **QCQP** problem, then the solver translates into canonical form of QCQP. For example, for a HQCQP, canonical form includes parameters $Q$, $A_i, b_i, \forall i$

To-dos:

- The modeling part does the canonicalization, works like cvxpy, yalmip, etc.
- Directly use COPT.

See @agrawal_rewriting_2018, @diamond_cvxpy_2016, @dunning_jump_2017, @lofberg_yalmip_2004


### SDP interface
The SDP interface should be solver **independent**. SDP interface starts with canonical form to create a SDP-relaxation. So the users do not have to derive SDP by themselves. The interface should output a SDP problem in a standard format, e.g., SDPA format, that can be accepted by any SDP solver.

To-dos including:

- starts with canonical form.
- interface with solver: create problems, extract solutions, status, etc.

### Local Refinement
Local Refinement from SDP solution to QP seems to be problem dependent, whereas we can start with:

- Use Gurobi to do the refinement
- use existing methods, including residual minimization (SNL), randomization (for BQP), see @luo_semidefinite_2010 and papers for SNL.
- add an **option** for user to choose a refinement method.


## Computational tests

Test on SNL, kissing problem, etc.

## Development plan

- start with Pure Python or Julia interface as a fast prototype. 
  - in Python one can use cvxpy or other AMLs; in Julia one may use JuMP. 
  - computational tests on kissing problem, SNL using different SDP solver can be handled at the same time
- add and move to C/C++ interface. does same thing as Python, then this backend with replace the pure Python one.
- Add Python, Matlab support for C interface.


## Details
We describe some of the details here.

### SDP Relaxations
We consider two types of SDP relaxations:

#### Method I:


# Reference