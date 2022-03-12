////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2022.                                                                             /
////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Created by C. Zhang on 2022/3/10.
//

#include "bb_copt.h"

Node_COPT::Node_COPT(long id, QP &qp, long parent_id, long depth, double parent_bound, double bound,
                     double primal_val) : Node(), p(qp) {
  this->id = id;
  this->id_parent = parent_id;
  this->depth = depth;
  this->val_rel = bound;
  this->val_prm = primal_val;
  this->val_rel_pa = parent_bound;
  // timers
  this->time_created = time(&time_created);
}

void Node_COPT::extract_solution() {
  p.extract_solution();
  val_rel = p.r.relax;
  val_prm = p.r.primal;
}

int Tree_COPT::iter(Node_COPT &node, Params &param, QP &qp, copt_env *env, long itern) {
  // prunes
  auto node_id = node.id;
  auto current_cuts = map_cuts[node_id];
  auto current_bound = map_bound[node_id];
  int info;
  std::string status;
  auto solve_time = 0.0;
  auto infeas = 0.0;
  if (node.val_rel_pa < param.lb) {
    // bound prune.
    info = 0;
    status = "-";
  } else {
    if (!node.bool_solved) {
      node.create_problem(env, current_cuts, current_bound);
      // warm-start is only needed if it has to be solved;
      // todo: no warm-start yet.
      // if (param.warmstart and node_id) {
      //   auto parent_r = map_result.at(node.id_parent);
      //   auto rws = Result_COPT(qp.n, qp.m, qp.d);
      //   rws.construct_init_point(parent_r, 0.99, current_cuts.size());
      //   node.p.assign_initial_point(rws, true);
      // }
      node.optimize();
      node.extract_solution();
    }
    if (node_id and map_num_unsolved_child[node.id_parent] == 0) {
      map_result.erase(node.id_parent);
    }
    auto node_r = node.get_solution();

    // evaluate
    auto Res = node_r.Res;

    double relax = node.val_rel;
    double primal = node.val_prm;

    infeas = Res.maxCoeff();
    // timers
    solve_time = difftime(node.time_opt_end, timer);

    // update upper bound by sdp relaxation
    if (node.id == 0) {
      param.ub = node_r.relax;
    } else {
      param.ub = node.val_rel_pa;
    }
    param.gap = (param.ub - param.lb) / abs((param.lb) + param.tolfeas);

    if (relax < param.lb) {
      // bound prune.
      info = 0;
      status = "-";
    } else if (param.gap <= param.tolgap) {
      info = 1;
      status = "finished";
    } else if (infeas < param.tolfeas) {
      // primal solution is feasible rank-1 solution
      info = 0;
      status = "P";
    } else {
      info = -1;
      status = "";
    }
    // update lower bound by primal solution
    if (primal > param.lb) {
      param.lb = primal;
      best_node_id = node_id;
      if (!best_result.empty()) {
        best_result.pop();
      }
      best_result.push(node_r);
    }
  }

  // status report
  auto left_nodes = queue.size();
  if ((itern % param.interval_logging == 0) || (info == 1) || (left_nodes == 0)) {
    gen_status_report(
        solve_time, node_id, left_nodes,
        current_cuts.size(), node.p.r.iterations,
        param.lb, param.ub, infeas, param.gap,
        param.lb, param.ub, status
    );
  }
  return info;
}


int Tree_COPT::run(QP &qp, Bound &root_b, copt_env *env, Params &param) {
  Node_COPT root(0, qp);
  double *priority_arr = qp.Dc[0].diagonal().data();
  std::vector<double> priority(priority_arr, priority_arr + qp.Dc[0].size());
  print_header();

  double gap;
  map_cuts[0] = root.p.cp;
  map_bound[0] = Bound(root_b);
  map_ub[0] = param.ub;
  time(&timer);
  queue.insert(std::pair<long, Node_COPT>(0, root));
  long itern = 0;
  while (!queue.empty()) {
    std::pair<long, double> next_kv = fetch_next();
    long node_id = next_kv.first;
    Node_COPT node = queue.at(node_id);
    long parent_id = node.id_parent;
    // all finished
    // if (next_kv.second <= 1e-6) {
    //   break;
    // }
    auto current_bound = map_bound[node_id];
    map_ub.erase(node_id);
    queue.erase(node_id);
    map_num_unsolved_child[node.id_parent] -= 1;
    int info = iter(node, param, qp, env, itern);
#if QCQP_BRANCH_DBG
    std::cout << "subproblem info:" << std::endl;
    std::cout << info << std::endl;
#endif
    if (info == 1) {
      break;
    } else if (info == 0) {
      map_bound.erase(node_id);
      map_cuts.erase(node_id);
    } else {
      auto node_r = node.get_solution();

      // branch
      Branch<Result_COPT> br{};
      br.maximum_priority(node_r, priority, 'z');
      br.imply_bounds(current_bound, qp, 'z');
      // create child
      long child_depth = node.depth + 1;
      long left_id = total_nodes + 1;
      long right_id = total_nodes + 2;

      // create cuts
      // nodes
      auto left_node = Node_COPT(
          left_id, qp, node_id, child_depth,
          node.val_rel, 0.0, 0.0
      );
      auto right_node = Node_COPT(
          right_id, qp, node_id, child_depth,
          node.val_rel, 0.0, 0.0
      );
      // update containers
      map_bound[left_id] = br.b_left;
      map_cuts[left_id] = node.p.cp;
      map_ub[left_id] = node.val_rel;
      map_num_cuts[left_id] = node.p.cp.size();
      queue.insert(std::pair<long, Node_COPT>(left_id, left_node));
      map_bound[right_id] = br.b_right;
      map_cuts[right_id] = node.p.cp;
      map_ub[right_id] = node.val_rel;
      map_num_cuts[right_id] = node.p.cp.size();
      queue.insert(std::pair<long, Node_COPT>(right_id, right_node));
      // add to result mapping temporarily
      map_result.insert(std::pair<long, Result_COPT>(node_id, node_r));

      // counting
      total_nodes += 2;
      map_num_unsolved_child[node_id] = 2;
      // clean up
      map_bound.erase(node_id);
      map_cuts.erase(node_id);
    }
    itern += 1;
  }

  return 1;
}