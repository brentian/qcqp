// branch and bound for QCQP
//  using SDP (DSDP) as backend.

#ifndef QCQP_BB_DSDP_H
#define QCQP_BB_DSDP_H

#include "qp.h"
#include "utils.h"
#include "dsdp5.h"
#include "cut.h"
#include "tree.h"
#include "bg_dsdp.h"
#include "bg_dsdp_cut.h"

class Node_DSDP : public Node {
public:
    QP_DSDP p;
    bool bool_solved = false;
    bool bool_setup = false;

    Node_DSDP(
        long id,
        QP &qp, // no dfts.
        long parent_id = -1,
        long depth = 0,
        double parent_bound = 1e6,
        double bound = 0.0,
        double primal_val = 0.0
    );


    void create_problem(CutPool &cp);

    void optimize();

    void extract_solution();

    Result_DSDP get_solution() const {
      return p.get_solution();
    }

};


class Tree_DSDP : public Tree<Node_DSDP, Result_DSDP> {
public:
    time_t timer{};

    int run(QP &qp, Params &param);

    int iter(Node_DSDP &node, Params &param, QP &qp);

    std::pair<long, double> fetch_next() {
      auto kv = get_max(map_ub);
      return kv;
    }

    template<typename KeyType, typename ValueType>
    std::pair<KeyType, ValueType> get_max(const std::map<KeyType, ValueType> &x) {
      using pairtype = std::pair<KeyType, ValueType>;
      return *std::max_element(x.begin(), x.end(), [](const pairtype &p1, const pairtype &p2) {
          return p1.second < p2.second;
      });
    }
};

class Branch_DSDP : public Branch<Result_DSDP> {
};


#endif //QCQP_BB_DSDP_H
