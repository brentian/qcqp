//
// Created by chuwen on 2021/9/9.
//

#ifndef QCQP_TREE_H
#define QCQP_TREE_H


#include <utility>

#include "cut.h"
#include "branch.h"
#include "result.h"
#include "qp.h"
#include "utils.h"


//template<typename T> class Node {
//public:
//
//    Node(long id, QP &qp, // no dfts.
//         long parent_id = -1, long depth = 0,
//         double bound = 0.0, double primal_val = 0.0,
//         double abs_time = 0.0, double create_time = 0.0, double solve_time = 0.0
//    );
//
//    T p;
//    long id;
//    long parent_id;
//    long depth;
//    double obj_bound;
//    double primal_val;
//    double abs_time;
//    double create_time;
//    double solve_time;
//};


class Node {
public:
    Node() = default;
//    Node(long id, QP &qp, // no dfts.
//         long parent_id = -1, long depth = 0,
//         double bound = 0.0, double primal_val = 0.0,
//         double abs_time = 0.0, double create_time = 0.0, double solve_time = 0.0
//    );

//    Backend p;
    long id;
    long id_parent;
    long depth;
    double val_rel_pa;
    double val_rel;
    double val_prm;
    time_t time_created;
    time_t time_opt_start;
    time_t time_opt_end;
    double time_solve;
    // Cut and variable bound
    //    CutPool ct;
    //    Bound bd;
    //    void attach_cuts(CutPool another_cp);
    //    void attach_bound(Bound &another);
    //    CutPool get_cuts();
};

template<typename T>
std::string to_string_with_precision(const T a_value, const int n = 3, const bool bool_scientific = false) {
    std::ostringstream out;
    out.precision(n);
    if (bool_scientific) {
        out << std::scientific << a_value;
        return out.str();
    }
    out << std::fixed << a_value;
    return out.str();
}

template<typename NodeType, typename ResultType>
class Tree {
public:
    std::map<long, NodeType> queue;
    std::stack<ResultType> best_result;
    std::map<long, Bound> map_bound; // variable box bound
    std::map<long, CutPool> map_cuts; // cut for each node
    std::map<long, double> map_ub;  // upper bound of the node (by parent value)
    std::map<long, ResultType> map_result; // result (only a few)
    std::map<long, int> map_num_unsolved_child; // number of unsolved child nodes.
    std::map<long, long> map_num_cuts; // number of cuts for parent.

    long best_node_id = 0;
    long total_nodes = 0;
    const std::string LOG_HEADER = "The Quadratic Constrained Quadratic Programming Solver\n";
    const std::string LOG_AUTHOR_INFO = "(c) Chuwen Zhang, 2021-2022 \n";
    // const std::string LOG_AUTHOR_INFO = "(c) Chuwen Zhang, Yinyu Ye, 2021-2022 \n";
    const std::vector<std::string> LOG_HEADER_ARR = {
            "time", "#/depth", "cuts",
            "ipm", "inf",
            "prm", "rel", "gap",
            "status"
    };
    const int LENGHTH_SLOT = 12;

    void print_header() {

        std::cout << LOG_BREAKER << std::endl;
        std::cout << std::string((LOG_BREAKER.size() - LOG_HEADER.size()) / 2, ' ') << LOG_HEADER;
        std::cout << std::string((LOG_BREAKER.size() - LOG_AUTHOR_INFO.size()) / 2, ' ') << LOG_AUTHOR_INFO;
        std::cout << LOG_BREAKER << std::endl;
        std::cout << ITER_HEADER;
    };

    std::string gen_header_slots() {
        auto ss = std::stringstream();
        for (auto ele: LOG_HEADER_ARR) {
            auto empty_size = LENGHTH_SLOT - ele.size();
            auto pre_size = empty_size / 2;
            auto aff_size = empty_size - pre_size;
            ss << "|" << std::string(pre_size, ' ') << ele << std::string(aff_size, ' ');
        }
        ss << "|" << std::endl;
        return ss.str();
    }

    const std::string ITER_HEADER = gen_header_slots();
    const std::string LOG_BREAKER = std::string(ITER_HEADER.size(), '#');

    void gen_status_report(
            double solve_time, long id_node, long depth_node,
            long cut_size, long iter_ipm,
            double prm, double relax, double inf, double gap,
            double lb, double ub, std::string status
    ) {
        auto _line_arr = {
                to_string_with_precision<double>(solve_time, 2),
                std::to_string(id_node),
                std::to_string(cut_size),
                std::to_string(iter_ipm),
                to_string_with_precision<double>(inf, 3, true),
                to_string_with_precision<double>(prm, 3, true),
                to_string_with_precision<double>(relax, 3, true),
                to_string_with_precision<double>(gap, 2, true),
                std::move(status)
        };
        auto ss = std::stringstream();
        for (auto ele: _line_arr) {
            auto empty_size = LENGHTH_SLOT - ele.size();
            auto pre_size = empty_size / 2;
            auto aff_size = empty_size - pre_size;
            ss << "|" << std::string(pre_size, ' ') << ele << std::string(aff_size, ' ');
        }
        ss << "|" << std::endl;
        std::cout << ss.str();
    }

};


#endif //QCQP_TREE_H

