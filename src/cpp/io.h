//
// Created by C. Zhang on 2021/9/5.
//

#ifndef QCQP_IO_H
#define QCQP_IO_H


#include <fstream>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

json parse_json(char *fp);

json parse_json(const std::string &fp);

void *get_arr(json &js, std::string key, double *data);

#endif //QCQP_IO_H
