//
// Created by C. Zhang on 2021/9/5.
//

#include "io.h"

json parse_json(char *fp) {
    using namespace std;
    ifstream ifs(fp);
    json _json = json::parse(ifs);
    return _json;
}

json parse_json(const std::string &fp) {
    using namespace std;
    ifstream ifs(fp);
    json _json = json::parse(ifs);
    return _json;
}

void get_arr(json &js, std::string key, double *data) {
    auto vec = js[key].get<std::vector<double>>();
    std::copy(vec.begin(), vec.end(), data);
}
