
message("Looking for sources in ${PROJECT_SOURCE_DIR}")

# add the executable
add_executable(test_util test_utils.cpp)
target_link_libraries(test_util PRIVATE qp nlohmann_json::nlohmann_json)

add_executable(test_qp test_qp.cpp)
target_link_libraries(test_qp PRIVATE qp nlohmann_json::nlohmann_json)

add_executable(test_sdpaqp test_bg_sdpa.cpp)
target_link_libraries(test_sdpaqp PRIVATE qp nlohmann_json::nlohmann_json)

add_executable(test_dsdp test_dsdp_stable.c)
target_link_libraries(test_dsdp PRIVATE ${DSDP_LIBRARY})