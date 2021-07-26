
message("Looking for sources in ${PROJECT_SOURCE_DIR}")

# add the executable
add_executable(utiltest test_utils.cpp)
target_link_libraries(utiltest PRIVATE qp nlohmann_json::nlohmann_json)

add_executable(qptest test_qp.cpp)
target_link_libraries(qptest PRIVATE qp nlohmann_json::nlohmann_json)

add_executable(sdpaqptest test_bg_sdpa.cpp)
target_link_libraries(sdpaqptest PRIVATE qp nlohmann_json::nlohmann_json)