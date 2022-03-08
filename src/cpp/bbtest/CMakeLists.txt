
message("Looking for sources in ${PROJECT_SOURCE_DIR}")

add_executable(test_dsdp_bb test_bg_dsdp.cpp)
target_link_libraries(test_dsdp_bb PRIVATE qp qpio qpdsdp)