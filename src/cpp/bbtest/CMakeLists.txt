
message("Looking for sources in ${PROJECT_SOURCE_DIR}")

add_executable(test_dsdp_bb test_bg_dsdp.cpp)
target_link_libraries(test_dsdp_bb PUBLIC qp qpio qpdsdp)

add_executable(test_copt_bb test_bg_copt.cpp)
target_link_libraries(test_copt_bb PUBLIC qp qpio qpcopt)