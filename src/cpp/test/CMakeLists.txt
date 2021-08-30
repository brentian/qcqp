
message("Looking for sources in ${PROJECT_SOURCE_DIR}")

# add the executable
add_executable(test_util test_utils.cpp)
target_link_libraries(test_util PRIVATE qp nlohmann_json::nlohmann_json)

add_executable(test_qp test_qp.cpp)
target_link_libraries(test_qp PRIVATE qp nlohmann_json::nlohmann_json)

# examples from solver library
add_executable(example_dsdp_stable example_dsdp_stable.c)
target_link_libraries(example_dsdp_stable PRIVATE ${DSDP_LIBRARIES})

add_executable(example_dsdp_maxcut example_dsdp_maxcut.c)
target_link_libraries(example_dsdp_maxcut PRIVATE ${DSDP_LIBRARIES})


# my examples
add_executable(test_sdpaqp test_bg_sdpa.cpp)
target_link_libraries(test_sdpaqp PRIVATE qp nlohmann_json::nlohmann_json)


add_executable(test_dsdp test_bg_dsdp.cpp)
target_link_libraries(test_dsdp PRIVATE qp ${DSDP_LIBRARIES} nlohmann_json::nlohmann_json)