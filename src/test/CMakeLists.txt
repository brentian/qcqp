
cmake_minimum_required(VERSION 3.15)

add_custom_target(
        test
        DEPENDS purepython
        COMMAND cp -r "${PROJECT_SOURCE_DIR}/test" ${CMAKE_BINARY_DIR}/
)