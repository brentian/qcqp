
cmake_minimum_required(VERSION 3.15)

add_custom_target(
        purepython
        COMMAND cp -r "${PROJECT_SOURCE_DIR}/pyqp" ${CMAKE_BINARY_DIR}/
)