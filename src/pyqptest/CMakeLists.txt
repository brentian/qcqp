
cmake_minimum_required(VERSION 3.15)

add_custom_target(
        pyqptest
        DEPENDS purepython
        COMMAND cp -r "${PROJECT_SOURCE_DIR}/pyqptest" ${CMAKE_BINARY_DIR}/
)