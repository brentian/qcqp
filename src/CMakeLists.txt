
cmake_minimum_required(VERSION 3.15)


# set the project name
project(qcqp VERSION 0.0.1 LANGUAGES C CXX)


# specify the C/C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# paths
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_PREFIX "~/.local/")


# add subdirectories
add_subdirectory(pyqp)
add_subdirectory(pyqptest)
add_subdirectory(thirdparty)
add_subdirectory(cpp)
