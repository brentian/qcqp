cmake_minimum_required(VERSION 3.15)

# globals
# add modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# eigen
# @using latest ver in 3rd party
find_package(Eigen3 REQUIRED)

# sdpa
find_package(SDPA)
find_package(LAPACK)
find_package(BLAS)
find_package(DSDP)

include_directories("./")
include_directories("./include")
include_directories(${SDPA_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${DSDP_INCLUDE_DIR})

# extra includes

# subdir
add_subdirectory(test)
add_subdirectory(bbtest)

# add libs
add_library(
        qp qcqp.h qcqp.cpp
        qp.h qp.cpp
        cut.h cut.cpp
        branch.h branch.cpp
        result.h result.cpp
        utils.h utils.cpp
        bg_dsdp.h bg_dsdp.cpp bg_dsdp_cut.h bg_dsdp_cut.cpp
        bg_sdpa.h bg_sdpa.cpp
        tree.cpp tree.h
        bb_dsdp.cpp bb_dsdp.h)

target_link_libraries(qp PRIVATE ${SDPA_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_library(qpio io.h io.cpp)
target_link_libraries(qpio PRIVATE nlohmann_json::nlohmann_json)
