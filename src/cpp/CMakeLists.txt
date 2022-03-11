cmake_minimum_required(VERSION 3.15)

# globals
# add modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# remote third-party
include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.tar.xz)
FetchContent_MakeAvailable(json)

# eigen
# @using latest ver in 3rd party
find_package(Eigen3 REQUIRED)

# sdpa
find_package(SDPA)
find_package(LAPACK)
find_package(BLAS)
find_package(DSDP)
find_package(COPT)

include_directories("./")
include_directories("./include")
include_directories(${SDPA_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${DSDP_INCLUDE_DIR})
include_directories(${COPT_INCLUDE_DIR})


# extra includes

# subdir
add_subdirectory(test)
add_subdirectory(bbtest)

# add libs
add_library(
        qp
        qp.h qp.cpp
        cut.h cut.cpp
        branch.h branch.cpp
        result.h result.cpp
        utils.h utils.cpp
        tree.cpp tree.h
)
target_link_libraries(qp PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_library(qpio io.h io.cpp)
target_link_libraries(qpio PUBLIC nlohmann_json::nlohmann_json)

add_library(qpsdpa bg_sdpa.h bg_sdpa.cpp)
target_link_libraries(qpsdpa PRIVATE qp qpio ${SDPA_LIBRARIES})

add_library(qpdsdp bg_dsdp.h bg_dsdp.cpp bg_dsdp_cut.h bg_dsdp_cut.cpp bb_dsdp.h bb_dsdp.cpp)
target_link_libraries(qpdsdp PRIVATE qp qpio ${DSDP_LIBRARIES})

add_library(qpcopt bg_copt.cpp bg_copt.h bb_copt.cpp bb_copt.h)
target_link_libraries(qpcopt PRIVATE qp qpio ${COPT_LIBRARIES})

