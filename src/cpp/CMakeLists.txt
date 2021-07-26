cmake_minimum_required(VERSION 3.15)

# globals
# add modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# eigen
find_package(Eigen3 REQUIRED)

# sdpa
find_package(SDPA)
find_package(LAPACK)
find_package(BLAS)

include_directories("./")
include_directories("./include")
include_directories(${SDPA_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
# extra includes

# subdir
add_subdirectory(test)


# add libs
add_library(qp utils.h utils.cpp qp.h qp.cpp bg_sdpa.h bg_sdpa.cpp)
target_link_libraries(qp PRIVATE ${SDPA_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} nlohmann_json::nlohmann_json)

