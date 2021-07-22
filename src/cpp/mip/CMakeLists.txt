
# add modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories("include")


# gurobi
find_package(GUROBI REQUIRED)
include_directories(${GUROBI_INCLUDE_DIRS})

# sdpa
find_package(SDPA)
find_package(LAPACK)
find_package(BLAS)

include_directories(${SDPA_INCLUDE_DIR})

# lib/bin
add_library(mip mip.cpp mosek_mip.cpp)
add_executable(mip_grb_single gurobi_single_mip.cpp)

target_link_libraries(mip_grb_single ${GUROBI_CXX_LIBRARY})
target_link_libraries(mip_grb_single ${GUROBI_LIBRARY})
add_executable(sdpa_try sdpa_try.cpp)
target_link_libraries(sdpa_try ${SDPA_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})