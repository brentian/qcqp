
# add modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# gurobi
find_package(GUROBI REQUIRED)
include_directories(${GUROBI_INCLUDE_DIRS})

add_library(mip mip.cpp mosek_mip.cpp)
add_executable(mip_grb_single gurobi_single_mip.cpp)

target_link_libraries(mip_grb_single ${GUROBI_CXX_LIBRARY})
target_link_libraries(mip_grb_single ${GUROBI_LIBRARY})
