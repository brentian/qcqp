
cmake_minimum_required(VERSION 3.15)

add_subdirectory(mip)

list(APPEND EXTRA_LIBS mip)
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/mip")

# add the executable
add_executable(null null.cpp)
target_link_libraries(null PUBLIC ${EXTRA_LIBS})

