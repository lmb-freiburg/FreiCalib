cmake_minimum_required(VERSION 2.6)
set (CMAKE_CXX_STANDARD 11)
project(ceres_librarypnp)
include_directories("include/")

find_package(Ceres REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})

add_executable(ceres_librarypnp main.cpp)
target_link_libraries(ceres_librarypnp ${CERES_LIBRARIES})

install(TARGETS ceres_librarypnp RUNTIME DESTINATION bin)
