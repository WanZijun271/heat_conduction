cmake_minimum_required(VERSION 3.20)
project(MyProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fdiagnostics-color=always -Wall -H)

set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message(STATUS "Find Eigen3")
endif()

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/main.cpp"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(app ${SRC_FILES})

target_link_libraries(app Eigen3::Eigen)

target_precompile_headers(app PRIVATE "${CMAKE_SOURCE_DIR}/include/pch.h")