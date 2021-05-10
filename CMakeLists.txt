cmake_minimum_required(VERSION 3.8.2)
project(MasterTheGameOfConnect4)

set(CMAKE_CXX_STANDARD 14)

# Set the output folder where your program will be created
# Reference: https://stackoverflow.com/questions/52022717/how-to-make-cmake-output-to-the-build-directory/52024730
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/connect4_3d_ai/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/connect4_3d_ai/build)

add_compile_options(-Wall)

#find_package(pybind11 REQUIRED)

add_subdirectory(third_party/pybind11)

#include_directories(include ${Eigen_INCLUDE_DIRS})


pybind11_add_module(connect4_3d_ai_engine
        connect4_3d_ai/src/module.cpp
        connect4_3d_ai/src/Movement.h
        connect4_3d_ai/src/Movement.cpp
        connect4_3d_ai/src/Movement.h
        connect4_3d_ai/src/Properties.cpp
        connect4_3d_ai/src/Properties.h
        connect4_3d_ai/src/Node.cpp
        connect4_3d_ai/src/Node.h
        connect4_3d_ai/src/Engine.cpp
        connect4_3d_ai/src/Engine.h
        connect4_3d_ai/src/MCTS.cpp
        connect4_3d_ai/src/MCTS.h
        connect4_3d_ai/src/tools.cpp
        connect4_3d_ai/src/tools.h)


add_executable(MasterTheGameOfConnect4.out
        connect4_3d_ai/src/Movement.h
        connect4_3d_ai/src/Movement.cpp
        connect4_3d_ai/src/Movement.h
        connect4_3d_ai/src/Properties.cpp
        connect4_3d_ai/src/Properties.h
        connect4_3d_ai/src/Node.cpp
        connect4_3d_ai/src/Node.h
        connect4_3d_ai/src/Engine.cpp
        connect4_3d_ai/src/Engine.h
        connect4_3d_ai/src/MCTS.cpp
        connect4_3d_ai/src/MCTS.h
        connect4_3d_ai/src/tools.cpp
        connect4_3d_ai/src/tools.h)