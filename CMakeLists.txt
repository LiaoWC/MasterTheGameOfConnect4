cmake_minimum_required(VERSION 3.19)
project(MasterTheGameOfConnect4)

set(CMAKE_CXX_STANDARD 14)


add_compile_options(-Wall)

#find_package(pybind11 REQUIRED)

add_subdirectory(pybind11)

#include_directories(include ${Eigen_INCLUDE_DIRS})

pybind11_add_module(example
        module.cpp
        mcts.cpp
        Movement.h
        Movement.cpp
        Movement.h
        Properties.cpp
        Properties.h
        Node.cpp
        Node.h
        Engine.cpp
        Engine.h
        MCTS.cpp
        MCTS.h
        tools.cpp
        tools.h)


add_executable(MasterTheGameOfConnect4
        mcts.cpp
        Movement.h
        Movement.cpp
        Movement.h
        Properties.cpp
        Properties.h
        Node.cpp
        Node.h
        Engine.cpp
        Engine.h
        MCTS.cpp
        MCTS.h
        tools.cpp
        tools.h)