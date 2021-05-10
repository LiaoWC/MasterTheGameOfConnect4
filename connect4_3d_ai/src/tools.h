#ifndef MASTERTHEGAMEOFCONNECT4_TOOLS_H
#define MASTERTHEGAMEOFCONNECT4_TOOLS_H
#include <vector>
#include <iostream>
#include <iomanip>


namespace tools {
    template<typename T>
    void print_vector_2d_plane(std::vector<std::vector<T>> plane) {
        std::cout << std::endl;
        for (const auto &line: plane) {
            for (const auto item: line) {
                std::cout << std::setw(14) << item;
            }
            std::cout << std::endl;
        }
    }
}

#endif //MASTERTHEGAMEOFCONNECT4_TOOLS_H
