#ifndef MASTERTHEGAMEOFCONNECT4_PROPERTIES_H
#define MASTERTHEGAMEOFCONNECT4_PROPERTIES_H

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>

class Properties {
public:
    double black_points{};
    double white_points{};
    double black_lines{};
    double white_lines{};

    Properties() = default;

    Properties(double black_points, double white_points, double black_lines, double white_lines);

    void print_properties() const;

    void output_properties() const;
};


#endif //MASTERTHEGAMEOFCONNECT4_PROPERTIES_H
