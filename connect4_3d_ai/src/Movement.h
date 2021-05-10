#ifndef MASTERTHEGAMEOFCONNECT4_MOVEMENT_H
#define MASTERTHEGAMEOFCONNECT4_MOVEMENT_H

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>


class Movement {
public:
    int l{};
    int x{};
    int y{};
    int color{};
    double prior{};
    double base_c{}; // TODO: make it can be tuned

    Movement();

    Movement(int l, int x, int y, int color);

    void print_movement() const;

    friend bool operator==(const Movement &m1, const Movement &m2);
};

#endif //MASTERTHEGAMEOFCONNECT4_MOVEMENT_H
