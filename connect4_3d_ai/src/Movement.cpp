#include "Movement.h"

bool operator==(const Movement &m1, const Movement &m2) {
    // TODO: get_all_possible_move function seems to give no player color information. It gives zero always?
    // To prevent some unexpected color occurring, we only check l, i, j
    if (m1.l == m2.l && m1.x == m2.x && m1.y == m2.y) { return true; } else { return false; }
}

Movement::Movement() {
    this->l = 0;
    this->x = 0;
    this->y = 0;
    this->color = 0;
    this->prior = 0;
    this->base_c = 0.05;
}

Movement::Movement(int l, int x, int y, int color) {
    this->l = l;
    this->x = x;
    this->y = y;
    this->color = color;
    this->prior = 0;
    this->base_c = 0.05;
}

void Movement::print_movement() const {
    std::cout << "[" << l << ", " << x << ", " << y << "]" << std::endl;
}