#include "Properties.h"

Properties::Properties(double black_points,
                       double white_points,
                       double black_lines,
                       double white_lines) {
    this->black_points = black_points;
    this->white_points = white_points;
    this->black_lines = black_lines;
    this->white_lines = white_lines;
}

void Properties::print_properties() const {
    std::cout << "Bp: " << black_points
              << ", Wp: " << white_points
              << ", Bl: " << black_lines
              << ", Wl: " << white_lines << std::endl;
}

void Properties::output_properties() const {
    std::ofstream ofs("output_properties_for_plotting.txt", std::fstream::trunc);
    ofs << "BlackPoints: " << this->black_points
        << ", WhitePoints: " << this->white_points
        << ", BlackLines: " << this->black_lines
        << ", WhiteLines: " << this->white_lines;
    ofs.close();
    std::cout << "Output properties string to \"output_properties_for_plotting.txt\" done!" << std::endl;
}
