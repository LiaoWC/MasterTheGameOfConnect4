#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "MCTS.h"
#include "Node.h"
#include "Movement.h"
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <string>

namespace py = pybind11;

typedef std::vector<std::vector<std::vector<int>>> vector3d_int;
typedef std::array<std::array<std::array<int, 6>, 6>, 6> array3d_int;

//////////////////////////////////////////////////////////////////////////
class PyNode {
public:
    std::shared_ptr<Node> data;

    PyNode(std::shared_ptr<Node> node) {
        this->data = node;
    }

    PyNode get_node_after_playing(Movement move) {
        return PyNode(this->data->get_node_after_playing(move));
    }

    pybind11::array_t<float> get_board() {
        std::cout << "!!!!!!!" << std::endl;
        pybind11::array_t<int, pybind11::array::c_style> arr({6, 6, 6});
        std::cout << "!!!!!!!" << std::endl;
        auto ra = arr.mutable_unchecked();
        std::cout << "!!!!!!!" << std::endl;
        //        for (size_t i = 0; i < N; i++) {
        //            for (size_t j = 0; j < M; j++) {
        //                ra(i, j) = vals[i][j];
        //            };
        //        };

        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < 6; k++) {
                    ra(i, j, k) = this->data->board[i][j][k];
                }
            }
        }
        std::cout << "!!!!!!!" << std::endl;
        return arr;
        //
        //
        //        return v;
    }

    int get_hands() {
        return this->data->hands;
    }

    Movement get_move() {
        return this->data->move;
    }

    int get_visit_cnt() {
        return this->data->visiting_count;
    }

    int get_value_sum() {
        return this->data->value_sum;
    }

    bool get_is_leaf() {
        return this->data->is_leaf;
    }

    int get_is_root() {
        return this->data->is_root;
    }

    int get_is_terminated() {
        return this->data->is_terminated;
    }

    std::vector<std::shared_ptr<Node>> get_children() {
        return this->data->children;
    }

    Properties get_properties() {
        return this->data->my_properties;
    }

    std::shared_ptr<Node> get_parent() {
        auto ptr = this->data->parent.lock();
        if (!ptr) {
            std::runtime_error("Get parent shared ptr failed.");
        }
        return ptr;
    }
};

PyNode get_init_node() {
    PyNode py_node(MCTS::get_init_node());
    return py_node;
}

class PyMCTS {
public:
    MCTS mcts;

    PyMCTS(const PyNode &root, int max_simulation_cnt, double max_time_sec, bool print_simulation_cnt) {
        this->mcts = MCTS(root.data, max_simulation_cnt, max_time_sec, print_simulation_cnt);
    }

    Movement run(bool playout_use_block_move,
                 bool reward_add_score_diff,
                 bool first_hand_center,
                 bool dominate_pruning,
                 bool prior_value) {
        return this->mcts.run(playout_use_block_move,
                              reward_add_score_diff,
                              first_hand_center,
                              dominate_pruning,
                              prior_value, false);
    }
};

//std::vector<std::string> get_mcts_result(PyNode root,
//                                         int max_simulation_cnt,
//                                         double max_time_sec,
//                                         bool print_simulation_cnt){
//    MCTS mcts(root.data,);
//}

//////////////////////////////////////////////////////////////////////////
/* Bind MatrixXd (or some other Eigen type) to Python */
typedef Eigen::MatrixXd Matrix;

typedef Matrix::Scalar Scalar;
constexpr bool rowMajor = Matrix::Flags & Eigen::RowMajorBit;


int add(int i, int j, int k) {
    return i + j * k;
}

std::vector<int> get_vector() {
    std::vector<int> a;
    for (int i = 0; i < 10; i++) {
        a.push_back(i);
    }
    return a;
}

std::tuple<int, int, int, int> genmove() {
    std::shared_ptr<Node> cur_node = MCTS::get_init_node();
    Movement move;
    for (int i = 0; i < 5; i++) {
        MCTS mcts(cur_node, 99, 1, true);
        move = mcts.run(false, false, true, true, true, false);
        cur_node = cur_node->get_node_after_playing(move);
    }
    move = cur_node->move;
    return std::make_tuple(move.l, move.x, move.y, move.color);
}

int kkkkkkkk(Movement m) {
    return m.color;
}

///////////////////////////////////////////////////////////////////////
class Mat {
public:
    Mat(int rows, int cols) : m_rows(rows), m_cols(cols) {
        m_data = new int[rows * cols];
    }

    int *data() { return m_data; }

    int rows() const { return m_rows; }

    int cols() const { return m_cols; }

private:
    int m_rows, m_cols;
    int *m_data;
};


///////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(connect4_3d_ai_engine, m) {
    // optional module docstring
    m.doc() = "Connect4-3D AI Engine";
    //
    m.def("get_init_node", &get_init_node);
    ///////////////////////////////////////////////
    // expose add function, and add keyword arguments and default arguments
    m.def("add", &add, "A function which adds two numbers", py::arg("i") = 1, py::arg("j") = 2, py::arg("k") = 2)
            .def("get_move", &genmove)
            .def("get_vector", &get_vector);
    // exporting variables
    m.attr("the_answer") = 42;
    py::object world = py::cast("World");
    m.attr("what") = world;
    m.def("kkkkkkkk", &kkkkkkkk);
    ////////////////////////////////////////////////////////////////
    py::class_<PyNode>(m, "Node")
            .def(py::init([](py::buffer const &b, int hands, Movement move, Properties properties) {
                py::buffer_info info = b.request();
                if (info.format != py::format_descriptor<int>::format() ||
                    info.ndim != 3 || info.shape[0] != 6 || info.shape[1] != 6 || info.shape[2] != 6)
                    throw std::runtime_error(
                            "Incompatible buffer format! Shape must be (6, 6, 6) and the data type must be int32.");
                int *v = new int[6 * 6 * 6];
                memcpy(v, info.ptr, sizeof(int) * 6 * 6 * 6);
                // std::vector<std::vector<std::vector<int>>> data;
                // std::array<int,6*6*6> data{};
                array3d_int data{};
                // std::shared_ptr<int[]> data = std::make_shared<int[]>(6 * 6 * 6);
                for (int i = 0; i < 6; i++) {
                    // data[i].resize(6);
                    for (int j = 0; j < 6; j++) {
                        // data[i][j].resize(6);
                        for (int k = 0; k < 6; k++) {
                            // data[i][j][k] = v[i * 36 + j * 6 + k];
                            std::cout << v[i * 36 + j * 6 + k] << std::endl;
                            // data[i * 36 + j * 6 + k] = v[i * 36 + j * 6 + k];
                            data[i][j][k] = v[i * 36 + j * 6 + k];
                        }
                    }
                }
                std::cout << "888888888888888888" << std::endl;
                delete[]v;
                std::cout << "888888888888888888" << std::endl;
                return PyNode(std::make_shared<Node>(data, hands, move, properties));
            }), py::arg("board"), py::arg("hands"), py::arg("move"), py::arg("properties"))
            .def("get_board", &PyNode::get_board)
            .def("get_hands", &PyNode::get_hands)
            .def("get_move", &PyNode::get_move)
            .def("get_visit_cnt", &PyNode::get_visit_cnt)
            .def("get_value_sum", &PyNode::get_value_sum)
            .def("get_is_leaf", &PyNode::get_is_leaf)
            .def("get_is_root", &PyNode::get_is_root)
            .def("get_is_terminated", &PyNode::get_is_terminated)
            .def("get_children", &PyNode::get_children)
            .def("get_properties", &PyNode::get_properties)
            .def("get_parent", &PyNode::get_parent)
            .def("get_node_after_playing", &PyNode::get_node_after_playing);
    py::class_<Movement>(m, "Movement")
            .def(py::init<int, int, int, int>(), py::arg("l"), py::arg("x"), py::arg("y"), py::arg("color"))
            .def_readwrite("l", &Movement::l)
            .def_readwrite("x", &Movement::x)
            .def_readwrite("y", &Movement::y)
            .def_readwrite("color", &Movement::color);
    py::class_<Properties>(m, "Properties")
            .def(py::init<>())
            .def(py::init<double, double, double, double>(),
                 py::arg("black_points") = 0, py::arg("white_points") = 0,
                 py::arg("black_lines") = 0, py::arg("white_lines") = 0)
            .def_readwrite("black_points", &Properties::black_points)
            .def_readwrite("white_points", &Properties::white_points)
            .def_readwrite("black_lines", &Properties::black_lines)
            .def_readwrite("white_lines", &Properties::white_lines);
    py::class_<PyMCTS>(m, "MCTS")
            .def(py::init<PyNode, int, double, bool>(),
                 py::arg("root"),
                 py::arg("max_simulation_cnt"),
                 py::arg("max_time_sec"),
                 py::arg("print_simulation_cnt"))
            .def("run", &PyMCTS::run,
                 py::arg("playout_use_block_move"),
                 py::arg("reward_add_score_diff"),
                 py::arg("first_hand_center"),
                 py::arg("dominate_pruning"),
                 py::arg("prior_value"));
}