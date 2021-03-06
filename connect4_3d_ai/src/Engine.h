#ifndef MASTERTHEGAMEOFCONNECT4_ENGINE_H
#define MASTERTHEGAMEOFCONNECT4_ENGINE_H

#include "Node.h"
#include "MCTS.h"
#include "Movement.h"
#include "Properties.h"
#include "tools.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>

#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <array>
#include <algorithm>
#include <iomanip>

#define NEG_INFINITE -9999999
#define POS_INFINITE 9999999
#define ILLEGAL_COLOR = -1
#define EMPTY_COLOR 0
#define BLACK_PLAYER_COLOR 1
#define WHITE_PLAYER_COLOR 2
// For expand's return value
#define EXPAND_NOT_PRUNE_PARENT 1
#define EXPAND_PRUNE_PARENT 2
#define EXPAND_ROOT_HAS_DOMINANT_MOVE 3

class Engine {
public:

};




#endif //MASTERTHEGAMEOFCONNECT4_ENGINE_H
