    vector<Movement> gen_block_move() {
        int temp_state[6][6][6];
        for (int l = 0; l < 6; l++) {
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    temp_state[l][i][j] = this->board[l][i][j];
                }
            }
        }
        bool flag = true;
        bool flag2 = true;
        vector<Movement> all_possible_moves = this->get_next_possible_move();
        int opponent_color = (this->hands % 2 == 0) ? 2 : 1;
        int self_color = (this->hands % 2 == 0) ? 1 : 2;
        int dirs[26][3];
        int cnt = 0;
        for (int l = -1; l <= 1; l++) {
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    if (l == 0 && i == 0 && j == 0)
                        continue;
                    dirs[cnt][0] = l;
                    dirs[cnt][1] = i;
                    dirs[cnt][2] = j;
                    cnt++;
                }
            }
        }
        int dirs2[13][3] = {
            {0, 0, 1},
            {0, 1, 0},
            {1, 0, 0},
            {0, 1, 1},
            {1, 0, 1},
            {1, 1, 0},
            {0, 1, -1},
            {1, 0, -1},
            {1, -1, 0},
            {1, 1, 1},
            {1, 1, -1},
            {1, -1, 1},
            {-1, 1, 1}
        };
        vector<Movement> block_moves;
        for (auto& all_possible_move : all_possible_moves) {
            double temp_p = 0;
            flag = true;
            flag2 = true;
            for (auto& dir : dirs) {
                unique_ptr<int[]> new_pos(new int[3]);
                new_pos[0] = all_possible_move.l + dir[0];
                new_pos[1] = all_possible_move.x + dir[1];
                new_pos[2] = all_possible_move.y + dir[2];

                int l = new_pos[0]; 
                int i = new_pos[1]; 
                int j = new_pos[2];
                if (!boundary_test(new_pos) || this->board[l][i][j] != opponent_color)
                    continue;
                new_pos[0] += dir[0];
                new_pos[1] += dir[1];
                new_pos[2] += dir[2];
                l = new_pos[0];
                i = new_pos[1];
                j = new_pos[2];
                if (!boundary_test(new_pos) || this->board[l][i][j] != opponent_color)
                    continue;
                new_pos[0] += dir[0];
                new_pos[1] += dir[1];
                new_pos[2] += dir[2];
                l = new_pos[0];
                i = new_pos[1];
                j = new_pos[2];
                if (!boundary_test(new_pos) || this->board[l][i][j] == self_color)
                    continue;
                Movement block_move;
                block_move.l = all_possible_move.l;
                block_move.x = all_possible_move.x;
                block_move.y = all_possible_move.y;
                block_move.color = self_color;
                block_move.p += block_move.c;
                block_moves.push_back(block_move);
                flag = false;
            }
            int l = all_possible_move.l;
            int i = all_possible_move.x;
            int j = all_possible_move.y;
            //cout << l << " " << i << " " << j << " " << c << endl;
            for (auto& dir2 : dirs2) {
                int cnt = 1;
                for (int mul = 1; mul <= 2; mul++) {
                    unique_ptr<int[]> new_pos(new int[3]);
                    new_pos[0] = l + dir2[0] * mul;
                    new_pos[1] = i + dir2[1] * mul;
                    new_pos[2] = j + dir2[2] * mul;
                    if (!boundary_test(new_pos))
                        break;
                    if (temp_state[new_pos[0]][new_pos[1]][new_pos[2]] != self_color)
                        break;
                    cnt += 1;
                }
                for (int mul = 1; mul <= 2; mul++) {
                    unique_ptr<int[]> new_pos(new int[3]);
                    new_pos[0] = l + dir2[0] * -mul;
                    new_pos[1] = i + dir2[1] * -mul;
                    new_pos[2] = j + dir2[2] * -mul;
                    if (!boundary_test(new_pos))
                        break;
                    if (temp_state[new_pos[0]][new_pos[1]][new_pos[2]] != self_color)
                        break;
                    cnt += 1;
                }
                //cout << "cnt: " << cnt << endl;
                while (cnt >= 3) {
                    Movement temp_block_move;
                    temp_p += temp_block_move.c * 2;
                    flag2 = false;
                    cnt--;
                }
            }
            if (!flag2) {
                Movement block_move;
                block_move.l = all_possible_move.l;
                block_move.x = all_possible_move.x;
                block_move.y = all_possible_move.y;
                block_move.color = self_color;
                block_move.p = temp_p;
                block_moves.push_back(block_move);
            }
            if (flag && flag2) {
                Movement block_move;
                block_move.l = all_possible_move.l;
                block_move.x = all_possible_move.x;
                block_move.y = all_possible_move.y;
                block_move.color = self_color;
                block_moves.push_back(block_move);
            }
        }
        srand(time(nullptr));
        return block_moves;
    }

class Movement {
public:
    int l{};
    int x{};
    int y{};
    int color{};
    double p{};
    double c{};

    Movement() {
        this->l = 0;
        this->x = 0;
        this->y = 0;
        this->color = 0;
        this->p = 0;
        this->c = 0.05;
    };

    Movement(int l, int x, int y, int color) {
        this->l = l;
        this->x = x;
        this->y = y;
        this->color = color;
        this->p = 0;
        this->c = 0.05;
    }

    void config_c(double newc) {
        this->c = newc;
    }

    void print_movement() const {
        cout << "[" << l << ", " << x << ", " << y << "]" << endl;
    }

    friend bool operator==(const Movement& m1, const Movement& m2);
};