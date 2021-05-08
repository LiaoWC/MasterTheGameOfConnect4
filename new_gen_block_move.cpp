vector<Movement> gen_block_move() {    
        bool flag = true;
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
        vector<Movement> block_moves;
        for (auto& all_possible_move : all_possible_moves) {
            flag = true;
            for (auto& dir : dirs) {
                unique_ptr<int[]> new_pos_a(new int[3]);
                new_pos_a[0] = all_possible_move.l + dir[0];
                new_pos_a[1] = all_possible_move.x + dir[1];
                new_pos_a[2] = all_possible_move.y + dir[2];

                unique_ptr<int[]> new_pos_b(new int[3]);
                new_pos_b[0] = all_possible_move.l + dir[0];
                new_pos_b[1] = all_possible_move.x + dir[1];
                new_pos_b[2] = all_possible_move.y + dir[2];
                int l = new_pos_b[0], i = new_pos_b[1], j = new_pos_b[2];
                if (boundary_test(new_pos_b) && this->board[l][i][j] == self_color) {
                    new_pos_b[0] += dir[0];
                    new_pos_b[1] += dir[1];
                    new_pos_b[2] += dir[2];
                    l = new_pos_b[0];
                    i = new_pos_b[1];
                    j = new_pos_b[2];
                    if (boundary_test(new_pos_b) && this->board[l][i][j] == self_color) {
                        Movement block_move_b;
                        block_move_b.l = all_possible_move.l;
                        block_move_b.x = all_possible_move.x;
                        block_move_b.y = all_possible_move.y;
                        block_move_b.color = self_color;
                        block_move_b.p += block_move_b.c * 2;
                        block_moves.push_back(block_move_b);
                        flag = false;
                    }
                }

                l = new_pos_a[0]; 
                i = new_pos_a[1]; 
                j = new_pos_a[2];
                if (!boundary_test(new_pos_a) || this->board[l][i][j] != opponent_color)
                    continue;
                new_pos_a[0] += dir[0];
                new_pos_a[1] += dir[1];
                new_pos_a[2] += dir[2];
                l = new_pos_a[0];
                i = new_pos_a[1];
                j = new_pos_a[2];
                if (!boundary_test(new_pos_a) || this->board[l][i][j] != opponent_color)
                    continue;
                Movement block_move_a;
                block_move_a.l = all_possible_move.l;
                block_move_a.x = all_possible_move.x;
                block_move_a.y = all_possible_move.y;
                block_move_a.color = self_color;
                block_move_a.p += block_move_a.c;
                block_moves.push_back(block_move_a);
                flag = false;
            }
            if (flag) {
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