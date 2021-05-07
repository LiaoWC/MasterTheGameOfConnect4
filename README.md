# 

## TODO

## Print board
![](./img/Figure_1.png)
1. Call the function to output board content.
2. Run python to show the graph.
```c++
Node *cur_node = ......
cur_node->output_board_string_for_plot_state();
// This will output the board content to a file named "board_content_for_plotting". 
```
```shell
python3 plot_state.py ......
# Usage:
# (1)
python3 plot_state.py random  
# (2)
python3 plot_state.py board_string_text_file_path  
# (3)
python3 plot_state.py board[0][0][0] board[0][0][1] ... (size length * size length * size length values totally)  
```



### Random
```c++
#include <random>
#include <iostream>

int main() {
    int max_number = 10;
    int n_random = 10000;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, 10);

    std::vector<int> record(max_number, 0);
    for (int i = 0; i < 1000000; i++) {
        int number = dist(rng);
        record[number-1]++;
    }
    for (int i =0;i<record.size();i++){
        std::cout << i+1 << ": " << record[i] << std::endl;
    }
    return EXIT_SUCCESS;
}
```