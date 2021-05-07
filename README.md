# 

## TODO

## What can we try?
- Policy target pruning



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