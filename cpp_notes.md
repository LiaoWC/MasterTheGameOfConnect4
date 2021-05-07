# Notes about C/C++

## C++ Shared Point Demo & Comparison with Conventional Raw Pointer
C++ smart pointers reference: https://www.geeksforgeeks.org/auto_ptr-unique_ptr-shared_ptr-weak_ptr-2/

```c++
#include <chrono>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <vector>

using namespace std;


class NodeUseSmart {
public:
    int a[999999];
    shared_ptr<NodeUseSmart> next;
    weak_ptr<NodeUseSmart> prev;
};

void test_doubly_linked_list_using_smart_ptr() {
    cout << "=== Start test using SMART pointers ===" << endl;
    shared_ptr<NodeUseSmart> head = make_shared<NodeUseSmart>();
    weak_ptr<NodeUseSmart> cur = head;
    for (int i = 0; i < 999; i++) {
        auto tmp = make_shared<NodeUseSmart>();
        if (auto p = cur.lock()) {
            p->next = tmp;
        } else {
            perror("weak_ptr lock error");
        }
        cur = tmp;
    }
    cout << "allocate done (delete after 5 sec)" << endl;
    sleep(5);
    // Delete
    head.reset();
    //
    cout << "Back to main function after 5 sec" << endl;
    sleep(5);
}


class NodeUseRaw {
public:
    int a[999999];
    NodeUseRaw *next;
    NodeUseRaw *prev;
};

void test_doubly_linked_list_using_raw_ptr() {
    cout << "=== Start test using conventional RAW pointers ===" << endl;
    NodeUseRaw *head = new NodeUseRaw;
    head->prev = nullptr;
    NodeUseRaw *cur = head;

    for (int i = 0; i < 999; i++) {
        NodeUseRaw *tmp = new NodeUseRaw();
        // I found the following two are different.
        // The former will use lots of memory, but the latter seems allocate nothing.
        // NodeUseRaw *tmp = new NodeUseRaw();
        // NodeUseRaw *tmp = new NodeUseRaw;
        cur->next = tmp;
        tmp->prev = cur;
        cur = tmp;
    }
    cout << "allocate done (delete after 5 sec)" << endl;
    sleep(5);
    // Delete
    delete head; // Method that not works
    //
    cout << "Back to main function after 5 sec" << endl;
    sleep(5);
}

int main() {
    cout << "Open your program to monitor how much memory this program uses." << endl;
    sleep(3);

    /* We'll see that we free all the memory allocated just by reset the head shared pointer.
     */
    test_doubly_linked_list_using_smart_ptr();

    /* In this part, you'll see the memory allocated won't be freed
     * even though it goes bach to the main function.
     * You may need to design some ways to free the memory.
     * */
    test_doubly_linked_list_using_raw_ptr();

    /////////////////////////////
    cout << "All done!" << endl;
    cout << "It'll terminate in 3 seconds." << endl;
    sleep(3);
    return 0;
}
```