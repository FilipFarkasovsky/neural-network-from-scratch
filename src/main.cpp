#include <iostream>
#include <tensor.h>
#include <vector>

int main() {
    Tensor test_tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});

    std::cout << test_tensor << std::endl;
    return 0;
}