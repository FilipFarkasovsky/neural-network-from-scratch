
#include "tensor.h"
#include <iostream>
#include <vector>

const std::string GREEN = "\033[32m";
const std::string RED   = "\033[31m";
const std::string RESET = "\033[0m";

int passed = 0;
int failed = 0;

void test(const std::string& name, bool condition) {
    if (condition) {
        std::cout << GREEN << "[PASS] " << RESET << name << '\n';
        ++passed;
    } else {
        std::cout << RED << "[FAIL] " << RESET << name << '\n';
        ++failed;
    }
}

int main() {
    // 1D tensor
    Tensor v(std::vector<float>{1.0f, 2.0f, 3.0f});

    test("Vector construction", v.shape() == std::vector<std::size_t>{3});

    test("Vector indexing", v(std::vector<std::size_t>{0}) == 1.0f);

    test("1D indexing", v(1) == 2.0f);

    v(1) = 5.0f;
    test("Element modification", v(1) == 5.0f);

    // 2D tensor
    Tensor m(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});

    test("Matrix construction", m.shape() == std::vector<std::size_t>{2, 2});

    test("Matrix vector indexing", m(std::vector<std::size_t>{1, 0}) == 3.0f);

    test("2D indexing", m(1, 1) == 4.0f);

    std::cout << "\n" << GREEN << passed << " passed" << RESET
              << ", " << RED << failed << " failed" << RESET << '\n';

    return failed == 0 ? 0 : 1;
}
