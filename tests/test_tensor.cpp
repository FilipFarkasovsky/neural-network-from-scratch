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

void test_invalid(const std::string& name, bool throws) {
    test(name, throws);
}

void test_constructors();
void test_indexing();

int main() {
    test_constructors();
    test_indexing();
    // test_addition();
    // test_matmul();

    std::cout << "\n" << GREEN << passed << " passed" << RESET
              << ", " << RED << failed << " failed" << RESET << '\n';

    return failed == 0 ? 0 : 1;
}

void test_constructors() {
    // 1. SCALAR
    Tensor scalar(5.0f);
    test("Scalar construction", scalar.numel() == 1);
    test("Scalar value", scalar.item() == 5.0f);


    // 2. VECTOR
    Tensor v(std::vector<float>{1, 2, 3});
    test("Vector construction", v.shape() == std::vector<std::size_t>{3});
    test("Vector numel", v.numel() == 3);

    Tensor empty_vector(std::vector<float>{});
    test("Empty vector construction", empty_vector.shape() == std::vector<std::size_t>{0});
    test("Empty vector numel", empty_vector.numel() == 0);

    // 3. MATRIX
    Tensor m(std::vector<std::vector<float>>{
        {1, 2, 3},
        {4, 5, 6}
    });
    test("Matrix construction", m.shape() == std::vector<std::size_t>{2, 3});
    test("Matrix numel", m.numel() == 6);

    Tensor row(std::vector<std::vector<float>>{{1, 2, 3}});
    test("Single-row matrix", row.shape() == std::vector<std::size_t>{1, 3});

    Tensor column(std::vector<std::vector<float>>{{1}, {2}, {3}});
    test("Single-column matrix", column.shape() == std::vector<std::size_t>{3, 1});

    Tensor single(std::vector<std::vector<float>>{{42}});
    test("1x1 matrix", single.numel() == 1);

    // 4. FLAT DATA + SHAPE
    Tensor t3(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, std::vector<std::size_t>{2, 2, 2});
    test("3D tensor construction", t3.shape() == std::vector<std::size_t>{2, 2, 2});
    test("3D tensor numel", t3.numel() == 8);
    test("3D tensor strides", t3.stride() == std::vector<std::size_t>{4, 2, 1});

    // Zero-sized dimensions
    Tensor zero_dim(std::vector<float>{}, std::vector<std::size_t>{2, 0, 3});

    test("Zero-sized dimension", zero_dim.numel() == 0);

    // Scalar represented by flat data + empty shape
    Tensor scalar2(std::vector<float>{42}, std::vector<std::size_t>{});

    test("Scalar from flat data", scalar2.item() == 42);

    // 5. INVALID CONSTRUCTIONS

    // Inconsistent row lengths
    test_invalid("Ragged matrix", [] {
        Tensor t(std::vector<std::vector<float>>{
            {1, 2},
            {3}
        });
    });

    // Empty nested matrix (if your implementation rejects it)
    test_invalid("Empty matrix", [] {
        Tensor t(std::vector<std::vector<float>>{});
    });

    // Data size does not match shape
    test_invalid("Incorrect shape size", [] {
        Tensor t(
            std::vector<float>{1, 2, 3, 4},
            std::vector<std::size_t>{2, 3}
        );
    });

    // Too many dimensions in shape, but wrong data size
    test_invalid("Incorrect 3D shape size", [] {
        Tensor t(
            std::vector<float>{1, 2, 3},
            std::vector<std::size_t>{2, 2, 2}
        );
    });

    // Empty shape requires exactly one element
    test_invalid("Empty shape with no data", [] {
        Tensor t(
            std::vector<float>{},
            std::vector<std::size_t>{}
        );
    });
}

void test_indexing() {
    Tensor m(std::vector<std::vector<float>>{{1, 2}, {3, 4}});

    test("2D indexing", m(1, 1) == 4);

    m(0, 0) = 10;

    test("Element modification", m(0, 0) == 10);
}
