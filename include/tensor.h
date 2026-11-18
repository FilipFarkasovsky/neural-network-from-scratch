#pragma once
#include <vector>
#include <ostream>
#include <memory>

class Tensor{
    private:
        std::vector<float> _data;
        std::vector<std::size_t> _shape;
        std::vector<std::size_t> _stride;

    public:
        Tensor(float data);
        Tensor(const std::vector<float> data);
        Tensor(const std::vector<std::vector<float>> data);

        float &item();
        const float &item() const;
        std::size_t numel() const;
        
        // General indexing
        float& operator()(const std::vector<std::size_t>& indices);
        const float& operator()(const std::vector<std::size_t>& indices) const;

        // 1D indexing
        float &operator()(std::size_t i);
        const float &operator()(std::size_t i) const;

        // 2D indexing
        float &operator()(std::size_t i, std::size_t j);
        const float &operator()(std::size_t i, std::size_t j) const;

        const std::vector<std::size_t>& shape() const;
        const std::vector<std::size_t>& stride() const;

        friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);

        std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& other) const;
        std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& other) const;
};