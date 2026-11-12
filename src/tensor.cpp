#include "tensor.h"
#include <iostream>
#include <vector>

Tensor::Tensor(float data) : _data(data), _shape{}, _stride{} {}

Tensor::Tensor(std::vector<float> data) : _data(data), _shape{data.size()}, _stride{1} {}

Tensor::Tensor(std::vector<std::vector<float>> data) 
    : _shape{data.size(), data[0].size()}, _stride{data[0].size(), 1}
    {
        // check consistency of dimensions
        std::size_t n_expected_columns = data[0].size();
        for(std::size_t i = 0; i < data.size(); i++)
        {
            if (data[i].size() != n_expected_columns)
            {
                throw std::invalid_argument("Dimensions are inconsistent.");
            }
        }
        
        // store in row major format
        for(std::size_t i = 0; i < data.size(); i++)
        {
            for (std::size_t j = 0; j < data[i].size(); j++)
            {
                _data.push_back(data[i][j]);
            }
        }
    }

const float &Tensor::item() const {
    if(_data.size() == 1)
    {
        return _data[0];
    }
    else 
    {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
}

float &Tensor::item(){
    if(_data.size() == 1)
    {
        return _data[0];
    }
    else 
    {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
}

const std::vector<std::size_t>& Tensor::shape() const {
    return _shape;
}

const std::vector<std::size_t>& Tensor::stride() const {
    return _stride;
}

float& Tensor::operator()(const std::vector<std::size_t>& indices) {
    if (indices.size() != _shape.size()) {
        throw std::invalid_argument("Incorrect number of indices.");
    }

    std::size_t flat_index = 0;

    for (std::size_t k = 0; k < indices.size(); k++) 
    {
        if (indices[k] >= _shape[k]) {
            throw std::out_of_range("Index out of bounds.");
        }

        flat_index += indices[k] * _stride[k];
    }

    return _data[flat_index];
}

const float& Tensor::operator()(const std::vector<std::size_t>& indices) const {
    if (indices.size() != _shape.size()) {
        throw std::invalid_argument("Incorrect number of indices.");
    }

    std::size_t flat_index = 0;

    for (std::size_t k = 0; k < indices.size(); k++) 
    {
        if (indices[k] >= _shape[k]) {
            throw std::out_of_range("Index out of bounds.");
        }

        flat_index += indices[k] * _stride[k];
    }

    return _data[flat_index];
}

float &Tensor::operator()(std::size_t i){
    return (*this)(std::vector<std::size_t>{i});
}

const float& Tensor::operator()(std::size_t i) const {
    return (*this)(std::vector<std::size_t>{i});
}

float& Tensor::operator()(std::size_t i, std::size_t j) {
    return (*this)(std::vector<std::size_t>{i, j});
}

const float& Tensor::operator()(std::size_t i, std::size_t j) const {
    return (*this)(std::vector<std::size_t>{i, j});
}

void print_tensor(std::ostream& os, 
                  const Tensor& tensor,
                  std::vector<std::size_t>& indices) {

    if (indices.size() == tensor.shape().size())
    {
        os << tensor(indices);
        return;
    }

    os << "[";

    for (std::size_t i = 0; i < tensor.shape()[indices.size()]; i++) 
    {
        indices.push_back(i);
        print_tensor(os, tensor, indices);
        indices.pop_back();
        if (i + 1 < tensor.shape()[indices.size()]) 
        {
            os << ", ";
        }
    }

    os << "]";
}

std::ostream& operator<<(std::ostream& os, const Tensor& obj) {
    if (obj.shape().empty()) {
        os << obj.item();
        return os;
    }

    std::vector<std::size_t> indices;
    print_tensor(os, obj, indices);

    return os;
}