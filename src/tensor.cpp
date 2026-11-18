#include "tensor.h"
#include <iostream>
#include <vector>

Tensor::Tensor(std::vector<float> data, std::vector<std::size_t> shape)
    : _data(std::move(data)),
      _shape(std::move(shape)),
      _stride(_shape.size())
{
    std::size_t stride = 1;

    // Calculate row-major strides
    for (std::size_t i = _shape.size(); i-- > 0;) {
        _stride[i] = stride;
        stride *= _shape[i];
    }

    // Validate the number of elements
    if (stride != _data.size()) {
        throw std::invalid_argument("Data size does not match tensor shape.");
    }
}

Tensor::Tensor(float data) : _data{data}, _shape{}, _stride{} {}

Tensor::Tensor(const std::vector<float> data) : _data(data), _shape{data.size()}, _stride{1} {}

Tensor::Tensor(const std::vector<std::vector<float>> data) 
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

std::size_t Tensor::numel() const {
    return _data.size();
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

float& Tensor::operator()(std::size_t i) {
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

std::shared_ptr<Tensor> Tensor::operator+(
    const std::shared_ptr<Tensor>& other) const
{
    // Same shape
    if (_shape == other->_shape) {
        std::vector<float> result(_data.size());

        for (std::size_t i = 0; i < _data.size(); ++i)
            result[i] = _data[i] + other->_data[i];

        return std::make_shared<Tensor>(std::move(result), _shape);
    }

    // Scalar + tensor
    if (_shape.empty() || other->_shape.empty()) {
        const bool scalar_first = _shape.empty();

        const float scalar = scalar_first ? item() : other->item();
        const auto& data = scalar_first ? other->_data : _data;
        const auto& shape = scalar_first ? other->_shape : _shape;

        std::vector<float> result(data.size());

        for (std::size_t i = 0; i < data.size(); ++i)
            result[i] = data[i] + scalar;

        return std::make_shared<Tensor>(
            std::move(result), shape);
    }

    // Vector + matrix or matrix + vector
    // Note: Broadcasting across rows
    if ((_shape.size() == 1 && other->_shape.size() == 2) ||
        (_shape.size() == 2 && other->_shape.size() == 1)) {

        const bool vector_first = _shape.size() == 1;

        const auto& matrix_shape = vector_first ? other->_shape : _shape;
        const auto& matrix_data  = vector_first ? other->_data : _data;
        const auto& vector_data  = vector_first ? _data : other->_data;

        const std::size_t rows = matrix_shape[0];
        const std::size_t cols = matrix_shape[1];

        if (vector_data.size() != cols)
            throw std::invalid_argument("Incompatible shapes.");

        std::vector<float> result(rows * cols);

        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                result[i * cols + j] = matrix_data[i * cols + j] + vector_data[j];
            }
        }

        return std::make_shared<Tensor>(std::move(result), matrix_shape);
    }

    // General broadcasting - will not be used probly
    const std::size_t rank = std::max(_shape.size(), other->_shape.size());
    std::vector<std::size_t> shape(rank);

    // Determine output shape
    for (std::size_t i = 0; i < rank; ++i) {
        const std::size_t a = i < _shape.size()
            ? _shape[_shape.size() - 1 - i] : 1;

        const std::size_t b = i < other->_shape.size()
            ? other->_shape[other->_shape.size() - 1 - i] : 1;

        if (a != b && a != 1 && b != 1)
            throw std::invalid_argument("Incompatible shapes.");

        shape[rank - 1 - i] = (a == 1) ? b : a;
    }

    // Calculate output strides
    std::vector<std::size_t> strides(rank);
    std::size_t size = 1;

    for (std::size_t i = rank; i-- > 0;) {
        strides[i] = size;
        size *= shape[i];
    }

    // Calculate result
    std::vector<float> result(size);

    for (std::size_t i = 0; i < size; ++i) {
        std::size_t ia = 0;
        std::size_t ib = 0;

        for (std::size_t d = 0; d < rank; ++d) {
            const std::size_t index = (i / strides[d]) % shape[d];

            // Map output index to the first input
            if (d >= rank - _shape.size()) {
                const std::size_t k = d - (rank - _shape.size());

                if (_shape[k] != 1)
                    ia += index * _stride[k];
            }

            // Map output index to the second input
            if (d >= rank - other->_shape.size()) {
                const std::size_t k = d - (rank - other->_shape.size());

                if (other->_shape[k] != 1)
                    ib += index * other->_stride[k];
            }
        }

        result[i] = _data[ia] + other->_data[ib];
    }

    return std::make_shared<Tensor>(std::move(result), shape);
}

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor>& other) const
{
    // Vector * Matrix
    if (_shape.size() == 1 && other->_shape.size() == 2) {

        const std::size_t rows = other->_shape[0];
        const std::size_t cols = other->_shape[1];

        if (_shape[0] != rows)
            throw std::invalid_argument("Incompatible dimensions.");

        std::vector<float> result(cols, 0.0f);

        for (std::size_t i = 0; i < rows; ++i) {
            const float v = _data[i];

            for (std::size_t j = 0; j < cols; ++j) {
                result[j] += v * other->_data[i * cols + j];
            }
        }

        return std::make_shared<Tensor>(std::move(result));
    }

    // Matrix * Vector
    if (_shape.size() == 2 && other->_shape.size() == 1) {
        const std::size_t rows = _shape[0];
        const std::size_t cols = _shape[1];

        if ((*other)._shape[0] != cols)
            throw std::invalid_argument("Matrix * Vector: incompatible dimensions.");

        std::vector<float> result(rows, 0.0f);

        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                result[i] += (*other)._data[j] * _data[i * cols + j];
            }
        }

        return std::make_shared<Tensor>(std::move(result));
    }

    // Matrix * Matrix
    if(_shape.size() == 2 && other->_shape.size() == 2 && _shape[1] == other->_shape[0]){
        const std::size_t rows = _shape[0];
        const std::size_t cols = other->_shape[1];
        const std::size_t inner = _shape[1];

        std::vector<float> result(rows * cols, 0.0f);

        const auto& A = _data;
        const auto& B = other->_data;

        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t k = 0; k < inner; ++k) {
                const float a = A[i * inner + k];
                for (std::size_t j = 0; j < cols; ++j) {
                    result[i * cols + j] += a * B[k * cols + j];
                }
            }
        }

        return std::make_shared<Tensor>(std::move(result), std::vector<std::size_t>{rows, cols});
    }   
    
    // TODO: Implement batch matrix multiplication
    throw std::invalid_argument("Matrix Multiplications of this shapes is not valid.");
}