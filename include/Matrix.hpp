#pragma once
#include <iostream>
#include <vector>
#include <cstddef>
#include <cassert>
#include <utility>

class Matrix
{
    private:
        size_t rows;
        size_t columns;
        std::vector<double> data;
        
    public:
        Matrix(size_t row,size_t cols)
        : rows(row),columns(cols),data(row*cols,0.0) {}

        double& operator()(size_t i,size_t j)
        {
            assert(i < rows && j < columns);
            return data[i*columns+j];//row-major
        }
        const double& operator()(size_t i,size_t j) const
        {
            assert(i < rows && j < columns);
            return data[i*columns+j];
        }//for read only matrices

        size_t row() const{return rows;}
        size_t cols() const{return columns;}
        
        Matrix submatrix(size_t row_start, size_t row_end, size_t col_start, size_t col_end) const;
        
        Matrix transpose() const;
        Matrix multiply(const Matrix& other) const;
        Matrix inverse() const;
        
        // Advanced Decompositions
        Matrix cholesky() const;
        std::pair<Matrix, Matrix> qr_decomposition() const;
        
        // Solvers
        Matrix forward_substitution(const Matrix& b) const;
        Matrix backward_substitution(const Matrix& b) const;
};