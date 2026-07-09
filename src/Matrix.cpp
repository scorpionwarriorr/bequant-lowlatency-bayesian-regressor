#include "Matrix.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

Matrix Matrix::submatrix(size_t row_start, size_t row_end, size_t col_start, size_t col_end) const
{
    assert(row_start <= row_end && row_end <= this->row());
    assert(col_start <= col_end && col_end <= this->cols());
    
    size_t new_rows = row_end - row_start;
    size_t new_cols = col_end - col_start;
    
    Matrix sub(new_rows, new_cols);
    for (size_t i = 0; i < new_rows; i++) {
        for (size_t j = 0; j < new_cols; j++) {
            sub(i, j) = (*this)(row_start + i, col_start + j);
        }
    }
    return sub;
}
Matrix Matrix::transpose() const
{
    Matrix T(this->cols(),this->row());
    for(int i=0;i<T.row();i++)
    {
        for(int j=0;j<T.cols();j++)
        {
            T(i,j)=(*this)(j,i);
        }
    }
    
    return T;

}

Matrix Matrix::multiply(const Matrix& other) const
{
    assert(this->cols()==other.row());
    Matrix T(this->row(),other.cols());
    for(int i=0;i<T.row();i++)
    {
        for(int j=0;j<T.cols();j++)
        {
            double sum=0.0;
            for(int k=0;k<this->cols();k++)
            {
                sum+=(*this)(i,k)*other(k,j);
            }
            T(i,j)=sum;
        }
    }
    return T;
}

Matrix Matrix::inverse() const
{
    assert(this->row()==this->cols());//inverse only for square matrices
    size_t n = this->row();

    //Now the augmented matrix [A|I]
    Matrix aug(n,2*n);

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            aug(i,j)=(*this)(i,j);
        }
    }//left hand side of the aug matrix

    for(int i=0;i<n;i++)
    {
        for(int j=n;j<2*n;j++)
        {
            if(i==j-n)
            {
                aug(i,j)=1.0;
            }
            else
            {
                aug(i,j)=0.0;
            }
        }
    }//putting the identity matrix on the right hand side

    //Now we use "gaussian elimination"
    for(int i=0;i<n;i++)
    {
        double ele=aug(i,i);
        assert(std::abs(ele) > 1e-12);// to ensure no division by 0
        for(int j=0;j<2*n;j++)
        {
            aug(i,j)=aug(i,j)/ele;
        }
        for(int r=0;r<n;r++)
        {
            if(r==i)
            {
                continue;
            }
            double diff=aug(r,i);
            for(int c=0;c<2*n;c++)
            {
                aug(r,c)=aug(r,c)-diff*aug(i,c);
            }
        }
    }
    Matrix inv(n,n);
        for(int i=0;i<n;i++)
        {
            for(int j=n;j<2*n;j++)
            {
                inv(i,j-n)=aug(i,j);
            }
        }
    return inv;

}

Matrix Matrix::cholesky() const
{
    // Cholesky decomposition requires a square matrix
    assert(this->row() == this->cols());
    size_t n = this->row();
    
    // Create a new matrix L initialized to 0.0
    Matrix L(n, n);
    
    // Implement the Cholesky algorithm
    for (size_t i = 0; i < n; i++) 
    {
        for (size_t j = 0; j <= i; j++) 
        {
            double sum = 0.0;
            
            // If we are on the diagonal
            if (j == i) 
            {
                for (size_t k = 0; k < j; k++) 
                {
                    sum += (L(j, k) * L(j, k));
                }
                L(j, j) = std::sqrt((*this)(j, j) - sum);
            } 
            // If we are below the diagonal
            else 
            {
                for (size_t k = 0; k < j; k++) 
                {
                    sum += (L(i, k) * L(j, k));
                }
                L(i, j) = ((*this)(i, j) - sum) / L(j, j);
            }
        }
    }
    
    return L;
}

Matrix Matrix::forward_substitution(const Matrix& b) const
{
    // Assumes 'this' is a Lower Triangular matrix (like L from Cholesky)
    assert(this->row() == this->cols());
    assert(this->row() == b.row() && b.cols() == 1);
    
    size_t n = this->row();
    Matrix z(n, 1); // The output vector
    
    for (size_t i = 0; i < n; i++) 
    {
        double sum = 0.0;
        for (size_t j = 0; j < i; j++) 
        {
            sum += (*this)(i, j) * z(j, 0);
        }
        z(i, 0) = (b(i, 0) - sum) / (*this)(i, i);
    }
    
    return z;
}
Matrix Matrix::backward_substitution(const Matrix& b) const
{
    // Assumes 'this' is an Upper Triangular matrix (like L^T from Cholesky)
    assert(this->row() == this->cols());
    assert(this->row() == b.row() && b.cols() == 1);
    
    size_t n = this->row();
    Matrix x(n, 1); // The final output vector
    
    // We iterate backwards from bottom to top
    for (int i = n - 1; i >= 0; i--) 
    {
        double sum = 0.0;
        for (int j = i + 1; j < (int)n; j++) 
        {
            sum += (*this)(i, j) * x(j, 0);
        }
        x(i, 0) = (b(i, 0) - sum) / (*this)(i, i);
    }
    
    return x;
}

std::pair<Matrix, Matrix> Matrix::qr_decomposition() const 
{
    // Note: We are explicitly using Householder Reflections for better numerical stability
    // rather than the classical Gram-Schmidt process.
    
    size_t m = this->row();
    size_t n = this->cols();
    
    // Q starts as an m x m Identity matrix
    Matrix Q(m, m);
    for (size_t i = 0; i < m; i++) Q(i, i) = 1.0;
    
    // R starts as a copy of the original matrix X
    Matrix R = *this;
    
    for (size_t k = 0; k < n; k++) 
    {
        // 1. Find the norm of the k-th column of R (below the diagonal)
        double norm_x = 0.0;
        for (size_t i = k; i < m; i++) {
            norm_x += R(i, k) * R(i, k);
        }
        norm_x = std::sqrt(norm_x);
        
        if (norm_x < 1e-12) continue; // Column is already zeroed out
        
        // 2. Define alpha to avoid catastrophic cancellation
        double alpha = (R(k, k) > 0) ? -norm_x : norm_x;
        
        // 3. Create the reflection vector u
        std::vector<double> u(m - k);
        u[0] = R(k, k) - alpha;
        double norm_u_sq = u[0] * u[0];
        
        for (size_t i = k + 1; i < m; i++) {
            u[i - k] = R(i, k);
            norm_u_sq += u[i - k] * u[i - k];
        }
        
        if (norm_u_sq < 1e-12) continue;
        
        // 4. Apply Householder reflection to R: R = (I - 2uu^T) R
        for (size_t j = k; j < n; j++) 
        {
            double dot = 0.0;
            for (size_t i = k; i < m; i++) {
                dot += u[i - k] * R(i, j);
            }
            double tau = 2.0 * dot / norm_u_sq;
            for (size_t i = k; i < m; i++) {
                R(i, j) -= tau * u[i - k];
            }
        }
        
        // 5. Apply Householder reflection to Q: Q = Q (I - 2uu^T)
        for (size_t i = 0; i < m; i++) 
        {
            double dot = 0.0;
            for (size_t j = k; j < m; j++) {
                dot += Q(i, j) * u[j - k];
            }
            double tau = 2.0 * dot / norm_u_sq;
            for (size_t j = k; j < m; j++) {
                Q(i, j) -= tau * u[j - k];
            }
        }
    }
    
    // Clean up floating point artifacts in the lower triangle of R
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < i && j < n; j++) {
            R(i, j) = 0.0;
        }
    }
    
    return std::make_pair(Q, R);
}
