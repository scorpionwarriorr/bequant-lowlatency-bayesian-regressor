#include "OLS.hpp"
#include <cassert>

void OLS::fit_inverse(const Matrix& X, const Matrix& y)
{
    assert(X.row()==y.row() && y.cols()==1);
    
    Matrix XT = X.transpose();
    Matrix A = XT.multiply(X);
    
    Matrix A_inv = A.inverse();
    Matrix temp = A_inv.multiply(XT);
    this->beta = temp.multiply(y);
    
    fitted = true;
}

void OLS::fit_cholesky(const Matrix& X, const Matrix& y)
{
    assert(X.row()==y.row() && y.cols()==1);
    
    Matrix XT = X.transpose();
    Matrix A = XT.multiply(X);
    Matrix b = XT.multiply(y);
    
    Matrix L = A.cholesky();
    Matrix LT = L.transpose();
    
    Matrix z = L.forward_substitution(b);
    this->beta = LT.backward_substitution(z);
    
    fitted = true;
}

void OLS::fit_qr(const Matrix& X, const Matrix& y)
{
    assert(X.row()==y.row() && y.cols()==1);
    
    // Decompose X directly into Q * R
    std::pair<Matrix, Matrix> qr = X.qr_decomposition();
    Matrix Q = qr.first;
    Matrix R = qr.second;
    
    // Calculate Q^T * y
    Matrix QT = Q.transpose();
    Matrix z = QT.multiply(y);
    
    // We only need the top 'p' rows of z and R, where p is the number of features.
    size_t p = X.cols();
    Matrix R_top = R.submatrix(0, p, 0, p);
    Matrix z_top = z.submatrix(0, p, 0, 1);
    
    // Solve R_top * beta = z_top using Backward Substitution
    this->beta = R_top.backward_substitution(z_top);
    
    fitted = true;
}

void OLS::fit_ridge(const Matrix& X, const Matrix& y, double lambda)
{
    assert(X.row()==y.row() && y.cols()==1);
    
    Matrix XT = X.transpose();
    Matrix A = XT.multiply(X);
    
    // Apply Ridge Regularization (add lambda to the diagonal of X^T * X)
    for (size_t i = 0; i < A.row(); i++) 
    {
        A(i, i) += lambda;
    }
    
    Matrix b = XT.multiply(y);
    
    // Because we added lambda, 'A' is mathematically guaranteed to be 
    // strictly Symmetric Positive Definite, so Cholesky is perfectly safe!
    Matrix L = A.cholesky();
    Matrix LT = L.transpose();
    
    Matrix z = L.forward_substitution(b);
    this->beta = LT.backward_substitution(z);
    
    fitted = true;
}

Matrix OLS::predict(const Matrix& X) const
{
    assert(fitted);
    Matrix T = X.multiply(this->beta);
    return T;
}

const Matrix& OLS::coefficients() const
{
    // just for access and see it, read only cause const
    return this->beta;
}