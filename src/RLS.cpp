#include "RLS.hpp"
#include <cassert>

RLS::RLS(double lambda) : beta(0, 0), P(0, 0), lambda(lambda), initialized(false) {}

void RLS::init(size_t num_features, double initial_P)
{
    // Initialize beta to zeros (num_features x 1)
    beta = Matrix(num_features, 1);
    
    // Initialize P to a diagonal matrix with initial_P on the diagonal
    P = Matrix(num_features, num_features);
    for (size_t i = 0; i < num_features; i++) 
    {
        P(i, i) = initial_P;
    }
    
    initialized = true;
}

void RLS::update(const Matrix& x, double y)
{
    assert(initialized);
    assert(x.row() == 1 && x.cols() == beta.row()); // x must be a single row (1 x p)
    
    // STEP 1: Calculate the Gain Vector (k)
    // Formula: k = (P * x^T) / (lambda + x * P * x^T)
    Matrix xT = x.transpose();
    Matrix P_xT = P.multiply(xT);     // (p x p) * (p x 1) = (p x 1)
    Matrix x_P_xT = x.multiply(P_xT); // (1 x p) * (p x 1) = (1 x 1)
    
    double denominator = lambda + x_P_xT(0, 0);
    
    Matrix k(beta.row(), 1);
    for (size_t i = 0; i < k.row(); i++) 
    {
        k(i, 0) = P_xT(i, 0) / denominator;
    }
    
    // STEP 2: Calculate the Prediction Error (e)
    // Formula: e = y - (x * beta)
    Matrix prediction = x.multiply(beta);
    double error = y - prediction(0, 0);
    
    // STEP 3: Update the Beta Weights
    // Formula: beta = beta + (k * e)
    for (size_t i = 0; i < beta.row(); i++) 
    {
        beta(i, 0) += k(i, 0) * error;
    }
    
    // STEP 4: Update the P Matrix (Inverse Covariance)
    // Formula: P = (P - (k * x * P)) / lambda
    Matrix k_x = k.multiply(x);       // (p x 1) * (1 x p) = (p x p)
    Matrix k_x_P = k_x.multiply(P);   // (p x p) * (p x p) = (p x p)
    
    for (size_t i = 0; i < P.row(); i++) 
    {
        for (size_t j = 0; j < P.cols(); j++) 
        {
            P(i, j) = (P(i, j) - k_x_P(i, j)) / lambda;
        }
    }
}

double RLS::predict(const Matrix& x) const
{
    assert(initialized);
    Matrix pred = x.multiply(beta);
    return pred(0, 0);
}

const Matrix& RLS::coefficients() const
{
    return beta;
}
