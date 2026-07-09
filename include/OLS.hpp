#pragma once
#include<Matrix.hpp>


class OLS
{
    private:
        Matrix beta;
        bool fitted;
    public:
        OLS(): beta(0,0),fitted(false){}
        
        // Fit model using classical explicit inverse (Slow, Unstable)
        void fit_inverse(const Matrix& X, const Matrix& y);
        
        // Fit model using Cholesky Decomposition (Fast, Needs well-conditioned data)
        void fit_cholesky(const Matrix& X, const Matrix& y);
        
        // Fit model using QR Decomposition (Fast, Highly Stable)
        void fit_qr(const Matrix& X, const Matrix& y);
        
        // Fit model using Ridge Regularization (Adds L2 penalty lambda to prevent overfitting)
        void fit_ridge(const Matrix& X, const Matrix& y, double lambda);

        // Predict outputs
    // X : (n x p)
    // returns (n x 1)
        Matrix predict(const Matrix& X) const;

        // Access coefficients
        const Matrix& coefficients() const;
};