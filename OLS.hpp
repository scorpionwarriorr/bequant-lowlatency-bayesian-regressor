#pragma once
#include<Matrix.hpp>


class OLS
{
    private:
        Matrix beta;
        bool fitted;
    public:
        OLS(): beta(0,0),fitted(false){}
        
        // Fit model: y = X beta + eps
    // X : (n x p)
    // y : (n x 1)
        void fit(const Matrix& X, const Matrix& y);

        // Predict outputs
    // X : (n x p)
    // returns (n x 1)
        Matrix predict(const Matrix& X) const;

        // Access coefficients
        const Matrix& coefficients() const;
};