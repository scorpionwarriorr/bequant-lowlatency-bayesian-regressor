#pragma once
#include "Matrix.hpp"

class RLS
{
    private:
        Matrix beta;       // The weight coefficients (p x 1)
        Matrix P;          // The inverse covariance matrix (p x p)
        double lambda;     // The forgetting factor (usually 0.99 to 1.0)
        bool initialized;

    public:
        // Constructor defaults to lambda = 0.99 for slight forgetting of old data
        RLS(double lambda = 0.99);

        // Initializes the dimensions before streaming starts
        // initial_P acts as a confidence parameter (usually set high, e.g., 1000.0)
        void init(size_t num_features, double initial_P = 1000.0);

        // Updates the model mathematically with a SINGLE new row of data (1 x p)
        // and its corresponding target value
        void update(const Matrix& x, double y);

        // Predicts the target for a single row of data (1 x p)
        double predict(const Matrix& x) const;

        // Access the current weights
        const Matrix& coefficients() const;
};
