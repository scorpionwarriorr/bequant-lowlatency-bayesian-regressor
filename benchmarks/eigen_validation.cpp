#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Our custom math engine
#include "../include/Matrix.hpp"
#include "../include/OLS.hpp"

// Eigen library (Industry Standard)
#include <Eigen/Dense>

using namespace std;

int main() {
    size_t n = 5000;
    size_t p = 10; // 10 features

    std::cout << "==========================================\n";
    std::cout << "   Eigen Validation Suite - Version 1\n";
    std::cout << "==========================================\n";
    std::cout << "Generating exact identical dataset (" << n << "x" << p << ")...\n";

    // Fixed seed for deterministic comparison
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    // Initialize our Custom Matrix and Eigen's Matrix
    Matrix X(n, p);
    Matrix y(n, 1);
    
    Eigen::MatrixXd X_eigen(n, p);
    Eigen::VectorXd y_eigen(n);

    for(size_t i=0; i<n; i++) {
        for(size_t j=0; j<p; j++) {
            double val = dist(gen);
            X(i, j) = val;
            X_eigen(i, j) = val;
        }
        double val_y = dist(gen);
        y(i, 0) = val_y;
        y_eigen(i) = val_y;
    }

    std::cout << "Solving using Custom Cholesky Decomposition...\n";
    OLS custom_model;
    custom_model.fit_cholesky(X, y);
    Matrix custom_beta = custom_model.coefficients();

    std::cout << "Solving using Eigen LLT (Cholesky) Solver...\n";
    // Formula: beta = (X^T * X)^-1 * (X^T * y)
    Eigen::MatrixXd XT = X_eigen.transpose();
    Eigen::VectorXd eigen_beta = (XT * X_eigen).llt().solve(XT * y_eigen);

    std::cout << "\n--- Coefficient Comparison ---\n";
    double mse = 0.0;
    for(size_t i=0; i<p; i++) {
        double diff = custom_beta(i, 0) - eigen_beta(i);
        mse += diff * diff;
        std::cout << "Feature " << i 
                  << " | Custom: " << custom_beta(i, 0) 
                  << "\t| Eigen: " << eigen_beta(i) 
                  << "\t| Delta: " << std::abs(diff) << "\n";
    }
    mse /= p;
    
    std::cout << "\n==========================================\n";
    std::cout << "Mean Squared Error (MSE): " << mse << "\n";
    
    if (mse < 1e-10) {
        std::cout << "[SUCCESS] VALIDATION PASSED!\n";
        std::cout << "Your custom engine computes the exact mathematically correct answer.\n";
    } else {
        std::cout << "[ERROR] VALIDATION FAILED! Numerical divergence detected.\n";
    }
    std::cout << "==========================================\n";

    return 0;
}
