#include <iostream>
#include <chrono>
#include <random>
#include "include/Matrix.hpp"
#include "include/OLS.hpp"

// Utility to generate a random matrix
Matrix generate_random_matrix(size_t rows, size_t cols) {
    Matrix M(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            M(i, j) = d(gen);
        }
    }
    return M;
}

int main() {
    std::cout << "Starting Benchmark: OLS with Gauss-Jordan Inverse vs Cholesky Decomposition" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    // Simulate 5000 rows of market data with 100 features
    size_t num_samples = 5000;
    size_t num_features = 100;

    std::cout << "Generating dataset... (" << num_samples << " rows, " << num_features << " features)" << std::endl;
    Matrix X = generate_random_matrix(num_samples, num_features);
    Matrix y = generate_random_matrix(num_samples, 1);

    // Precompute A = X^T * X and b = X^T * y for fair mathematical comparison
    Matrix XT = X.transpose();
    Matrix A = XT.multiply(X);
    Matrix b = XT.multiply(y);
    
    // We modify the diagonal slightly to ensure it is strongly Symmetric Positive Definite for Cholesky
    for (size_t i = 0; i < A.row(); i++) {
        A(i, i) += 10.0;
    }

    std::cout << "Running Benchmark..." << std::endl;

    // 1. Gauss-Jordan Inverse Method (The old way)
    auto start_inverse = std::chrono::high_resolution_clock::now();
    
    Matrix A_inv = A.inverse();
    Matrix beta_inverse = A_inv.multiply(b);
    
    auto end_inverse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inverse_ms = end_inverse - start_inverse;

    // 2. Cholesky Decomposition Method (The new way)
    auto start_cholesky = std::chrono::high_resolution_clock::now();
    
    Matrix L = A.cholesky();
    Matrix LT = L.transpose();
    Matrix z = L.forward_substitution(b);
    Matrix beta_cholesky = LT.backward_substitution(z);
    
    auto end_cholesky = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cholesky_ms = end_cholesky - start_cholesky;

    // Output Results
    std::cout << "\nResults:" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Explicit Inverse Time : " << inverse_ms.count() << " ms" << std::endl;
    std::cout << "Cholesky Decomp Time  : " << cholesky_ms.count() << " ms" << std::endl;
    std::cout << "Speedup               : " << inverse_ms.count() / cholesky_ms.count() << "x faster" << std::endl;
    std::cout << "---------------------------------" << std::endl;

    std::cout << "\nVerifying Math (First 3 Beta Coefficients):" << std::endl;
    std::cout << "Inverse  : [" << beta_inverse(0,0) << ", " << beta_inverse(1,0) << ", " << beta_inverse(2,0) << "]" << std::endl;
    std::cout << "Cholesky : [" << beta_cholesky(0,0) << ", " << beta_cholesky(1,0) << ", " << beta_cholesky(2,0) << "]" << std::endl;

    return 0;
}
