#include <iostream>
#include <random>
#include <iomanip>
#include "include/Matrix.hpp"
#include "include/RLS.hpp"

int main() {
    std::cout << "Starting RLS Convergence Test..." << std::endl;
    std::cout << "Target: y = 2x + 3 + noise (Beta should converge to [3.0, 2.0])\n" << std::endl;

    // Initialize RLS for 2 features (Intercept and 'x')
    // We use lambda = 1.0 (no forgetting) so it perfectly converges on the static line
    RLS rls(1.0); 
    rls.init(2, 1000.0); // 2 features, initial confidence (P) = 1000.0

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, 0.5); // Add a little noise
    std::uniform_real_distribution<> x_dist(0, 10);

    for (int i = 1; i <= 100; i++) {
        double x_val = x_dist(gen);
        
        // y = 2x + 3 + noise
        double y_val = 2.0 * x_val + 3.0 + noise(gen);

        // Create the input row [1.0, x_val]. Notice the 1.0 is the intercept trick!
        Matrix x_row(1, 2);
        x_row(0, 0) = 1.0;
        x_row(0, 1) = x_val;

        // Feed one sample at a time!
        rls.update(x_row, y_val);

        // Print progress every 10 steps to watch it learn
        if (i % 10 == 0 || i == 1) {
            const Matrix& beta = rls.coefficients();
            std::cout << "Step " << std::setw(3) << i 
                      << " | Beta: [" << std::fixed << std::setprecision(4) 
                      << beta(0, 0) << ", " << beta(1, 0) << "]" << std::endl;
        }
    }

    std::cout << "\nTest Complete. If Beta is close to [3.0000, 2.0000], RLS is working!" << std::endl;
    return 0;
}
