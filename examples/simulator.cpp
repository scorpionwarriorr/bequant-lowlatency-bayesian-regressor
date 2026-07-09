#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "include/Matrix.hpp"
#include "include/RLS.hpp"

int main() {
    std::cout << "Starting Streaming Simulator..." << std::endl;
    
    std::ifstream file("market_data.csv");
    if (!file.is_open()) {
        std::cerr << "Failed to open market_data.csv!" << std::endl;
        return 1;
    }
    
    std::ofstream out("results.csv");
    out << "Tick,Beta0,Beta1,Beta2,Beta3" << std::endl;
    
    std::string line;
    std::getline(file, line); // Skip the CSV header
    
    // Initialize RLS with lambda = 0.99 to allow forgetting old data
    RLS rls(0.99);
    // 4 features: Intercept, Spread, Volume, Imbalance
    rls.init(4, 1000.0); 
    
    int tick = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        
        double spread, volume, imbalance, y;
        
        // Parse CSV row
        std::getline(ss, val, ','); spread = std::stod(val);
        std::getline(ss, val, ','); volume = std::stod(val);
        std::getline(ss, val, ','); imbalance = std::stod(val);
        std::getline(ss, val, ','); y = std::stod(val);
        
        // Create input vector WITH intercept
        Matrix x(1, 4);
        x(0, 0) = 1.0; // The magic intercept column
        x(0, 1) = spread;
        x(0, 2) = volume;
        x(0, 3) = imbalance;
        
        // Stream the single tick into the model
        rls.update(x, y);
        
        // Log the weights for this tick to our results file
        const Matrix& b = rls.coefficients();
        out << tick << "," << b(0,0) << "," << b(1,0) << "," << b(2,0) << "," << b(3,0) << "\n";
        
        tick++;
    }
    
    std::cout << "Simulation complete. Processed " << tick << " market ticks." << std::endl;
    return 0;
}
