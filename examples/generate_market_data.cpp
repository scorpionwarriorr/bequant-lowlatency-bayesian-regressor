#include <iostream>
#include <fstream>
#include <random>

int main() {
    std::ofstream out("market_data.csv");
    out << "Spread,Volume,Imbalance,Target_Return\n";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> spread_dist(0.1, 2.0);
    std::uniform_real_distribution<> vol_dist(10.0, 100.0);
    std::uniform_real_distribution<> imb_dist(-1.0, 1.0);
    std::normal_distribution<> noise_dist(0, 0.5);
    
    for (int i = 0; i < 5000; i++) {
        double spread = spread_dist(gen);
        double volume = vol_dist(gen);
        double imbalance = imb_dist(gen);
        double noise = noise_dist(gen);
        
        double y;
        if (i < 2500) {
            // Regime 1: Normal Market Conditions
            // True Betas: [5.0, 0.5, -1.2, 0.8]
            y = 5.0 + (0.5 * spread) + (-1.2 * volume) + (0.8 * imbalance) + noise;
        } else {
            // Regime 2: Market Crash (Shift in behavior)
            // True Betas: [2.0, -0.8, 0.5, -0.2]
            y = 2.0 + (-0.8 * spread) + (0.5 * volume) + (-0.2 * imbalance) + noise;
        }
        
        out << spread << "," << volume << "," << imbalance << "," << y << "\n";
    }
    
    std::cout << "Synthetic market data generated successfully (market_data.csv)!" << std::endl;
    return 0;
}
