import csv
import random

def generate_data():
    with open('market_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Spread', 'Volume', 'Imbalance', 'Target_Return'])
        
        for i in range(5000):
            spread = random.uniform(0.1, 2.0)
            volume = random.uniform(10.0, 100.0)
            imbalance = random.uniform(-1.0, 1.0)
            noise = random.gauss(0, 0.5)
            
            if i < 2500:
                # Regime 1: Normal Market Conditions
                # True Betas: [5.0, 0.5, -1.2, 0.8]
                y = 5.0 + (0.5 * spread) + (-1.2 * volume) + (0.8 * imbalance) + noise
            else:
                # Regime 2: Market Crash (Shift in behavior)
                # True Betas: [2.0, -0.8, 0.5, -0.2]
                y = 2.0 + (-0.8 * spread) + (0.5 * volume) + (-0.2 * imbalance) + noise
                
            writer.writerow([spread, volume, imbalance, y])
            
if __name__ == '__main__':
    generate_data()
    print("Synthetic market data generated successfully!")
