import matplotlib.pyplot as plt
import csv
import os

def plot_results():
    ticks = []
    b0, b1, b2, b3 = [], [], [], []

    with open('results.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            ticks.append(int(row[0]))
            b0.append(float(row[1]))
            b1.append(float(row[2]))
            b2.append(float(row[3]))
            b3.append(float(row[4]))

    plt.figure(figsize=(12, 7))
    
    # Plot our RLS weights
    plt.plot(ticks, b0, label='Beta 0 (Intercept)')
    plt.plot(ticks, b1, label='Beta 1 (Spread)')
    plt.plot(ticks, b2, label='Beta 2 (Volume)')
    plt.plot(ticks, b3, label='Beta 3 (Imbalance)')

    # Draw a line where the market crashed
    plt.axvline(x=2500, color='r', linestyle='--', linewidth=2, label='Regime Shift (Market Crash)')

    plt.title('RLS Adaptation to Market Regime Shift', fontsize=16)
    plt.xlabel('Market Ticks', fontsize=12)
    plt.ylabel('Beta Weight', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), 'rls_adaptation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    plot_results()
