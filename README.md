# Adaptive Regression Engine

A custom C++ implementation of linear regression algorithms built from scratch to explore numerical linear algebra, online learning, and systems performance tradeoffs. This project implements both Batch Ordinary Least Squares (OLS) and Recursive Least Squares (RLS) to evaluate their computational complexities and latency profiles.

The engine has been benchmarked using Google Benchmark, and its numerical accuracy has been validated against the Eigen library.

## Motivation
In quantitative engineering, third-party libraries (like Eigen or LAPACK) are powerful but abstract away the underlying mathematics and memory management. To understand the systems-level tradeoffs of regression models, this project was built to implement covariance matrix construction and matrix decompositions (Cholesky, QR) from first principles. The goal is to analyze the latency characteristics of batch training versus recursive updates before introducing performance optimizations like SIMD or zero-allocation buffers.

## Architecture & Algorithms

### 1. Custom Matrix Library
The foundation is a scratch-built `Matrix` class handling basic operations and numerical decompositions:
*   **Cholesky Decomposition ($LL^T$)**: Used as the primary solver for symmetric positive-definite systems.
*   **QR Decomposition**: Implemented using Householder reflections for numerical stability on ill-conditioned datasets.
*   **Forward/Backward Substitution**: Solves triangular systems post-decomposition.

### 2. Regression Models
*   **Batch OLS (Inverse)**: The naive baseline implementation.
*   **Batch OLS (Cholesky)**: A more efficient batch solver for non-collinear datasets.
*   **Batch OLS (QR)**: Mathematically robust for collinear datasets, though computationally heavier.
*   **Ridge Regression (L2)**: Integrates L2 regularization into the Cholesky solver.
*   **Recursive Least Squares (RLS)**: An online learning engine that uses the Sherman-Morrison formula to recursively update model weights on incoming data points, bypassing the need for full covariance matrix inversion.

## Streaming Simulator & Adaptation Experiment
To evaluate the RLS implementation, the repository includes a custom data simulator (`simulator.cpp`).
*   **The Experiment**: The simulator generates a synthetic streaming dataset where the underlying data distribution abruptly shifts halfway through the stream.
*   **The Result**: The RLS engine adapts to changing data distributions through recursive updates without re-training on historical batch data.

## Benchmark Methodology
Performance was measured using **Google Benchmark** on a single thread (`g++ -O3`).
*   **Batch Regression**: Benchmarked by training a model on a full $5000 \times 100$ matrix.
*   **Streaming Update**: Benchmarked by updating an existing model with a single $1 \times 100$ data point.

### Benchmark Results
*See `benchmark_summary.md` and `benchmark_results.txt` for detailed outputs.*

| Algorithm | Method | Latency | 
| :--- | :--- | :--- | 
| **Batch OLS (Inverse)** | `(X^T X)^-1 X^T y` | **69.6 ms** |
| **Batch OLS (Cholesky)** | `LL^T` Decomposition | **37.3 ms** | 
| **Batch Ridge (L2)** | Cholesky with `+ \lambda I` | **36.9 ms** | 
| **Online RLS (Update)** | Sherman-Morrison | **575 µs / update** | 

*Note: RLS avoids the full batch recalculation (37ms) by processing single updates (575µs).*

## Eigen Correctness Validation
To verify numerical accuracy, the output of the custom Cholesky solver was validated against Eigen's `LLT` solver using an identical randomized $5000 \times 10$ dataset.

*   **Maximum Absolute Error:** $5.03 \times 10^{-17}$
*   **Mean Squared Error (MSE):** $5.22 \times 10^{-34}$

The custom implementation matches Eigen up to floating-point precision constraints. *(See `eigen_validation_results.txt`)*.

## Repository Structure
```text
Adaptive-Regression-Engine/
├── README.md
├── .gitignore
├── include/                # Header definitions
├── src/                    # Custom implementation files (Matrix, OLS, RLS)
├── benchmarks/             # Google Benchmark suite and Eigen validation scripts
├── scripts/                # Python visualization and data generation scripts
├── data/                   # Generated market data and regression results (ignored by git)
├── docs/                   # Benchmark results, tables, and PNG charts
├── examples/               # Standalone C++ simulators and test scripts
└── third_party/            # External dependencies (Eigen)
```

## Build Instructions
This project requires `g++` and `cmake` (if rebuilding Google Benchmark).

To compile the Eigen validation script:
```bash
g++ -O3 -I third_party/eigen -I include benchmarks/eigen_validation.cpp src/Matrix.cpp src/OLS.cpp -o eigen_validation
```

## Future Work (Version 2)
Version 1 establishes a mathematically correct and benchmarked baseline. Future iterations will focus on systems-level performance improvements:
1.  **Zero-Allocation Matrix Design:** The current 575µs latency in RLS is heavily bottlenecked by heap allocations (`new double[]`). Moving to statically-sized, pass-by-reference stack buffers will reduce allocation overhead.
2.  **SIMD Vectorization:** Manually vectorizing matrix multiplication loops.
3.  **Algorithmic Improvements:** Implementing a "Thin QR" decomposition to optimize overdetermined systems, and extending RLS into a Kalman Filter.