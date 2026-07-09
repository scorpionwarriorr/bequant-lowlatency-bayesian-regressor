## Google Benchmark Summary

The following benchmarks were compiled using `g++ -O3` and executed on a single thread. The goal of this benchmark is to measure the algorithmic complexity and latency of our custom linear algebra implementation.

### Batch Regression Latency
This test measures the time required to compute the model coefficients ($\beta$) from a complete batch dataset of **5,000 rows and 100 features**.

| Algorithm | Method | Latency | Notes |
| :--- | :--- | :--- | :--- |
| **OLS (Inverse)** | `(X^T X)^-1 X^T y` | **69.6 ms** | Explicit matrix inversion is computationally expensive and generally avoided. |
| **OLS (Cholesky)** | `LL^T` Decomposition | **37.3 ms** | ~2x faster than explicit inverse. This is the standard for non-collinear batch OLS. |
| **Ridge (L2)** | Cholesky with `+ \lambda I` | **36.9 ms** | Proves that adding L2 regularization incurs effectively zero latency overhead. |
| **OLS (QR)** | Full Householder QR | **2601 ms** | Extremely slow due to the explicit construction of the full $5000 \times 5000$ $Q$ matrix. |

### Streaming Regression Latency
This test measures the time required to update the model weights recursively when a **single new market tick** (1 row, 100 features) arrives.

| Algorithm | Method | Latency / Tick | Notes |
| :--- | :--- | :--- | :--- |
| **RLS (Update)** | Sherman-Morrison Formula | **575 µs** | Updates weights in microseconds. Avoiding full batch recalculation (37ms) yields a ~64x reduction in tick-to-trade latency. |

*Note: The RLS update latency (575 µs) is currently bottlenecked by heap allocations within the `Matrix` class. Achieving nanosecond latency requires refactoring the math engine to support zero-allocation pass-by-reference buffers.*
