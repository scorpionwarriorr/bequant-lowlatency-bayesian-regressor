#include <benchmark/benchmark.h>
#include "../include/Matrix.hpp"
#include "../include/OLS.hpp"
#include "../include/RLS.hpp"
#include <random>

// Helper to generate reproducible random matrices for benchmarking
Matrix generate_random_matrix(size_t rows, size_t cols) {
    Matrix m(rows, cols);
    std::mt19937 gen(42); // Fixed seed for deterministic benchmarks
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m(i, j) = dist(gen);
        }
    }
    return m;
}

static void BM_OLS_Inverse(benchmark::State& state) {
    size_t rows = 5000;
    size_t cols = 100;
    Matrix X = generate_random_matrix(rows, cols);
    Matrix y = generate_random_matrix(rows, 1);
    
    for (auto _ : state) {
        OLS model;
        model.fit_inverse(X, y);
        benchmark::DoNotOptimize(model.coefficients());
    }
}
BENCHMARK(BM_OLS_Inverse)->Unit(benchmark::kMillisecond);

static void BM_OLS_Cholesky(benchmark::State& state) {
    size_t rows = 5000;
    size_t cols = 100;
    Matrix X = generate_random_matrix(rows, cols);
    // Make X well-conditioned so X^T X is SPD
    for(size_t i = 0; i < cols; i++) X(i, i) += 10.0;
    Matrix y = generate_random_matrix(rows, 1);
    
    for (auto _ : state) {
        OLS model;
        model.fit_cholesky(X, y);
        benchmark::DoNotOptimize(model.coefficients());
    }
}
BENCHMARK(BM_OLS_Cholesky)->Unit(benchmark::kMillisecond);

static void BM_OLS_QR(benchmark::State& state) {
    size_t rows = 5000;
    size_t cols = 100;
    Matrix X = generate_random_matrix(rows, cols);
    Matrix y = generate_random_matrix(rows, 1);
    
    for (auto _ : state) {
        OLS model;
        model.fit_qr(X, y);
        benchmark::DoNotOptimize(model.coefficients());
    }
}
BENCHMARK(BM_OLS_QR)->Unit(benchmark::kMillisecond);

static void BM_OLS_Ridge(benchmark::State& state) {
    size_t rows = 5000;
    size_t cols = 100;
    Matrix X = generate_random_matrix(rows, cols);
    Matrix y = generate_random_matrix(rows, 1);
    double lambda = 1.0;
    
    for (auto _ : state) {
        OLS model;
        model.fit_ridge(X, y, lambda);
        benchmark::DoNotOptimize(model.coefficients());
    }
}
BENCHMARK(BM_OLS_Ridge)->Unit(benchmark::kMillisecond);

static void BM_RLS_Update(benchmark::State& state) {
    size_t cols = 100;
    Matrix x = generate_random_matrix(1, cols); // Single row
    double y = 1.0;
    
    RLS model(0.99);
    model.init(cols, 1000.0);
    
    for (auto _ : state) {
        model.update(x, y);
        benchmark::DoNotOptimize(model.coefficients());
    }
}
// RLS updates one row, so we measure in microseconds/nanoseconds
BENCHMARK(BM_RLS_Update)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
