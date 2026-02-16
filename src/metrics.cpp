#include "metrics.hpp"
#include <cassert>


double metrics::rss(const std::vector<double>& y,
                    const std::vector<double>& y_pred) {
    double sum = 0.0;
    const size_t n = y.size();

    for (size_t i = 0; i < n; ++i) {
        const double r = y[i] - y_pred[i];
        sum += r * r;
    }
    return sum;
}

double metrics::tss(const std::vector<double>& y) {
    const size_t n = y.size();

    double mean = 0.0;
    for (double v : y) mean += v;
    mean /= n;

    double sum = 0.0;
    for (double v : y) {
        const double d = v - mean;
        sum += d * d;
    }
    return sum;
}

double metrics::r_squared(const std::vector<double>& y,
                          const std::vector<double>& y_pred) {
    const double rss_val = rss(y, y_pred);
    const double tss_val = tss(y);
    return 1.0 - (rss_val / tss_val);
}
