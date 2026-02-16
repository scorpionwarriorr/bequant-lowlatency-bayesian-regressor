#pragma once
#include <vector>

namespace metrics {

double rss(const std::vector<double>& y,
           const std::vector<double>& y_pred);

double tss(const std::vector<double>& y);

double r_squared(const std::vector<double>& y,
                 const std::vector<double>& y_pred);

}
