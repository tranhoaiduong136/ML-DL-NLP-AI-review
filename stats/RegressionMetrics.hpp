/**
 * @file RegressionMetrics.hpp
 * @brief Regression metrics following SOLID principles
 * @details Single Responsibility: Each class handles one specific metric
 */

#ifndef REGRESSION_METRICS_HPP
#define REGRESSION_METRICS_HPP

#include "IMetric.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace metrics {

/**
 * @brief Mean Squared Error (SRP)
 * MSE = 1/N * sum((y_true - y_pred)^2)
 */
class MeanSquaredError : public IMetric<double> {
public:
    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            double diff = y_true[i] - y_pred[i];
            sum += diff * diff;
        }
        return sum / y_true.size();
    }

    std::string name() const override { return "MSE"; }
    bool higherIsBetter() const override { return false; }
};

/**
 * @brief Root Mean Squared Error (SRP)
 * RMSE = sqrt(MSE)
 */
class RootMeanSquaredError : public IMetric<double> {
    MeanSquaredError mse_;
public:
    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        return std::sqrt(mse_.calculate(y_true, y_pred));
    }

    std::string name() const override { return "RMSE"; }
    bool higherIsBetter() const override { return false; }
};

/**
 * @brief Mean Absolute Error (SRP)
 * MAE = 1/N * sum(|y_true - y_pred|)
 */
class MeanAbsoluteError : public IMetric<double> {
public:
    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            sum += std::abs(y_true[i] - y_pred[i]);
        }
        return sum / y_true.size();
    }

    std::string name() const override { return "MAE"; }
    bool higherIsBetter() const override { return false; }
};

/**
 * @brief Median Absolute Error (SRP)
 * MedAE = median(|y_true - y_pred|)
 */
class MedianAbsoluteError : public IMetric<double> {
public:
    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        std::vector<double> errors(y_true.size());
        for (size_t i = 0; i < y_true.size(); ++i) {
            errors[i] = std::abs(y_true[i] - y_pred[i]);
        }

        std::sort(errors.begin(), errors.end());
        size_t mid = errors.size() / 2;

        if (errors.size() % 2 == 0) {
            return (errors[mid - 1] + errors[mid]) / 2.0;
        }
        return errors[mid];
    }

    std::string name() const override { return "MedAE"; }
    bool higherIsBetter() const override { return false; }
};

/**
 * @brief R-Squared (Coefficient of Determination) (SRP)
 * R^2 = 1 - SS_res / SS_tot
 */
class RSquared : public IMetric<double> {
public:
    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        double mean = std::accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();

        double ssRes = 0.0, ssTot = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            ssRes += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
            ssTot += (y_true[i] - mean) * (y_true[i] - mean);
        }

        if (ssTot == 0.0) return 0.0;
        return 1.0 - (ssRes / ssTot);
    }

    std::string name() const override { return "R-Squared"; }
    bool higherIsBetter() const override { return true; }
};

/**
 * @brief Adjusted R-Squared (OCP - configurable via n_features)
 * Adj_R^2 = 1 - (1 - R^2) * (n - 1) / (n - p - 1)
 */
class AdjustedRSquared : public IMetric<double> {
    int nFeatures_;
    RSquared r2_;
public:
    explicit AdjustedRSquared(int nFeatures) : nFeatures_(nFeatures) {}

    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        double r2 = r2_.calculate(y_true, y_pred);
        int n = static_cast<int>(y_true.size());

        if (n - nFeatures_ - 1 <= 0) return r2;
        return 1.0 - (1.0 - r2) * (n - 1) / (n - nFeatures_ - 1);
    }

    std::string name() const override { return "AdjustedR-Squared"; }
    bool higherIsBetter() const override { return true; }
};

/**
 * @brief Mean Absolute Percentage Error (SRP)
 * MAPE = 100/N * sum(|y_true - y_pred| / |y_true|)
 */
class MeanAbsolutePercentageError : public IMetric<double> {
    double epsilon_;
public:
    explicit MeanAbsolutePercentageError(double epsilon = 1e-10) : epsilon_(epsilon) {}

    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            sum += std::abs((y_true[i] - y_pred[i]) / (std::abs(y_true[i]) + epsilon_));
        }
        return 100.0 * sum / y_true.size();
    }

    std::string name() const override { return "MAPE"; }
    bool higherIsBetter() const override { return false; }
};

/**
 * @brief Symmetric Mean Absolute Percentage Error (SRP)
 * sMAPE = 100/N * sum(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))
 */
class SymmetricMAPE : public IMetric<double> {
public:
    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            double denom = (std::abs(y_true[i]) + std::abs(y_pred[i])) / 2.0;
            if (denom != 0.0) {
                sum += std::abs(y_true[i] - y_pred[i]) / denom;
            }
        }
        return 100.0 * sum / y_true.size();
    }

    std::string name() const override { return "sMAPE"; }
    bool higherIsBetter() const override { return false; }
};

/**
 * @brief Mean Squared Logarithmic Error (SRP)
 * MSLE = 1/N * sum((log(1+y_true) - log(1+y_pred))^2)
 */
class MeanSquaredLogError : public IMetric<double> {
public:
    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            double diff = std::log1p(y_true[i]) - std::log1p(y_pred[i]);
            sum += diff * diff;
        }
        return sum / y_true.size();
    }

    std::string name() const override { return "MSLE"; }
    bool higherIsBetter() const override { return false; }
};

/**
 * @brief Root Mean Squared Logarithmic Error (SRP)
 */
class RootMeanSquaredLogError : public IMetric<double> {
    MeanSquaredLogError msle_;
public:
    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        return std::sqrt(msle_.calculate(y_true, y_pred));
    }

    std::string name() const override { return "RMSLE"; }
    bool higherIsBetter() const override { return false; }
};

/**
 * @brief Maximum Error (SRP)
 * MaxError = max(|y_true - y_pred|)
 */
class MaxError : public IMetric<double> {
public:
    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        double maxErr = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            maxErr = std::max(maxErr, std::abs(y_true[i] - y_pred[i]));
        }
        return maxErr;
    }

    std::string name() const override { return "MaxError"; }
    bool higherIsBetter() const override { return false; }
};

/**
 * @brief Explained Variance Score (SRP)
 * EVS = 1 - Var(y_true - y_pred) / Var(y_true)
 */
class ExplainedVarianceScore : public IMetric<double> {
public:
    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        size_t n = y_true.size();

        // Calculate residuals and their variance
        std::vector<double> residuals(n);
        double residMean = 0.0;
        for (size_t i = 0; i < n; ++i) {
            residuals[i] = y_true[i] - y_pred[i];
            residMean += residuals[i];
        }
        residMean /= n;

        double residVar = 0.0;
        for (size_t i = 0; i < n; ++i) {
            residVar += (residuals[i] - residMean) * (residuals[i] - residMean);
        }
        residVar /= n;

        // Calculate variance of y_true
        double trueMean = std::accumulate(y_true.begin(), y_true.end(), 0.0) / n;
        double trueVar = 0.0;
        for (size_t i = 0; i < n; ++i) {
            trueVar += (y_true[i] - trueMean) * (y_true[i] - trueMean);
        }
        trueVar /= n;

        if (trueVar == 0.0) return 0.0;
        return 1.0 - residVar / trueVar;
    }

    std::string name() const override { return "ExplainedVariance"; }
    bool higherIsBetter() const override { return true; }
};

/**
 * @brief Huber Loss (OCP - configurable via delta parameter)
 * Combines MSE and MAE properties
 */
class HuberLoss : public IMetric<double> {
    double delta_;
public:
    explicit HuberLoss(double delta = 1.0) : delta_(delta) {}

    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            double absError = std::abs(y_true[i] - y_pred[i]);

            if (absError <= delta_) {
                sum += 0.5 * absError * absError;
            } else {
                sum += delta_ * absError - 0.5 * delta_ * delta_;
            }
        }
        return sum / y_true.size();
    }

    std::string name() const override { return "HuberLoss"; }
    bool higherIsBetter() const override { return false; }
};

/**
 * @brief Quantile Loss / Pinball Loss (OCP - configurable via quantile)
 */
class QuantileLoss : public IMetric<double> {
    double quantile_;
public:
    explicit QuantileLoss(double quantile = 0.5) : quantile_(quantile) {}

    double calculate(const std::vector<double>& y_true,
                     const std::vector<double>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            double error = y_true[i] - y_pred[i];
            if (error >= 0) {
                sum += quantile_ * error;
            } else {
                sum += (quantile_ - 1.0) * error;
            }
        }
        return sum / y_true.size();
    }

    std::string name() const override { return "QuantileLoss"; }
    bool higherIsBetter() const override { return false; }
};

} // namespace metrics

#endif // REGRESSION_METRICS_HPP
