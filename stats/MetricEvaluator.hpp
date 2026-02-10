/**
 * @file MetricEvaluator.hpp
 * @brief High-level metric evaluation utilities (SRP, DIP)
 * @details Provides convenient evaluation of multiple metrics
 */

#ifndef METRIC_EVALUATOR_HPP
#define METRIC_EVALUATOR_HPP

#include "IMetric.hpp"
#include "MetricFactory.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <iomanip>

namespace metrics {

/**
 * @brief Result container for metric evaluation
 */
struct MetricResult {
    std::string name;
    double value;
    bool higherIsBetter;

    void print() const {
        std::cout << std::setw(20) << std::left << name << ": "
                  << std::fixed << std::setprecision(4) << value
                  << (higherIsBetter ? " (higher=better)" : " (lower=better)")
                  << std::endl;
    }
};

/**
 * @brief Evaluator for classification metrics (SRP)
 * @details Depends on IMetric abstraction (DIP)
 */
class ClassificationEvaluator {
    std::vector<std::unique_ptr<IMetric<int>>> metrics_;

public:
    /**
     * @brief Add a metric to evaluate
     */
    void addMetric(std::unique_ptr<IMetric<int>> metric) {
        metrics_.push_back(std::move(metric));
    }

    /**
     * @brief Add a metric by name using factory
     */
    void addMetric(const std::string& name) {
        metrics_.push_back(getMetricFactory().createClassificationMetric(name));
    }

    /**
     * @brief Add all default classification metrics
     */
    void addAllDefaultMetrics() {
        addMetric("accuracy");
        addMetric("precision");
        addMetric("recall");
        addMetric("specificity");
        addMetric("f1");
        addMetric("mcc");
        addMetric("kappa");
    }

    /**
     * @brief Evaluate all added metrics
     */
    std::vector<MetricResult> evaluate(const std::vector<int>& y_true,
                                         const std::vector<int>& y_pred) const {
        std::vector<MetricResult> results;

        for (const auto& metric : metrics_) {
            MetricResult result;
            result.name = metric->name();
            result.value = metric->calculate(y_true, y_pred);
            result.higherIsBetter = metric->higherIsBetter();
            results.push_back(result);
        }

        return results;
    }

    /**
     * @brief Evaluate and print results
     */
    void evaluateAndPrint(const std::vector<int>& y_true,
                           const std::vector<int>& y_pred) const {
        std::cout << "\n=== Classification Metrics ===\n";
        for (const auto& result : evaluate(y_true, y_pred)) {
            result.print();
        }
    }
};

/**
 * @brief Evaluator for regression metrics (SRP)
 * @details Depends on IMetric abstraction (DIP)
 */
class RegressionEvaluator {
    std::vector<std::unique_ptr<IMetric<double>>> metrics_;

public:
    /**
     * @brief Add a metric to evaluate
     */
    void addMetric(std::unique_ptr<IMetric<double>> metric) {
        metrics_.push_back(std::move(metric));
    }

    /**
     * @brief Add a metric by name using factory
     */
    void addMetric(const std::string& name) {
        metrics_.push_back(getMetricFactory().createRegressionMetric(name));
    }

    /**
     * @brief Add all default regression metrics
     */
    void addAllDefaultMetrics() {
        addMetric("mse");
        addMetric("rmse");
        addMetric("mae");
        addMetric("medae");
        addMetric("r2");
        addMetric("mape");
        addMetric("smape");
        addMetric("max_error");
        addMetric("explained_variance");
        addMetric("huber");
    }

    /**
     * @brief Evaluate all added metrics
     */
    std::vector<MetricResult> evaluate(const std::vector<double>& y_true,
                                         const std::vector<double>& y_pred) const {
        std::vector<MetricResult> results;

        for (const auto& metric : metrics_) {
            MetricResult result;
            result.name = metric->name();
            result.value = metric->calculate(y_true, y_pred);
            result.higherIsBetter = metric->higherIsBetter();
            results.push_back(result);
        }

        return results;
    }

    /**
     * @brief Evaluate and print results
     */
    void evaluateAndPrint(const std::vector<double>& y_true,
                           const std::vector<double>& y_pred) const {
        std::cout << "\n=== Regression Metrics ===\n";
        for (const auto& result : evaluate(y_true, y_pred)) {
            result.print();
        }
    }
};

/**
 * @brief Probabilistic metric evaluator (SRP)
 */
class ProbabilisticEvaluator {
    std::vector<std::unique_ptr<IProbabilisticMetric<double>>> metrics_;

public:
    void addMetric(std::unique_ptr<IProbabilisticMetric<double>> metric) {
        metrics_.push_back(std::move(metric));
    }

    void addAllDefaultMetrics() {
        metrics_.push_back(std::make_unique<BinaryCrossEntropy>());
        metrics_.push_back(std::make_unique<RocAucScore>());
        metrics_.push_back(std::make_unique<AveragePrecisionScore>());
    }

    std::map<std::string, double> evaluate(const std::vector<int>& y_true,
                                             const std::vector<double>& y_prob) const {
        std::map<std::string, double> results;

        for (const auto& metric : metrics_) {
            results[metric->name()] = metric->calculate(y_true, y_prob);
        }

        return results;
    }

    void evaluateAndPrint(const std::vector<int>& y_true,
                           const std::vector<double>& y_prob) const {
        std::cout << "\n=== Probabilistic Metrics ===\n";
        for (const auto& [name, value] : evaluate(y_true, y_prob)) {
            std::cout << std::setw(20) << std::left << name << ": "
                      << std::fixed << std::setprecision(4) << value << std::endl;
        }
    }
};

} // namespace metrics

#endif // METRIC_EVALUATOR_HPP
