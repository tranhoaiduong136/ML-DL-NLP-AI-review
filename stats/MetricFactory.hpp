/**
 * @file MetricFactory.hpp
 * @brief Factory for creating metrics (Dependency Inversion Principle)
 * @details Clients depend on abstractions (IMetricFactory), not concrete implementations
 */

#ifndef METRIC_FACTORY_HPP
#define METRIC_FACTORY_HPP

#include "IMetric.hpp"
#include "ClassificationMetrics.hpp"
#include "RegressionMetrics.hpp"
#include <unordered_map>
#include <functional>
#include <memory>
#include <stdexcept>

namespace metrics {

/**
 * @brief Concrete factory implementing IMetricFactory (DIP, OCP)
 * @details Open for extension (register new metrics) without modifying existing code
 */
class MetricFactory : public IMetricFactory {
public:
    using ClassificationCreator = std::function<std::unique_ptr<IMetric<int>>()>;
    using RegressionCreator = std::function<std::unique_ptr<IMetric<double>>()>;

private:
    std::unordered_map<std::string, ClassificationCreator> classificationMetrics_;
    std::unordered_map<std::string, RegressionCreator> regressionMetrics_;

public:
    MetricFactory() {
        registerDefaultMetrics();
    }

    /**
     * @brief Register a new classification metric (OCP - extend without modify)
     */
    void registerClassificationMetric(const std::string& name,
                                        ClassificationCreator creator) {
        classificationMetrics_[name] = std::move(creator);
    }

    /**
     * @brief Register a new regression metric (OCP - extend without modify)
     */
    void registerRegressionMetric(const std::string& name,
                                   RegressionCreator creator) {
        regressionMetrics_[name] = std::move(creator);
    }

    std::unique_ptr<IMetric<int>> createClassificationMetric(
            const std::string& name) const override {
        auto it = classificationMetrics_.find(name);
        if (it == classificationMetrics_.end()) {
            throw std::invalid_argument("Unknown classification metric: " + name);
        }
        return it->second();
    }

    std::unique_ptr<IMetric<double>> createRegressionMetric(
            const std::string& name) const override {
        auto it = regressionMetrics_.find(name);
        if (it == regressionMetrics_.end()) {
            throw std::invalid_argument("Unknown regression metric: " + name);
        }
        return it->second();
    }

    /**
     * @brief List all available classification metrics
     */
    std::vector<std::string> listClassificationMetrics() const {
        std::vector<std::string> names;
        for (const auto& pair : classificationMetrics_) {
            names.push_back(pair.first);
        }
        return names;
    }

    /**
     * @brief List all available regression metrics
     */
    std::vector<std::string> listRegressionMetrics() const {
        std::vector<std::string> names;
        for (const auto& pair : regressionMetrics_) {
            names.push_back(pair.first);
        }
        return names;
    }

private:
    void registerDefaultMetrics() {
        // Classification metrics
        registerClassificationMetric("accuracy",
            []() { return std::make_unique<Accuracy>(); });
        registerClassificationMetric("precision",
            []() { return std::make_unique<Precision>(); });
        registerClassificationMetric("recall",
            []() { return std::make_unique<Recall>(); });
        registerClassificationMetric("specificity",
            []() { return std::make_unique<Specificity>(); });
        registerClassificationMetric("f1",
            []() { return std::make_unique<F1Score>(); });
        registerClassificationMetric("mcc",
            []() { return std::make_unique<MatthewsCorrelationCoefficient>(); });
        registerClassificationMetric("kappa",
            []() { return std::make_unique<CohensKappa>(); });

        // Regression metrics
        registerRegressionMetric("mse",
            []() { return std::make_unique<MeanSquaredError>(); });
        registerRegressionMetric("rmse",
            []() { return std::make_unique<RootMeanSquaredError>(); });
        registerRegressionMetric("mae",
            []() { return std::make_unique<MeanAbsoluteError>(); });
        registerRegressionMetric("medae",
            []() { return std::make_unique<MedianAbsoluteError>(); });
        registerRegressionMetric("r2",
            []() { return std::make_unique<RSquared>(); });
        registerRegressionMetric("mape",
            []() { return std::make_unique<MeanAbsolutePercentageError>(); });
        registerRegressionMetric("smape",
            []() { return std::make_unique<SymmetricMAPE>(); });
        registerRegressionMetric("msle",
            []() { return std::make_unique<MeanSquaredLogError>(); });
        registerRegressionMetric("rmsle",
            []() { return std::make_unique<RootMeanSquaredLogError>(); });
        registerRegressionMetric("max_error",
            []() { return std::make_unique<MaxError>(); });
        registerRegressionMetric("explained_variance",
            []() { return std::make_unique<ExplainedVarianceScore>(); });
        registerRegressionMetric("huber",
            []() { return std::make_unique<HuberLoss>(); });
    }
};

/**
 * @brief Singleton access to the metric factory
 */
inline MetricFactory& getMetricFactory() {
    static MetricFactory instance;
    return instance;
}

} // namespace metrics

#endif // METRIC_FACTORY_HPP
