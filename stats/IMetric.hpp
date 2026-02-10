/**
 * @file IMetric.hpp
 * @brief Interface Segregation: Small, focused metric interfaces
 * @details Base interfaces for ML/DL evaluation metrics following SOLID principles
 */

#ifndef IMETRIC_HPP
#define IMETRIC_HPP

#include <vector>
#include <string>
#include <memory>

namespace metrics {

/**
 * @brief Base metric interface (Interface Segregation Principle)
 * @tparam T Input type for predictions/labels
 * @tparam R Return type for the metric result
 */
template<typename T, typename R = double>
class IMetric {
public:
    virtual ~IMetric() = default;

    /**
     * @brief Calculate the metric value
     * @param y_true Ground truth values
     * @param y_pred Predicted values
     * @return Metric result
     */
    virtual R calculate(const std::vector<T>& y_true,
                        const std::vector<T>& y_pred) const = 0;

    /**
     * @brief Get metric name
     * @return Human-readable metric name
     */
    virtual std::string name() const = 0;

    /**
     * @brief Check if higher values are better
     * @return true if higher is better, false otherwise
     */
    virtual bool higherIsBetter() const = 0;
};

/**
 * @brief Interface for probability-based metrics (ISP)
 */
template<typename T = double>
class IProbabilisticMetric {
public:
    virtual ~IProbabilisticMetric() = default;

    /**
     * @brief Calculate metric using probability predictions
     * @param y_true Ground truth binary labels
     * @param y_prob Predicted probabilities
     * @return Metric result
     */
    virtual double calculate(const std::vector<int>& y_true,
                             const std::vector<T>& y_prob) const = 0;

    virtual std::string name() const = 0;
};

/**
 * @brief Interface for multi-class metrics (ISP)
 */
class IMultiClassMetric {
public:
    virtual ~IMultiClassMetric() = default;

    virtual double calculate(const std::vector<int>& y_true,
                             const std::vector<int>& y_pred,
                             int num_classes) const = 0;

    virtual std::string name() const = 0;
};

/**
 * @brief Factory interface for creating metrics (DIP)
 */
class IMetricFactory {
public:
    virtual ~IMetricFactory() = default;

    virtual std::unique_ptr<IMetric<int>> createClassificationMetric(
        const std::string& name) const = 0;

    virtual std::unique_ptr<IMetric<double>> createRegressionMetric(
        const std::string& name) const = 0;
};

} // namespace metrics

#endif // IMETRIC_HPP
