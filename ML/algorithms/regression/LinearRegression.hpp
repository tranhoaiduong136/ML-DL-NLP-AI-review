#ifndef ML_ALGORITHMS_LINEAR_REGRESSION_HPP
#define ML_ALGORITHMS_LINEAR_REGRESSION_HPP

#include <vector>
#include <stdexcept>
#include <cstddef>

namespace ml {
namespace algorithms {

/**
 * @brief Abstract interface for regression models (Interface Segregation Principle)
 *
 * Defines the contract for supervised learning regression models.
 * Following Dependency Inversion Principle - depend on abstractions.
 */
class IRegressor {
public:
    virtual ~IRegressor() = default;

    virtual void fit(const std::vector<std::vector<double>>& X,
                     const std::vector<double>& y) = 0;

    virtual std::vector<double> predict(const std::vector<std::vector<double>>& X) const = 0;

    virtual double score(const std::vector<std::vector<double>>& X,
                         const std::vector<double>& y) const = 0;
};

/**
 * @brief Abstract interface for models that expose coefficients
 */
class ILinearModel {
public:
    virtual ~ILinearModel() = default;

    virtual std::vector<double> getCoefficients() const = 0;
    virtual double getIntercept() const = 0;
};

/**
 * @brief Linear Regression using Ordinary Least Squares (OLS)
 *
 * Implements linear regression following SOLID principles:
 * - Single Responsibility: Only handles linear regression fitting/prediction
 * - Open/Closed: Extends IRegressor interface, can be extended via inheritance
 * - Liskov Substitution: Can be used wherever IRegressor is expected
 * - Interface Segregation: Implements focused interfaces
 * - Dependency Inversion: Depends on abstractions (IRegressor)
 */
class LinearRegression final : public IRegressor, public ILinearModel {
public:
    /**
     * @brief Construct a new Linear Regression model
     * @param fitIntercept Whether to calculate the intercept (default: true)
     */
    explicit LinearRegression(bool fitIntercept = true);

    ~LinearRegression() override = default;

    // Disable copy (Single Responsibility - model state shouldn't be accidentally shared)
    LinearRegression(const LinearRegression&) = delete;
    LinearRegression& operator=(const LinearRegression&) = delete;

    // Enable move semantics (C++11)
    LinearRegression(LinearRegression&&) noexcept = default;
    LinearRegression& operator=(LinearRegression&&) noexcept = default;

    /**
     * @brief Fit the linear model to training data
     * @param X Feature matrix (n_samples x n_features)
     * @param y Target values (n_samples)
     * @throws std::invalid_argument if X and y dimensions don't match
     */
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y) override;

    /**
     * @brief Predict target values for input features
     * @param X Feature matrix (n_samples x n_features)
     * @return Predicted values
     * @throws std::logic_error if model hasn't been fitted
     * @throws std::invalid_argument if feature dimensions don't match training data
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const override;

    /**
     * @brief Calculate R² score (coefficient of determination)
     * @param X Feature matrix
     * @param y True target values
     * @return R² score (1.0 is perfect prediction)
     */
    double score(const std::vector<std::vector<double>>& X,
                 const std::vector<double>& y) const override;

    /**
     * @brief Get the model coefficients (weights)
     * @return Vector of coefficients for each feature
     */
    std::vector<double> getCoefficients() const override;

    /**
     * @brief Get the intercept (bias term)
     * @return Intercept value
     */
    double getIntercept() const override;

    /**
     * @brief Check if the model has been fitted
     * @return true if fit() has been called successfully
     */
    bool isFitted() const noexcept;

private:
    std::vector<double> coefficients_;
    double intercept_;
    bool fitIntercept_;
    bool fitted_;
    std::size_t numFeatures_;

    void validateInputDimensions(const std::vector<std::vector<double>>& X,
                                  const std::vector<double>& y) const;

    void validatePredictionInput(const std::vector<std::vector<double>>& X) const;

    std::vector<std::vector<double>> computeXtX(const std::vector<std::vector<double>>& X) const;

    std::vector<double> computeXtY(const std::vector<std::vector<double>>& X,
                                    const std::vector<double>& y) const;

    std::vector<double> solveLinearSystem(std::vector<std::vector<double>>& A,
                                           std::vector<double>& b) const;

    double computeMean(const std::vector<double>& v) const;
};

} // namespace algorithms
} // namespace ml

#endif // ML_ALGORITHMS_LINEAR_REGRESSION_HPP
