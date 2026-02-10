#ifndef ML_ALGORITHMS_REGULARIZED_LINEAR_REGRESSION_HPP
#define ML_ALGORITHMS_REGULARIZED_LINEAR_REGRESSION_HPP

#include "LinearRegression.hpp"
#include "Regularization.hpp"

#include <memory>
#include <cstddef>

namespace ml {
namespace algorithms {

/**
 * @brief Linear Regression with Regularization support
 *
 * Implements regularized linear regression following SOLID principles:
 * - Single Responsibility: Handles regularized linear regression fitting/prediction
 * - Open/Closed: Uses Strategy Pattern with IRegularizer for extensibility
 * - Liskov Substitution: Can be used wherever IRegressor is expected
 * - Interface Segregation: Implements focused interfaces (IRegressor, ILinearModel)
 * - Dependency Inversion: Depends on IRegularizer abstraction
 *
 * Supports:
 * - Ridge Regression (L2): Closed-form solution with modified normal equations
 * - Lasso Regression (L1): Coordinate descent algorithm
 * - Elastic Net: Combination of L1 and L2 with coordinate descent
 *
 * The model minimizes: ||y - Xβ||² + penalty(β)
 *
 * Where penalty depends on the regularizer:
 * - L2 (Ridge): λ * Σβj²
 * - L1 (Lasso): λ * Σ|βj|
 * - Elastic Net: λ * (α * Σ|βj| + (1-α) * Σβj²)
 */
class RegularizedLinearRegression : public IRegressor, public ILinearModel {
public:
    /**
     * @brief Construct with a specific regularizer
     * @param regularizer The regularization strategy to use (ownership transferred)
     * @param fitIntercept Whether to calculate the intercept (default: true)
     * @param maxIterations Maximum iterations for coordinate descent (default: 1000)
     * @param tolerance Convergence tolerance for coordinate descent (default: 1e-4)
     */
    explicit RegularizedLinearRegression(
        std::unique_ptr<IRegularizer> regularizer,
        bool fitIntercept = true,
        std::size_t maxIterations = 1000,
        double tolerance = 1e-4);

    /**
     * @brief Construct with default NoRegularizer (equivalent to OLS)
     * @param fitIntercept Whether to calculate the intercept (default: true)
     */
    explicit RegularizedLinearRegression(bool fitIntercept = true);

    ~RegularizedLinearRegression() override = default;

    // Disable copy (model state and unique_ptr shouldn't be accidentally shared)
    RegularizedLinearRegression(const RegularizedLinearRegression&) = delete;
    RegularizedLinearRegression& operator=(const RegularizedLinearRegression&) = delete;

    // Enable move semantics
    RegularizedLinearRegression(RegularizedLinearRegression&&) noexcept = default;
    RegularizedLinearRegression& operator=(RegularizedLinearRegression&&) noexcept = default;

    /**
     * @brief Fit the regularized linear model to training data
     * @param X Feature matrix (n_samples x n_features)
     * @param y Target values (n_samples)
     * @throws std::invalid_argument if X and y dimensions don't match
     *
     * Uses closed-form solution for L2 regularization,
     * coordinate descent for L1 and Elastic Net.
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

    /**
     * @brief Get the regularizer being used
     * @return Const reference to the regularizer
     */
    const IRegularizer& getRegularizer() const;

    /**
     * @brief Get the number of iterations used in last fit (for iterative methods)
     * @return Number of iterations (0 for closed-form solutions)
     */
    std::size_t getIterations() const noexcept;

    /**
     * @brief Set a new regularizer
     * @param regularizer The new regularization strategy (ownership transferred)
     *
     * Note: This resets the fitted state - you must call fit() again.
     */
    void setRegularizer(std::unique_ptr<IRegularizer> regularizer);

private:
    std::unique_ptr<IRegularizer> regularizer_;
    std::vector<double> coefficients_;
    double intercept_;
    bool fitIntercept_;
    bool fitted_;
    std::size_t numFeatures_;
    std::size_t maxIterations_;
    double tolerance_;
    std::size_t lastIterations_;

    // Validation methods
    void validateInputDimensions(const std::vector<std::vector<double>>& X,
                                  const std::vector<double>& y) const;

    void validatePredictionInput(const std::vector<std::vector<double>>& X) const;

    // Fitting methods
    void fitClosedForm(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y);

    void fitCoordinateDescent(const std::vector<std::vector<double>>& X,
                               const std::vector<double>& y);

    // Matrix operations
    std::vector<std::vector<double>> computeXtX(const std::vector<std::vector<double>>& X) const;

    std::vector<double> computeXtY(const std::vector<std::vector<double>>& X,
                                    const std::vector<double>& y) const;

    std::vector<double> solveLinearSystem(std::vector<std::vector<double>>& A,
                                           std::vector<double>& b) const;

    // Utility methods
    double computeMean(const std::vector<double>& v) const;

    std::vector<double> computeColumnMeans(const std::vector<std::vector<double>>& X) const;

    std::vector<double> computeColumnStds(const std::vector<std::vector<double>>& X,
                                           const std::vector<double>& means) const;

    void standardizeFeatures(std::vector<std::vector<double>>& X,
                             std::vector<double>& means,
                             std::vector<double>& stds) const;
};

// =============================================================================
// Factory functions for convenient construction (following Factory Method pattern)
// =============================================================================

/**
 * @brief Create a Ridge Regression model (L2 regularization)
 * @param lambda Regularization strength (λ ≥ 0)
 * @param fitIntercept Whether to calculate the intercept
 * @return Configured RegularizedLinearRegression instance
 */
inline std::unique_ptr<RegularizedLinearRegression> createRidgeRegression(
    double lambda,
    bool fitIntercept = true)
{
    return std::make_unique<RegularizedLinearRegression>(
        std::make_unique<L2Regularizer>(lambda),
        fitIntercept);
}

/**
 * @brief Create a Lasso Regression model (L1 regularization)
 * @param lambda Regularization strength (λ ≥ 0)
 * @param fitIntercept Whether to calculate the intercept
 * @param maxIterations Maximum iterations for coordinate descent
 * @param tolerance Convergence tolerance
 * @return Configured RegularizedLinearRegression instance
 */
inline std::unique_ptr<RegularizedLinearRegression> createLassoRegression(
    double lambda,
    bool fitIntercept = true,
    std::size_t maxIterations = 1000,
    double tolerance = 1e-4)
{
    return std::make_unique<RegularizedLinearRegression>(
        std::make_unique<L1Regularizer>(lambda),
        fitIntercept,
        maxIterations,
        tolerance);
}

/**
 * @brief Create an Elastic Net Regression model
 * @param lambda Overall regularization strength (λ ≥ 0)
 * @param alpha L1 ratio (0 ≤ α ≤ 1), where 1 = pure Lasso, 0 = pure Ridge
 * @param fitIntercept Whether to calculate the intercept
 * @param maxIterations Maximum iterations for coordinate descent
 * @param tolerance Convergence tolerance
 * @return Configured RegularizedLinearRegression instance
 */
inline std::unique_ptr<RegularizedLinearRegression> createElasticNetRegression(
    double lambda,
    double alpha,
    bool fitIntercept = true,
    std::size_t maxIterations = 1000,
    double tolerance = 1e-4)
{
    return std::make_unique<RegularizedLinearRegression>(
        std::make_unique<ElasticNetRegularizer>(lambda, alpha),
        fitIntercept,
        maxIterations,
        tolerance);
}

} // namespace algorithms
} // namespace ml

#endif // ML_ALGORITHMS_REGULARIZED_LINEAR_REGRESSION_HPP
