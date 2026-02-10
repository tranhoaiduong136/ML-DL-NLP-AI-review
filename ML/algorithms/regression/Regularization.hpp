#ifndef ML_ALGORITHMS_REGULARIZATION_HPP
#define ML_ALGORITHMS_REGULARIZATION_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <string>

namespace ml {
namespace algorithms {

/**
 * @brief Abstract interface for regularization strategies (Interface Segregation Principle)
 *
 * Defines the contract for regularization techniques used in linear models.
 * Following Strategy Pattern and Dependency Inversion Principle.
 *
 * Regularization adds a penalty term to the loss function:
 * Loss = RSS + penalty(coefficients)
 *
 * Where RSS = Σ(y - Xβ)² is the residual sum of squares.
 */
class IRegularizer {
public:
    virtual ~IRegularizer() = default;

    /**
     * @brief Calculate the regularization penalty for given coefficients
     * @param coefficients Model coefficients (excluding intercept)
     * @return Penalty value to add to the loss function
     */
    virtual double penalty(const std::vector<double>& coefficients) const = 0;

    /**
     * @brief Calculate the gradient of the penalty term
     * @param coefficients Model coefficients (excluding intercept)
     * @return Gradient vector (subgradient for L1)
     */
    virtual std::vector<double> gradient(const std::vector<double>& coefficients) const = 0;

    /**
     * @brief Apply regularization to X'X matrix (for closed-form solutions)
     * @param XtX The X'X matrix to modify
     * @param hasIntercept Whether the first element is the intercept term
     *
     * For Ridge regression, this adds λI to the diagonal (excluding intercept).
     * For L1/Elastic Net, this may be a no-op (iterative methods needed).
     */
    virtual void applyToXtX(std::vector<std::vector<double>>& XtX,
                            bool hasIntercept) const = 0;

    /**
     * @brief Apply soft-thresholding for coordinate descent
     * @param value The value to threshold
     * @param threshold The threshold value
     * @return Soft-thresholded value
     */
    virtual double softThreshold(double value, double threshold) const = 0;

    /**
     * @brief Get the regularization strength (lambda)
     * @return Lambda value
     */
    virtual double getLambda() const = 0;

    /**
     * @brief Check if this regularizer supports closed-form solution
     * @return true if closed-form (like Ridge), false if iterative needed (like Lasso)
     */
    virtual bool supportsClosedForm() const = 0;

    /**
     * @brief Get a descriptive name for this regularizer
     * @return Name string
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Clone this regularizer (for polymorphic copying)
     * @return Unique pointer to a copy of this regularizer
     */
    virtual std::unique_ptr<IRegularizer> clone() const = 0;
};

/**
 * @brief No regularization (null object pattern)
 *
 * Provides a default implementation that applies no penalty.
 * Useful as a default value and for testing.
 */
class NoRegularizer final : public IRegularizer {
public:
    NoRegularizer() = default;
    ~NoRegularizer() override = default;

    NoRegularizer(const NoRegularizer&) = default;
    NoRegularizer& operator=(const NoRegularizer&) = default;
    NoRegularizer(NoRegularizer&&) noexcept = default;
    NoRegularizer& operator=(NoRegularizer&&) noexcept = default;

    double penalty(const std::vector<double>& /*coefficients*/) const override {
        return 0.0;
    }

    std::vector<double> gradient(const std::vector<double>& coefficients) const override {
        return std::vector<double>(coefficients.size(), 0.0);
    }

    void applyToXtX(std::vector<std::vector<double>>& /*XtX*/,
                    bool /*hasIntercept*/) const override {
        // No modification needed
    }

    double softThreshold(double value, double /*threshold*/) const override {
        return value;
    }

    double getLambda() const override {
        return 0.0;
    }

    bool supportsClosedForm() const override {
        return true;
    }

    std::string getName() const override {
        return "None";
    }

    std::unique_ptr<IRegularizer> clone() const override {
        return std::make_unique<NoRegularizer>(*this);
    }
};

/**
 * @brief L2 Regularization (Ridge Regression)
 *
 * Adds L2 penalty: λ * Σβj²
 *
 * Properties:
 * - Shrinks coefficients towards zero but doesn't set them to exactly zero
 * - Handles multicollinearity well
 * - Has closed-form solution: (X'X + λI)⁻¹ X'y
 * - Differentiable everywhere
 */
class L2Regularizer final : public IRegularizer {
public:
    /**
     * @brief Construct L2 regularizer with given strength
     * @param lambda Regularization strength (λ ≥ 0)
     * @throws std::invalid_argument if lambda < 0
     */
    explicit L2Regularizer(double lambda);

    ~L2Regularizer() override = default;

    L2Regularizer(const L2Regularizer&) = default;
    L2Regularizer& operator=(const L2Regularizer&) = default;
    L2Regularizer(L2Regularizer&&) noexcept = default;
    L2Regularizer& operator=(L2Regularizer&&) noexcept = default;

    /**
     * @brief Calculate L2 penalty: λ * Σβj²
     */
    double penalty(const std::vector<double>& coefficients) const override;

    /**
     * @brief Calculate L2 gradient: 2λβ
     */
    std::vector<double> gradient(const std::vector<double>& coefficients) const override;

    /**
     * @brief Add λ to diagonal of X'X (excluding intercept if present)
     */
    void applyToXtX(std::vector<std::vector<double>>& XtX,
                    bool hasIntercept) const override;

    double softThreshold(double value, double /*threshold*/) const override;

    double getLambda() const override;

    bool supportsClosedForm() const override;

    std::string getName() const override;

    std::unique_ptr<IRegularizer> clone() const override;

private:
    double lambda_;
};

/**
 * @brief L1 Regularization (Lasso Regression)
 *
 * Adds L1 penalty: λ * Σ|βj|
 *
 * Properties:
 * - Can shrink coefficients to exactly zero (feature selection)
 * - Produces sparse solutions
 * - No closed-form solution (requires iterative methods like coordinate descent)
 * - Not differentiable at zero (uses subgradient)
 */
class L1Regularizer final : public IRegularizer {
public:
    /**
     * @brief Construct L1 regularizer with given strength
     * @param lambda Regularization strength (λ ≥ 0)
     * @throws std::invalid_argument if lambda < 0
     */
    explicit L1Regularizer(double lambda);

    ~L1Regularizer() override = default;

    L1Regularizer(const L1Regularizer&) = default;
    L1Regularizer& operator=(const L1Regularizer&) = default;
    L1Regularizer(L1Regularizer&&) noexcept = default;
    L1Regularizer& operator=(L1Regularizer&&) noexcept = default;

    /**
     * @brief Calculate L1 penalty: λ * Σ|βj|
     */
    double penalty(const std::vector<double>& coefficients) const override;

    /**
     * @brief Calculate L1 subgradient: λ * sign(β)
     */
    std::vector<double> gradient(const std::vector<double>& coefficients) const override;

    /**
     * @brief No modification for L1 (requires iterative solution)
     */
    void applyToXtX(std::vector<std::vector<double>>& XtX,
                    bool hasIntercept) const override;

    /**
     * @brief Apply soft-thresholding: sign(x) * max(|x| - threshold, 0)
     */
    double softThreshold(double value, double threshold) const override;

    double getLambda() const override;

    bool supportsClosedForm() const override;

    std::string getName() const override;

    std::unique_ptr<IRegularizer> clone() const override;

private:
    double lambda_;
};

/**
 * @brief Elastic Net Regularization
 *
 * Combines L1 and L2 penalties: λ * (α * Σ|βj| + (1-α) * Σβj²)
 *
 * Properties:
 * - α = 1: Pure Lasso (L1)
 * - α = 0: Pure Ridge (L2)
 * - 0 < α < 1: Combination of both
 * - Handles multicollinearity (from L2)
 * - Produces sparse solutions (from L1)
 * - Grouping effect: correlated features get similar coefficients
 */
class ElasticNetRegularizer final : public IRegularizer {
public:
    /**
     * @brief Construct Elastic Net regularizer
     * @param lambda Overall regularization strength (λ ≥ 0)
     * @param alpha L1 ratio (0 ≤ α ≤ 1), where 1 = pure L1, 0 = pure L2
     * @throws std::invalid_argument if lambda < 0 or alpha not in [0, 1]
     */
    ElasticNetRegularizer(double lambda, double alpha);

    ~ElasticNetRegularizer() override = default;

    ElasticNetRegularizer(const ElasticNetRegularizer&) = default;
    ElasticNetRegularizer& operator=(const ElasticNetRegularizer&) = default;
    ElasticNetRegularizer(ElasticNetRegularizer&&) noexcept = default;
    ElasticNetRegularizer& operator=(ElasticNetRegularizer&&) noexcept = default;

    /**
     * @brief Calculate Elastic Net penalty: λ * (α * Σ|βj| + (1-α) * Σβj²)
     */
    double penalty(const std::vector<double>& coefficients) const override;

    /**
     * @brief Calculate combined gradient/subgradient
     */
    std::vector<double> gradient(const std::vector<double>& coefficients) const override;

    /**
     * @brief Apply L2 component to X'X matrix
     */
    void applyToXtX(std::vector<std::vector<double>>& XtX,
                    bool hasIntercept) const override;

    /**
     * @brief Apply soft-thresholding for L1 component
     */
    double softThreshold(double value, double threshold) const override;

    double getLambda() const override;

    /**
     * @brief Get the L1 ratio (alpha)
     * @return Alpha value in [0, 1]
     */
    double getAlpha() const;

    /**
     * @brief Get the effective L1 penalty strength
     * @return λ * α
     */
    double getL1Penalty() const;

    /**
     * @brief Get the effective L2 penalty strength
     * @return λ * (1 - α)
     */
    double getL2Penalty() const;

    bool supportsClosedForm() const override;

    std::string getName() const override;

    std::unique_ptr<IRegularizer> clone() const override;

private:
    double lambda_;
    double alpha_;
};

} // namespace algorithms
} // namespace ml

#endif // ML_ALGORITHMS_REGULARIZATION_HPP
