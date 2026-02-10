#include "Regularization.hpp"

#include <cmath>
#include <numeric>

namespace ml {
namespace algorithms {

// =============================================================================
// L2Regularizer (Ridge) Implementation
// =============================================================================

L2Regularizer::L2Regularizer(double lambda)
    : lambda_(lambda)
{
    if (lambda < 0.0) {
        throw std::invalid_argument(
            "Regularization strength lambda must be non-negative, got: " +
            std::to_string(lambda));
    }
}

double L2Regularizer::penalty(const std::vector<double>& coefficients) const
{
    double sumSquared = 0.0;
    for (const auto& coef : coefficients) {
        sumSquared += coef * coef;
    }
    return lambda_ * sumSquared;
}

std::vector<double> L2Regularizer::gradient(const std::vector<double>& coefficients) const
{
    std::vector<double> grad;
    grad.reserve(coefficients.size());

    for (const auto& coef : coefficients) {
        grad.push_back(2.0 * lambda_ * coef);
    }

    return grad;
}

void L2Regularizer::applyToXtX(std::vector<std::vector<double>>& XtX,
                                bool hasIntercept) const
{
    if (XtX.empty()) {
        return;
    }

    // Add lambda to diagonal elements (excluding intercept if present)
    const std::size_t startIdx = hasIntercept ? 1 : 0;

    for (std::size_t i = startIdx; i < XtX.size(); ++i) {
        XtX[i][i] += lambda_;
    }
}

double L2Regularizer::softThreshold(double value, double /*threshold*/) const
{
    // L2 doesn't use soft-thresholding, return value scaled
    return value / (1.0 + lambda_);
}

double L2Regularizer::getLambda() const
{
    return lambda_;
}

bool L2Regularizer::supportsClosedForm() const
{
    return true;
}

std::string L2Regularizer::getName() const
{
    return "L2 (Ridge)";
}

std::unique_ptr<IRegularizer> L2Regularizer::clone() const
{
    return std::make_unique<L2Regularizer>(*this);
}

// =============================================================================
// L1Regularizer (Lasso) Implementation
// =============================================================================

L1Regularizer::L1Regularizer(double lambda)
    : lambda_(lambda)
{
    if (lambda < 0.0) {
        throw std::invalid_argument(
            "Regularization strength lambda must be non-negative, got: " +
            std::to_string(lambda));
    }
}

double L1Regularizer::penalty(const std::vector<double>& coefficients) const
{
    double sumAbs = 0.0;
    for (const auto& coef : coefficients) {
        sumAbs += std::abs(coef);
    }
    return lambda_ * sumAbs;
}

std::vector<double> L1Regularizer::gradient(const std::vector<double>& coefficients) const
{
    std::vector<double> grad;
    grad.reserve(coefficients.size());

    for (const auto& coef : coefficients) {
        // Subgradient: sign(coef) * lambda
        // At zero, we use 0 as the subgradient (any value in [-lambda, lambda] is valid)
        if (coef > 0.0) {
            grad.push_back(lambda_);
        } else if (coef < 0.0) {
            grad.push_back(-lambda_);
        } else {
            grad.push_back(0.0);
        }
    }

    return grad;
}

void L1Regularizer::applyToXtX(std::vector<std::vector<double>>& /*XtX*/,
                                bool /*hasIntercept*/) const
{
    // L1 regularization doesn't modify X'X - requires iterative solution
    // This is intentionally a no-op
}

double L1Regularizer::softThreshold(double value, double threshold) const
{
    // Soft-thresholding operator: S(z, γ) = sign(z) * max(|z| - γ, 0)
    if (value > threshold) {
        return value - threshold;
    } else if (value < -threshold) {
        return value + threshold;
    } else {
        return 0.0;
    }
}

double L1Regularizer::getLambda() const
{
    return lambda_;
}

bool L1Regularizer::supportsClosedForm() const
{
    return false;
}

std::string L1Regularizer::getName() const
{
    return "L1 (Lasso)";
}

std::unique_ptr<IRegularizer> L1Regularizer::clone() const
{
    return std::make_unique<L1Regularizer>(*this);
}

// =============================================================================
// ElasticNetRegularizer Implementation
// =============================================================================

ElasticNetRegularizer::ElasticNetRegularizer(double lambda, double alpha)
    : lambda_(lambda)
    , alpha_(alpha)
{
    if (lambda < 0.0) {
        throw std::invalid_argument(
            "Regularization strength lambda must be non-negative, got: " +
            std::to_string(lambda));
    }

    if (alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument(
            "L1 ratio alpha must be in [0, 1], got: " +
            std::to_string(alpha));
    }
}

double ElasticNetRegularizer::penalty(const std::vector<double>& coefficients) const
{
    double l1Penalty = 0.0;
    double l2Penalty = 0.0;

    for (const auto& coef : coefficients) {
        l1Penalty += std::abs(coef);
        l2Penalty += coef * coef;
    }

    // λ * (α * L1 + (1-α) * L2)
    return lambda_ * (alpha_ * l1Penalty + (1.0 - alpha_) * l2Penalty);
}

std::vector<double> ElasticNetRegularizer::gradient(const std::vector<double>& coefficients) const
{
    std::vector<double> grad;
    grad.reserve(coefficients.size());

    for (const auto& coef : coefficients) {
        // L1 subgradient component
        double l1Grad = 0.0;
        if (coef > 0.0) {
            l1Grad = 1.0;
        } else if (coef < 0.0) {
            l1Grad = -1.0;
        }

        // L2 gradient component
        double l2Grad = 2.0 * coef;

        // Combined: λ * (α * sign(coef) + (1-α) * 2 * coef)
        grad.push_back(lambda_ * (alpha_ * l1Grad + (1.0 - alpha_) * l2Grad));
    }

    return grad;
}

void ElasticNetRegularizer::applyToXtX(std::vector<std::vector<double>>& XtX,
                                        bool hasIntercept) const
{
    if (XtX.empty()) {
        return;
    }

    // Apply only the L2 component to the diagonal
    const double l2Lambda = lambda_ * (1.0 - alpha_);
    const std::size_t startIdx = hasIntercept ? 1 : 0;

    for (std::size_t i = startIdx; i < XtX.size(); ++i) {
        XtX[i][i] += l2Lambda;
    }
}

double ElasticNetRegularizer::softThreshold(double value, double threshold) const
{
    // Soft-thresholding for the L1 component
    // Scaled by (1 + λ(1-α)) for Elastic Net coordinate descent
    double l2Factor = 1.0 + lambda_ * (1.0 - alpha_);

    if (value > threshold) {
        return (value - threshold) / l2Factor;
    } else if (value < -threshold) {
        return (value + threshold) / l2Factor;
    } else {
        return 0.0;
    }
}

double ElasticNetRegularizer::getLambda() const
{
    return lambda_;
}

double ElasticNetRegularizer::getAlpha() const
{
    return alpha_;
}

double ElasticNetRegularizer::getL1Penalty() const
{
    return lambda_ * alpha_;
}

double ElasticNetRegularizer::getL2Penalty() const
{
    return lambda_ * (1.0 - alpha_);
}

bool ElasticNetRegularizer::supportsClosedForm() const
{
    // Only pure L2 (alpha = 0) supports closed-form
    return alpha_ == 0.0;
}

std::string ElasticNetRegularizer::getName() const
{
    return "Elastic Net (alpha=" + std::to_string(alpha_) + ")";
}

std::unique_ptr<IRegularizer> ElasticNetRegularizer::clone() const
{
    return std::make_unique<ElasticNetRegularizer>(*this);
}

} // namespace algorithms
} // namespace ml
