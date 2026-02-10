#include "LinearRegression.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

namespace ml {
namespace algorithms {

LinearRegression::LinearRegression(bool fitIntercept)
    : coefficients_()
    , intercept_(0.0)
    , fitIntercept_(fitIntercept)
    , fitted_(false)
    , numFeatures_(0)
{
}

void LinearRegression::fit(const std::vector<std::vector<double>>& X,
                            const std::vector<double>& y)
{
    validateInputDimensions(X, y);

    const std::size_t numSamples = X.size();
    const std::size_t numFeatures = X[0].size();

    // Adjust for intercept by adding a column of ones
    std::vector<std::vector<double>> XAugmented;
    XAugmented.reserve(numSamples);

    if (fitIntercept_) {
        for (const auto& row : X) {
            std::vector<double> augmentedRow;
            augmentedRow.reserve(numFeatures + 1);
            augmentedRow.push_back(1.0); // Intercept term
            augmentedRow.insert(augmentedRow.end(), row.begin(), row.end());
            XAugmented.push_back(std::move(augmentedRow));
        }
    } else {
        XAugmented = X;
    }

    // Compute X^T * X
    auto XtX = computeXtX(XAugmented);

    // Compute X^T * y
    auto XtY = computeXtY(XAugmented, y);

    // Solve the normal equations: (X^T * X) * beta = X^T * y
    auto solution = solveLinearSystem(XtX, XtY);

    // Extract intercept and coefficients
    if (fitIntercept_) {
        intercept_ = solution[0];
        coefficients_.assign(solution.begin() + 1, solution.end());
    } else {
        intercept_ = 0.0;
        coefficients_ = std::move(solution);
    }

    numFeatures_ = numFeatures;
    fitted_ = true;
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X) const
{
    validatePredictionInput(X);

    std::vector<double> predictions;
    predictions.reserve(X.size());

    for (const auto& sample : X) {
        double prediction = intercept_;
        for (std::size_t j = 0; j < coefficients_.size(); ++j) {
            prediction += coefficients_[j] * sample[j];
        }
        predictions.push_back(prediction);
    }

    return predictions;
}

double LinearRegression::score(const std::vector<std::vector<double>>& X,
                                const std::vector<double>& y) const
{
    auto predictions = predict(X);

    const double yMean = computeMean(y);

    double ssRes = 0.0; // Residual sum of squares
    double ssTot = 0.0; // Total sum of squares

    for (std::size_t i = 0; i < y.size(); ++i) {
        const double residual = y[i] - predictions[i];
        const double deviation = y[i] - yMean;
        ssRes += residual * residual;
        ssTot += deviation * deviation;
    }

    // Handle edge case where all y values are the same
    if (ssTot < std::numeric_limits<double>::epsilon()) {
        return (ssRes < std::numeric_limits<double>::epsilon()) ? 1.0 : 0.0;
    }

    return 1.0 - (ssRes / ssTot);
}

std::vector<double> LinearRegression::getCoefficients() const
{
    return coefficients_;
}

double LinearRegression::getIntercept() const
{
    return intercept_;
}

bool LinearRegression::isFitted() const noexcept
{
    return fitted_;
}

void LinearRegression::validateInputDimensions(const std::vector<std::vector<double>>& X,
                                                const std::vector<double>& y) const
{
    if (X.empty()) {
        throw std::invalid_argument("Feature matrix X cannot be empty");
    }

    if (y.empty()) {
        throw std::invalid_argument("Target vector y cannot be empty");
    }

    if (X.size() != y.size()) {
        throw std::invalid_argument(
            "Number of samples in X (" + std::to_string(X.size()) +
            ") must match length of y (" + std::to_string(y.size()) + ")");
    }

    const std::size_t numFeatures = X[0].size();
    if (numFeatures == 0) {
        throw std::invalid_argument("Feature matrix X must have at least one feature");
    }

    // Check all rows have same number of features
    for (std::size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != numFeatures) {
            throw std::invalid_argument(
                "Inconsistent number of features at row " + std::to_string(i) +
                ": expected " + std::to_string(numFeatures) +
                ", got " + std::to_string(X[i].size()));
        }
    }
}

void LinearRegression::validatePredictionInput(const std::vector<std::vector<double>>& X) const
{
    if (!fitted_) {
        throw std::logic_error("Model has not been fitted. Call fit() before predict()");
    }

    if (X.empty()) {
        throw std::invalid_argument("Feature matrix X cannot be empty");
    }

    for (std::size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != numFeatures_) {
            throw std::invalid_argument(
                "Sample " + std::to_string(i) + " has " + std::to_string(X[i].size()) +
                " features, but model was trained with " + std::to_string(numFeatures_));
        }
    }
}

std::vector<std::vector<double>> LinearRegression::computeXtX(
    const std::vector<std::vector<double>>& X) const
{
    const std::size_t numFeatures = X[0].size();
    const std::size_t numSamples = X.size();

    // Initialize result matrix with zeros
    std::vector<std::vector<double>> result(numFeatures, std::vector<double>(numFeatures, 0.0));

    // Compute X^T * X
    for (std::size_t i = 0; i < numFeatures; ++i) {
        for (std::size_t j = i; j < numFeatures; ++j) { // Exploit symmetry
            double sum = 0.0;
            for (std::size_t k = 0; k < numSamples; ++k) {
                sum += X[k][i] * X[k][j];
            }
            result[i][j] = sum;
            result[j][i] = sum; // Symmetric matrix
        }
    }

    return result;
}

std::vector<double> LinearRegression::computeXtY(const std::vector<std::vector<double>>& X,
                                                   const std::vector<double>& y) const
{
    const std::size_t numFeatures = X[0].size();
    const std::size_t numSamples = X.size();

    std::vector<double> result(numFeatures, 0.0);

    for (std::size_t j = 0; j < numFeatures; ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < numSamples; ++i) {
            sum += X[i][j] * y[i];
        }
        result[j] = sum;
    }

    return result;
}

std::vector<double> LinearRegression::solveLinearSystem(std::vector<std::vector<double>>& A,
                                                          std::vector<double>& b) const
{
    // Gaussian elimination with partial pivoting
    const std::size_t n = A.size();

    // Forward elimination
    for (std::size_t col = 0; col < n; ++col) {
        // Find pivot (partial pivoting for numerical stability)
        std::size_t maxRow = col;
        double maxVal = std::abs(A[col][col]);

        for (std::size_t row = col + 1; row < n; ++row) {
            if (std::abs(A[row][col]) > maxVal) {
                maxVal = std::abs(A[row][col]);
                maxRow = row;
            }
        }

        // Check for singular matrix
        if (maxVal < std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error(
                "Matrix is singular or nearly singular. "
                "Features may be linearly dependent.");
        }

        // Swap rows
        if (maxRow != col) {
            std::swap(A[col], A[maxRow]);
            std::swap(b[col], b[maxRow]);
        }

        // Eliminate column
        for (std::size_t row = col + 1; row < n; ++row) {
            const double factor = A[row][col] / A[col][col];
            for (std::size_t j = col; j < n; ++j) {
                A[row][j] -= factor * A[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    std::vector<double> x(n);
    for (std::size_t i = n; i-- > 0;) {
        double sum = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }

    return x;
}

double LinearRegression::computeMean(const std::vector<double>& v) const
{
    if (v.empty()) {
        return 0.0;
    }

    const double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / static_cast<double>(v.size());
}

} // namespace algorithms
} // namespace ml
