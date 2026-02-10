#include "PCA.hpp"
#include <Eigen/SVD>

namespace ml {

PCA::PCA(int n_components)
    : n_components_(n_components)
    , n_features_(0)
    , fitted_(false) {
    if (n_components < 0) {
        throw std::invalid_argument("n_components must be non-negative");
    }
}

PCA& PCA::fit(const Matrix& X) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("Input matrix cannot be empty");
    }

    const int n_samples = static_cast<int>(X.rows());
    n_features_ = static_cast<int>(X.cols());

    // Determine actual number of components
    int max_components = std::min(n_samples, n_features_);
    if (n_components_ == 0 || n_components_ > max_components) {
        n_components_ = max_components;
    }

    // Center the data
    mean_ = X.colwise().mean();
    Matrix X_centered = X.rowwise() - mean_.transpose();

    // SVD decomposition: X = U * S * V^T
    // Principal components are rows of V^T (columns of V)
    Eigen::JacobiSVD<Matrix> svd(X_centered, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Store principal components (each row is a component)
    components_ = svd.matrixV().leftCols(n_components_).transpose();

    // Compute explained variance
    Vector singular_values = svd.singularValues().head(n_components_);
    explained_variance_ = singular_values.array().square() / (n_samples - 1);

    // Compute explained variance ratio
    double total_variance = (X_centered.array().square().sum()) / (n_samples - 1);
    explained_variance_ratio_ = explained_variance_ / total_variance;

    fitted_ = true;
    return *this;
}

Matrix PCA::transform(const Matrix& X) const {
    validateFitted();
    validateInput(X);

    // Center and project onto principal components
    Matrix X_centered = X.rowwise() - mean_.transpose();
    return X_centered * components_.transpose();
}

Matrix PCA::fitTransform(const Matrix& X) {
    fit(X);
    return transform(X);
}

Matrix PCA::inverseTransform(const Matrix& X_reduced) const {
    validateFitted();

    if (X_reduced.cols() != n_components_) {
        throw std::invalid_argument(
            "Input has " + std::to_string(X_reduced.cols()) +
            " features, expected " + std::to_string(n_components_));
    }

    // Project back and add mean
    return (X_reduced * components_).rowwise() + mean_.transpose();
}

void PCA::validateFitted() const {
    if (!fitted_) {
        throw std::runtime_error("PCA model has not been fitted. Call fit() first.");
    }
}

void PCA::validateInput(const Matrix& X) const {
    if (X.cols() != n_features_) {
        throw std::invalid_argument(
            "Input has " + std::to_string(X.cols()) +
            " features, expected " + std::to_string(n_features_));
    }
}

} // namespace ml
