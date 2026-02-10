#include "LDA.hpp"
#include <Eigen/Eigenvalues>
#include <set>
#include <algorithm>

namespace ml {

LDA::LDA(int n_components)
    : n_components_(n_components)
    , n_features_(0)
    , n_classes_(0)
    , fitted_(false) {
    if (n_components < 0) {
        throw std::invalid_argument("n_components must be non-negative");
    }
}

LDA& LDA::fit(const Matrix& X, const Labels& y) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("Input matrix cannot be empty");
    }
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }

    const int n_samples = static_cast<int>(X.rows());
    n_features_ = static_cast<int>(X.cols());

    // Count unique classes
    std::set<int> unique_labels(y.data(), y.data() + y.size());
    n_classes_ = static_cast<int>(unique_labels.size());

    if (n_classes_ < 2) {
        throw std::invalid_argument("LDA requires at least 2 classes");
    }

    // Determine actual number of components (max is n_classes - 1)
    int max_components = n_classes_ - 1;
    if (n_components_ == 0 || n_components_ > max_components) {
        n_components_ = max_components;
    }

    // Compute overall mean
    mean_ = X.colwise().mean();

    // Compute scatter matrices
    Matrix Sw, Sb;
    computeScatterMatrices(X, y, Sw, Sb);

    // Solve generalized eigenvalue problem: Sb * v = lambda * Sw * v
    // Equivalent to: Sw^(-1) * Sb * v = lambda * v
    // Use pseudo-inverse for numerical stability
    Eigen::JacobiSVD<Matrix> svd(Sw, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = 1e-10 * std::max(Sw.cols(), Sw.rows()) *
                       svd.singularValues().array().abs().maxCoeff();

    Matrix Sw_pinv = svd.matrixV() *
        (svd.singularValues().array().abs() > tolerance)
            .select(svd.singularValues().array().inverse(), 0)
            .matrix().asDiagonal() *
        svd.matrixU().transpose();

    Matrix target = Sw_pinv * Sb;

    // Eigendecomposition
    Eigen::EigenSolver<Matrix> solver(target);
    Vector eigenvalues = solver.eigenvalues().real();
    Matrix eigenvectors = solver.eigenvectors().real();

    // Sort eigenvectors by eigenvalues (descending)
    std::vector<std::pair<double, int>> eigen_pairs;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        eigen_pairs.emplace_back(eigenvalues(i), i);
    }
    std::sort(eigen_pairs.begin(), eigen_pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Select top n_components eigenvectors
    scalings_.resize(n_features_, n_components_);
    Vector sorted_eigenvalues(n_components_);

    for (int i = 0; i < n_components_; ++i) {
        int idx = eigen_pairs[i].second;
        scalings_.col(i) = eigenvectors.col(idx);
        sorted_eigenvalues(i) = std::max(0.0, eigen_pairs[i].first);
    }

    // Compute explained variance ratio
    double total = sorted_eigenvalues.sum();
    if (total > 0) {
        explained_variance_ratio_ = sorted_eigenvalues / total;
    } else {
        explained_variance_ratio_ = Vector::Zero(n_components_);
    }

    fitted_ = true;
    return *this;
}

void LDA::computeScatterMatrices(const Matrix& X, const Labels& y,
                                  Matrix& Sw, Matrix& Sb) {
    const int n_samples = static_cast<int>(X.rows());

    // Initialize scatter matrices
    Sw = Matrix::Zero(n_features_, n_features_);
    Sb = Matrix::Zero(n_features_, n_features_);

    // Compute class means and scatter matrices
    class_means_.clear();
    class_means_.resize(n_classes_, Vector::Zero(n_features_));
    std::vector<int> class_counts(n_classes_, 0);

    // First pass: compute class means
    for (int i = 0; i < n_samples; ++i) {
        int label = y(i);
        class_means_[label] += X.row(i).transpose();
        class_counts[label]++;
    }

    for (int c = 0; c < n_classes_; ++c) {
        if (class_counts[c] > 0) {
            class_means_[c] /= class_counts[c];
        }
    }

    // Second pass: compute within-class scatter (Sw)
    for (int i = 0; i < n_samples; ++i) {
        int label = y(i);
        Vector diff = X.row(i).transpose() - class_means_[label];
        Sw += diff * diff.transpose();
    }

    // Compute between-class scatter (Sb)
    for (int c = 0; c < n_classes_; ++c) {
        Vector diff = class_means_[c] - mean_;
        Sb += class_counts[c] * (diff * diff.transpose());
    }
}

Matrix LDA::transform(const Matrix& X) const {
    validateFitted();
    validateInput(X);

    // Center and project onto discriminant axes
    Matrix X_centered = X.rowwise() - mean_.transpose();
    return X_centered * scalings_;
}

Matrix LDA::fitTransform(const Matrix& X, const Labels& y) {
    fit(X, y);
    return transform(X);
}

void LDA::validateFitted() const {
    if (!fitted_) {
        throw std::runtime_error("LDA model has not been fitted. Call fit() first.");
    }
}

void LDA::validateInput(const Matrix& X) const {
    if (X.cols() != n_features_) {
        throw std::invalid_argument(
            "Input has " + std::to_string(X.cols()) +
            " features, expected " + std::to_string(n_features_));
    }
}

} // namespace ml
