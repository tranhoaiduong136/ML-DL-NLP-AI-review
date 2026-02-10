#ifndef PCA_HPP
#define PCA_HPP

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

namespace ml {

/**
 * Principal Component Analysis (PCA)
 *
 * Reduces dimensionality by projecting data onto principal components
 * that capture the maximum variance.
 *
 * Usage:
 *   PCA pca(2);                    // Keep 2 components
 *   pca.fit(X);                    // Fit on training data
 *   auto X_reduced = pca.transform(X);  // Transform data
 *   // Or combine: auto X_reduced = pca.fitTransform(X);
 */
class PCA {
public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    /**
     * @param n_components Number of principal components to keep.
     *                     If 0, keeps all components.
     */
    explicit PCA(int n_components = 0);

    /**
     * Fit the PCA model to training data.
     * @param X Data matrix (n_samples x n_features)
     * @return Reference to this for method chaining
     */
    PCA& fit(const Matrix& X);

    /**
     * Transform data to reduced dimensionality.
     * @param X Data matrix (n_samples x n_features)
     * @return Transformed matrix (n_samples x n_components)
     */
    Matrix transform(const Matrix& X) const;

    /**
     * Fit and transform in one step.
     * @param X Data matrix (n_samples x n_features)
     * @return Transformed matrix (n_samples x n_components)
     */
    Matrix fitTransform(const Matrix& X);

    /**
     * Reconstruct data from reduced representation.
     * @param X_reduced Reduced matrix (n_samples x n_components)
     * @return Reconstructed matrix (n_samples x n_features)
     */
    Matrix inverseTransform(const Matrix& X_reduced) const;

    // Accessors
    const Matrix& components() const { return components_; }
    const Vector& explainedVariance() const { return explained_variance_; }
    const Vector& explainedVarianceRatio() const { return explained_variance_ratio_; }
    const Vector& mean() const { return mean_; }
    int nComponents() const { return n_components_; }
    bool isFitted() const { return fitted_; }

private:
    void validateFitted() const;
    void validateInput(const Matrix& X) const;

    int n_components_;
    int n_features_;
    bool fitted_;

    Matrix components_;           // Principal components (n_components x n_features)
    Vector explained_variance_;   // Variance explained by each component
    Vector explained_variance_ratio_;  // Proportion of variance explained
    Vector mean_;                 // Feature means for centering
};

} // namespace ml

#endif // PCA_HPP
