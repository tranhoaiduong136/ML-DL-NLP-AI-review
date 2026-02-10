#ifndef LDA_HPP
#define LDA_HPP

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

namespace ml {

/**
 * Linear Discriminant Analysis (LDA)
 *
 * Supervised dimensionality reduction that maximizes class separability.
 * Projects data onto axes that maximize between-class variance while
 * minimizing within-class variance.
 *
 * Usage:
 *   LDA lda(2);                         // Reduce to 2 dimensions
 *   lda.fit(X, y);                      // Fit with labels
 *   auto X_reduced = lda.transform(X);  // Transform data
 *   // Or combine: auto X_reduced = lda.fitTransform(X, y);
 */
class LDA {
public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using Labels = Eigen::VectorXi;

    /**
     * @param n_components Number of discriminant components to keep.
     *                     Maximum is (n_classes - 1). If 0, uses maximum.
     */
    explicit LDA(int n_components = 0);

    /**
     * Fit the LDA model to labeled training data.
     * @param X Data matrix (n_samples x n_features)
     * @param y Class labels (n_samples), must be integers 0 to n_classes-1
     * @return Reference to this for method chaining
     */
    LDA& fit(const Matrix& X, const Labels& y);

    /**
     * Transform data to reduced dimensionality.
     * @param X Data matrix (n_samples x n_features)
     * @return Transformed matrix (n_samples x n_components)
     */
    Matrix transform(const Matrix& X) const;

    /**
     * Fit and transform in one step.
     * @param X Data matrix (n_samples x n_features)
     * @param y Class labels (n_samples)
     * @return Transformed matrix (n_samples x n_components)
     */
    Matrix fitTransform(const Matrix& X, const Labels& y);

    // Accessors
    const Matrix& scalings() const { return scalings_; }
    const Vector& explainedVarianceRatio() const { return explained_variance_ratio_; }
    const Vector& mean() const { return mean_; }
    const std::vector<Vector>& classMeans() const { return class_means_; }
    int nComponents() const { return n_components_; }
    int nClasses() const { return n_classes_; }
    bool isFitted() const { return fitted_; }

private:
    void validateFitted() const;
    void validateInput(const Matrix& X) const;
    void computeScatterMatrices(const Matrix& X, const Labels& y,
                                 Matrix& Sw, Matrix& Sb);

    int n_components_;
    int n_features_;
    int n_classes_;
    bool fitted_;

    Matrix scalings_;             // Projection matrix (n_features x n_components)
    Vector explained_variance_ratio_;  // Proportion of discriminant info
    Vector mean_;                 // Overall mean for centering
    std::vector<Vector> class_means_;  // Mean of each class
};

} // namespace ml

#endif // LDA_HPP
