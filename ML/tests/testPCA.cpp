#include "../algorithms/dimension_reduction/PCA.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>

namespace {

constexpr double TOLERANCE = 1e-6;

bool approxEqual(double a, double b, double tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

bool matrixApproxEqual(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, double tol = TOLERANCE) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    return (A - B).cwiseAbs().maxCoeff() < tol;
}

class TestRunner {
public:
    void run(const std::string& name, void (*testFunc)()) {
        std::cout << "Running: " << name << "... ";
        try {
            testFunc();
            std::cout << "PASSED\n";
            ++passed_;
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << "\n";
            ++failed_;
        } catch (...) {
            std::cout << "FAILED: Unknown exception\n";
            ++failed_;
        }
        ++total_;
    }

    int summary() const {
        std::cout << "\n========================================\n";
        std::cout << "Test Results: " << passed_ << "/" << total_ << " passed";
        if (failed_ > 0) {
            std::cout << " (" << failed_ << " failed)";
        }
        std::cout << "\n========================================\n";
        return (failed_ == 0) ? 0 : 1;
    }

private:
    int passed_ = 0;
    int failed_ = 0;
    int total_ = 0;
};

// Test: Basic fit and transform
void testBasicFitTransform() {
    ml::PCA pca(2);

    Eigen::MatrixXd X(5, 3);
    X << 1.0, 2.0, 3.0,
         2.0, 4.0, 6.0,
         3.0, 6.0, 9.0,
         4.0, 8.0, 12.0,
         5.0, 10.0, 15.0;

    pca.fit(X);

    assert(pca.isFitted());
    assert(pca.nComponents() == 2);

    auto X_transformed = pca.transform(X);
    assert(X_transformed.rows() == 5);
    assert(X_transformed.cols() == 2);
}

// Test: fitTransform convenience method
void testFitTransform() {
    ml::PCA pca(2);

    Eigen::MatrixXd X(4, 3);
    X << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0,
         1.0, 1.0, 1.0;

    auto X_transformed = pca.fitTransform(X);

    assert(pca.isFitted());
    assert(X_transformed.rows() == 4);
    assert(X_transformed.cols() == 2);
}

// Test: Explained variance sums to <= 1
void testExplainedVarianceRatio() {
    ml::PCA pca(3);

    Eigen::MatrixXd X(10, 5);
    X.setRandom();

    pca.fit(X);

    auto ratio = pca.explainedVarianceRatio();
    double sum = ratio.sum();

    assert(sum > 0.0);
    assert(sum <= 1.0 + TOLERANCE);

    // Ratios should be in descending order
    for (int i = 1; i < ratio.size(); ++i) {
        assert(ratio(i - 1) >= ratio(i) - TOLERANCE);
    }
}

// Test: Inverse transform reconstructs data
void testInverseTransform() {
    ml::PCA pca(0); // Keep all components

    Eigen::MatrixXd X(4, 3);
    X << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 9.0,
         10.0, 11.0, 12.0;

    pca.fit(X);

    auto X_transformed = pca.transform(X);
    auto X_reconstructed = pca.inverseTransform(X_transformed);

    assert(matrixApproxEqual(X, X_reconstructed, 1e-10));
}

// Test: Reduced components lose some info
void testDimensionalityReduction() {
    ml::PCA pca(1); // Keep only 1 component

    Eigen::MatrixXd X(5, 3);
    X << 1.0, 2.0, 1.5,
         2.0, 3.0, 2.8,
         3.0, 5.0, 4.1,
         4.0, 6.0, 5.2,
         5.0, 8.0, 6.3;

    pca.fit(X);

    auto X_transformed = pca.transform(X);
    assert(X_transformed.cols() == 1);

    auto X_reconstructed = pca.inverseTransform(X_transformed);

    // Reconstruction won't be perfect with 1 component
    double reconstruction_error = (X - X_reconstructed).norm();
    assert(reconstruction_error > 0.0);
}

// Test: Mean centering
void testMeanCentering() {
    ml::PCA pca(2);

    Eigen::MatrixXd X(3, 2);
    X << 1.0, 4.0,
         2.0, 5.0,
         3.0, 6.0;

    pca.fit(X);

    auto mean = pca.mean();
    assert(approxEqual(mean(0), 2.0));
    assert(approxEqual(mean(1), 5.0));
}

// Test: Components are orthonormal
void testComponentsOrthonormal() {
    ml::PCA pca(3);

    Eigen::MatrixXd X(20, 5);
    X.setRandom();

    pca.fit(X);

    auto components = pca.components();
    auto identity = components * components.transpose();

    Eigen::MatrixXd expected = Eigen::MatrixXd::Identity(3, 3);
    assert(matrixApproxEqual(identity, expected, 1e-10));
}

// Test: n_components = 0 keeps all
void testKeepAllComponents() {
    ml::PCA pca(0);

    Eigen::MatrixXd X(10, 4);
    X.setRandom();

    pca.fit(X);

    assert(pca.nComponents() == 4); // min(10 samples, 4 features)
}

// Test: n_components capped at min(samples, features)
void testComponentsCapped() {
    ml::PCA pca(100); // Request more than possible

    Eigen::MatrixXd X(5, 3);
    X.setRandom();

    pca.fit(X);

    assert(pca.nComponents() == 3); // Capped at n_features
}

// Test: Transform before fit throws
void testTransformBeforeFit() {
    ml::PCA pca(2);

    Eigen::MatrixXd X(3, 2);
    X.setRandom();

    bool exceptionThrown = false;
    try {
        pca.transform(X);
    } catch (const std::runtime_error&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Empty input throws
void testEmptyInput() {
    ml::PCA pca(2);

    Eigen::MatrixXd X(0, 0);

    bool exceptionThrown = false;
    try {
        pca.fit(X);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Wrong feature count in transform throws
void testWrongFeatureCount() {
    ml::PCA pca(2);

    Eigen::MatrixXd X_train(5, 3);
    X_train.setRandom();
    pca.fit(X_train);

    Eigen::MatrixXd X_test(3, 4); // Wrong number of features
    X_test.setRandom();

    bool exceptionThrown = false;
    try {
        pca.transform(X_test);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Negative n_components throws
void testNegativeComponents() {
    bool exceptionThrown = false;
    try {
        ml::PCA pca(-1);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Method chaining
void testMethodChaining() {
    ml::PCA pca(2);

    Eigen::MatrixXd X(5, 3);
    X.setRandom();

    // fit() returns *this for chaining
    auto& ref = pca.fit(X);
    assert(&ref == &pca);
}

// Test: isFitted state
void testIsFittedState() {
    ml::PCA pca(2);

    assert(!pca.isFitted());

    Eigen::MatrixXd X(5, 3);
    X.setRandom();
    pca.fit(X);

    assert(pca.isFitted());
}

// Test: Collinear data
void testCollinearData() {
    ml::PCA pca(2);

    // All columns are multiples of each other
    Eigen::MatrixXd X(5, 3);
    for (int i = 0; i < 5; ++i) {
        X(i, 0) = i + 1;
        X(i, 1) = 2 * (i + 1);
        X(i, 2) = 3 * (i + 1);
    }

    pca.fit(X);

    // First component should explain nearly all variance
    auto ratio = pca.explainedVarianceRatio();
    assert(ratio(0) > 0.99);
}

// Test: Single sample
void testSingleSample() {
    ml::PCA pca(1);

    Eigen::MatrixXd X(1, 3);
    X << 1.0, 2.0, 3.0;

    pca.fit(X);

    // Should handle gracefully
    assert(pca.isFitted());
}

// Test: Explained variance non-negative
void testExplainedVarianceNonNegative() {
    ml::PCA pca(3);

    Eigen::MatrixXd X(10, 5);
    X.setRandom();

    pca.fit(X);

    auto variance = pca.explainedVariance();
    for (int i = 0; i < variance.size(); ++i) {
        assert(variance(i) >= -TOLERANCE);
    }
}

} // anonymous namespace

int main() {
    std::cout << "========================================\n";
    std::cout << "         PCA Unit Tests\n";
    std::cout << "========================================\n\n";

    TestRunner runner;

    runner.run("Basic Fit Transform", testBasicFitTransform);
    runner.run("fitTransform", testFitTransform);
    runner.run("Explained Variance Ratio", testExplainedVarianceRatio);
    runner.run("Inverse Transform", testInverseTransform);
    runner.run("Dimensionality Reduction", testDimensionalityReduction);
    runner.run("Mean Centering", testMeanCentering);
    runner.run("Components Orthonormal", testComponentsOrthonormal);
    runner.run("Keep All Components", testKeepAllComponents);
    runner.run("Components Capped", testComponentsCapped);
    runner.run("Transform Before Fit", testTransformBeforeFit);
    runner.run("Empty Input", testEmptyInput);
    runner.run("Wrong Feature Count", testWrongFeatureCount);
    runner.run("Negative Components", testNegativeComponents);
    runner.run("Method Chaining", testMethodChaining);
    runner.run("isFitted State", testIsFittedState);
    runner.run("Collinear Data", testCollinearData);
    runner.run("Single Sample", testSingleSample);
    runner.run("Explained Variance Non-Negative", testExplainedVarianceNonNegative);

    return runner.summary();
}
