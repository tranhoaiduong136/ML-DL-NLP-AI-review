#include "../algorithms/dimension_reduction/LDA.hpp"

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

// Test: Basic fit and transform with 2 classes
void testBasicTwoClasses() {
    ml::LDA lda(1);

    // Two well-separated classes
    Eigen::MatrixXd X(6, 2);
    X << 1.0, 1.0,
         1.5, 1.2,
         1.2, 0.8,
         5.0, 5.0,
         5.5, 5.2,
         5.2, 4.8;

    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    lda.fit(X, y);

    assert(lda.isFitted());
    assert(lda.nComponents() == 1); // max is n_classes - 1

    auto X_transformed = lda.transform(X);
    assert(X_transformed.rows() == 6);
    assert(X_transformed.cols() == 1);
}

// Test: fitTransform convenience method
void testFitTransform() {
    ml::LDA lda(1);

    Eigen::MatrixXd X(6, 3);
    X << 1.0, 2.0, 3.0,
         1.1, 2.1, 3.1,
         1.2, 1.9, 2.9,
         5.0, 6.0, 7.0,
         5.1, 6.1, 7.1,
         4.9, 5.9, 6.9;

    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    auto X_transformed = lda.fitTransform(X, y);

    assert(lda.isFitted());
    assert(X_transformed.rows() == 6);
    assert(X_transformed.cols() == 1);
}

// Test: Three classes gives max 2 components
void testThreeClasses() {
    ml::LDA lda(0); // Request max components

    Eigen::MatrixXd X(9, 3);
    X << 1.0, 1.0, 1.0,
         1.1, 1.1, 1.1,
         0.9, 0.9, 0.9,
         5.0, 1.0, 1.0,
         5.1, 1.1, 1.1,
         4.9, 0.9, 0.9,
         1.0, 5.0, 1.0,
         1.1, 5.1, 1.1,
         0.9, 4.9, 0.9;

    Eigen::VectorXi y(9);
    y << 0, 0, 0, 1, 1, 1, 2, 2, 2;

    lda.fit(X, y);

    assert(lda.nClasses() == 3);
    assert(lda.nComponents() == 2); // n_classes - 1
}

// Test: Class separation in transformed space
void testClassSeparation() {
    ml::LDA lda(1);

    // Two clearly separated classes
    Eigen::MatrixXd X(8, 2);
    X << 0.0, 0.0,
         0.1, 0.1,
         0.0, 0.1,
         0.1, 0.0,
         10.0, 10.0,
         10.1, 10.1,
         10.0, 10.1,
         10.1, 10.0;

    Eigen::VectorXi y(8);
    y << 0, 0, 0, 0, 1, 1, 1, 1;

    auto X_transformed = lda.fitTransform(X, y);

    // Class 0 samples should be well separated from class 1 samples
    double mean_class0 = 0.0;
    double mean_class1 = 0.0;
    for (int i = 0; i < 4; ++i) {
        mean_class0 += X_transformed(i, 0);
        mean_class1 += X_transformed(i + 4, 0);
    }
    mean_class0 /= 4;
    mean_class1 /= 4;

    // Classes should be separated
    assert(std::abs(mean_class0 - mean_class1) > 1.0);
}

// Test: Class means are computed correctly
void testClassMeans() {
    ml::LDA lda(1);

    Eigen::MatrixXd X(4, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         10.0, 20.0,
         30.0, 40.0;

    Eigen::VectorXi y(4);
    y << 0, 0, 1, 1;

    lda.fit(X, y);

    auto class_means = lda.classMeans();

    assert(class_means.size() == 2);
    assert(approxEqual(class_means[0](0), 2.0));  // mean of 1, 3
    assert(approxEqual(class_means[0](1), 3.0));  // mean of 2, 4
    assert(approxEqual(class_means[1](0), 20.0)); // mean of 10, 30
    assert(approxEqual(class_means[1](1), 30.0)); // mean of 20, 40
}

// Test: Overall mean
void testOverallMean() {
    ml::LDA lda(1);

    Eigen::MatrixXd X(4, 2);
    X << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0,
         7.0, 8.0;

    Eigen::VectorXi y(4);
    y << 0, 0, 1, 1;

    lda.fit(X, y);

    auto mean = lda.mean();
    assert(approxEqual(mean(0), 4.0)); // mean of 1,3,5,7
    assert(approxEqual(mean(1), 5.0)); // mean of 2,4,6,8
}

// Test: Explained variance ratio sums to 1
void testExplainedVarianceRatio() {
    ml::LDA lda(2);

    Eigen::MatrixXd X(12, 4);
    X.setRandom();
    X.block(0, 0, 4, 4) += Eigen::MatrixXd::Constant(4, 4, 0.0);
    X.block(4, 0, 4, 4) += Eigen::MatrixXd::Constant(4, 4, 5.0);
    X.block(8, 0, 4, 4) += Eigen::MatrixXd::Constant(4, 4, 10.0);

    Eigen::VectorXi y(12);
    y << 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2;

    lda.fit(X, y);

    auto ratio = lda.explainedVarianceRatio();
    double sum = ratio.sum();

    assert(sum > 0.0);
    assert(sum <= 1.0 + TOLERANCE);
}

// Test: Transform before fit throws
void testTransformBeforeFit() {
    ml::LDA lda(1);

    Eigen::MatrixXd X(4, 2);
    X.setRandom();

    bool exceptionThrown = false;
    try {
        lda.transform(X);
    } catch (const std::runtime_error&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Empty input throws
void testEmptyInput() {
    ml::LDA lda(1);

    Eigen::MatrixXd X(0, 0);
    Eigen::VectorXi y(0);

    bool exceptionThrown = false;
    try {
        lda.fit(X, y);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Mismatched X and y sizes throws
void testMismatchedSizes() {
    ml::LDA lda(1);

    Eigen::MatrixXd X(5, 2);
    X.setRandom();

    Eigen::VectorXi y(3); // Wrong size
    y << 0, 1, 0;

    bool exceptionThrown = false;
    try {
        lda.fit(X, y);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Single class throws
void testSingleClass() {
    ml::LDA lda(1);

    Eigen::MatrixXd X(4, 2);
    X.setRandom();

    Eigen::VectorXi y(4);
    y << 0, 0, 0, 0; // Only one class

    bool exceptionThrown = false;
    try {
        lda.fit(X, y);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Wrong feature count in transform throws
void testWrongFeatureCount() {
    ml::LDA lda(1);

    Eigen::MatrixXd X_train(6, 3);
    X_train.setRandom();
    X_train.block(0, 0, 3, 3) += Eigen::MatrixXd::Constant(3, 3, 0.0);
    X_train.block(3, 0, 3, 3) += Eigen::MatrixXd::Constant(3, 3, 5.0);

    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    lda.fit(X_train, y);

    Eigen::MatrixXd X_test(2, 4); // Wrong number of features
    X_test.setRandom();

    bool exceptionThrown = false;
    try {
        lda.transform(X_test);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Negative n_components throws
void testNegativeComponents() {
    bool exceptionThrown = false;
    try {
        ml::LDA lda(-1);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Method chaining
void testMethodChaining() {
    ml::LDA lda(1);

    Eigen::MatrixXd X(4, 2);
    X << 1.0, 1.0,
         1.1, 1.1,
         5.0, 5.0,
         5.1, 5.1;

    Eigen::VectorXi y(4);
    y << 0, 0, 1, 1;

    // fit() returns *this for chaining
    auto& ref = lda.fit(X, y);
    assert(&ref == &lda);
}

// Test: isFitted state
void testIsFittedState() {
    ml::LDA lda(1);

    assert(!lda.isFitted());

    Eigen::MatrixXd X(4, 2);
    X << 1.0, 1.0,
         1.1, 1.1,
         5.0, 5.0,
         5.1, 5.1;

    Eigen::VectorXi y(4);
    y << 0, 0, 1, 1;

    lda.fit(X, y);

    assert(lda.isFitted());
}

// Test: Components capped at n_classes - 1
void testComponentsCapped() {
    ml::LDA lda(100); // Request more than possible

    Eigen::MatrixXd X(6, 5);
    X.setRandom();
    X.block(0, 0, 3, 5) += Eigen::MatrixXd::Constant(3, 5, 0.0);
    X.block(3, 0, 3, 5) += Eigen::MatrixXd::Constant(3, 5, 5.0);

    Eigen::VectorXi y(6);
    y << 0, 0, 0, 1, 1, 1;

    lda.fit(X, y);

    assert(lda.nComponents() == 1); // Capped at n_classes - 1
}

// Test: Scalings matrix has correct dimensions
void testScalingsDimensions() {
    ml::LDA lda(2);

    Eigen::MatrixXd X(9, 4);
    X.setRandom();
    X.block(0, 0, 3, 4) += Eigen::MatrixXd::Constant(3, 4, 0.0);
    X.block(3, 0, 3, 4) += Eigen::MatrixXd::Constant(3, 4, 5.0);
    X.block(6, 0, 3, 4) += Eigen::MatrixXd::Constant(3, 4, 10.0);

    Eigen::VectorXi y(9);
    y << 0, 0, 0, 1, 1, 1, 2, 2, 2;

    lda.fit(X, y);

    auto scalings = lda.scalings();
    assert(scalings.rows() == 4);  // n_features
    assert(scalings.cols() == 2);  // n_components
}

// Test: Many classes
void testManyClasses() {
    ml::LDA lda(0); // Max components

    Eigen::MatrixXd X(20, 5);
    X.setRandom();
    for (int i = 0; i < 5; ++i) {
        X.block(i * 4, 0, 4, 5) += Eigen::MatrixXd::Constant(4, 5, i * 5.0);
    }

    Eigen::VectorXi y(20);
    for (int i = 0; i < 5; ++i) {
        y.segment(i * 4, 4).setConstant(i);
    }

    lda.fit(X, y);

    assert(lda.nClasses() == 5);
    assert(lda.nComponents() == 4); // n_classes - 1
}

// Test: Non-contiguous class labels
void testNonContiguousLabels() {
    ml::LDA lda(1);

    Eigen::MatrixXd X(6, 2);
    X << 1.0, 1.0,
         1.1, 1.1,
         1.2, 1.2,
         5.0, 5.0,
         5.1, 5.1,
         5.2, 5.2;

    Eigen::VectorXi y(6);
    y << 0, 0, 0, 5, 5, 5; // Non-contiguous labels (0 and 5)

    lda.fit(X, y);

    assert(lda.nClasses() == 2);
    assert(lda.isFitted());
}

} // anonymous namespace

int main() {
    std::cout << "========================================\n";
    std::cout << "         LDA Unit Tests\n";
    std::cout << "========================================\n\n";

    TestRunner runner;

    runner.run("Basic Two Classes", testBasicTwoClasses);
    runner.run("fitTransform", testFitTransform);
    runner.run("Three Classes", testThreeClasses);
    runner.run("Class Separation", testClassSeparation);
    runner.run("Class Means", testClassMeans);
    runner.run("Overall Mean", testOverallMean);
    runner.run("Explained Variance Ratio", testExplainedVarianceRatio);
    runner.run("Transform Before Fit", testTransformBeforeFit);
    runner.run("Empty Input", testEmptyInput);
    runner.run("Mismatched Sizes", testMismatchedSizes);
    runner.run("Single Class", testSingleClass);
    runner.run("Wrong Feature Count", testWrongFeatureCount);
    runner.run("Negative Components", testNegativeComponents);
    runner.run("Method Chaining", testMethodChaining);
    runner.run("isFitted State", testIsFittedState);
    runner.run("Components Capped", testComponentsCapped);
    runner.run("Scalings Dimensions", testScalingsDimensions);
    runner.run("Many Classes", testManyClasses);
    runner.run("Non-Contiguous Labels", testNonContiguousLabels);

    return runner.summary();
}
