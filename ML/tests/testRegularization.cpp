#include "../algorithms/regression/Regularization.hpp"
#include "../algorithms/regression/RegularizedLinearRegression.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>
#include <memory>

namespace {

constexpr double TOLERANCE = 1e-6;
constexpr double LOOSE_TOLERANCE = 0.1; // For regularized methods (includes regularization bias)

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

// =============================================================================
// NoRegularizer Tests
// =============================================================================

void testNoRegularizerPenalty() {
    ml::algorithms::NoRegularizer reg;
    std::vector<double> coeffs = {1.0, 2.0, 3.0};

    assert(approxEqual(reg.penalty(coeffs), 0.0));
    assert(reg.getLambda() == 0.0);
    assert(reg.supportsClosedForm());
    assert(reg.getName() == "None");
}

void testNoRegularizerGradient() {
    ml::algorithms::NoRegularizer reg;
    std::vector<double> coeffs = {1.0, -2.0, 3.0};

    auto grad = reg.gradient(coeffs);

    assert(grad.size() == 3);
    for (const auto& g : grad) {
        assert(approxEqual(g, 0.0));
    }
}

// =============================================================================
// L2Regularizer (Ridge) Tests
// =============================================================================

void testL2RegularizerConstruction() {
    ml::algorithms::L2Regularizer reg(1.0);

    assert(approxEqual(reg.getLambda(), 1.0));
    assert(reg.supportsClosedForm());
    assert(reg.getName() == "L2 (Ridge)");
}

void testL2RegularizerInvalidLambda() {
    bool exceptionThrown = false;
    try {
        ml::algorithms::L2Regularizer reg(-1.0);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }
    assert(exceptionThrown);
}

void testL2RegularizerPenalty() {
    ml::algorithms::L2Regularizer reg(2.0);
    std::vector<double> coeffs = {1.0, 2.0, 3.0};

    // Penalty = λ * Σβj² = 2 * (1 + 4 + 9) = 28
    assert(approxEqual(reg.penalty(coeffs), 28.0));
}

void testL2RegularizerGradient() {
    ml::algorithms::L2Regularizer reg(2.0);
    std::vector<double> coeffs = {1.0, -2.0, 3.0};

    // Gradient = 2λβ
    auto grad = reg.gradient(coeffs);

    assert(grad.size() == 3);
    assert(approxEqual(grad[0], 4.0));   // 2 * 2 * 1
    assert(approxEqual(grad[1], -8.0));  // 2 * 2 * (-2)
    assert(approxEqual(grad[2], 12.0));  // 2 * 2 * 3
}

void testL2RegularizerApplyToXtX() {
    ml::algorithms::L2Regularizer reg(0.5);

    std::vector<std::vector<double>> XtX = {
        {4.0, 1.0, 2.0},
        {1.0, 5.0, 3.0},
        {2.0, 3.0, 6.0}
    };

    // Without intercept - add λ to all diagonal
    reg.applyToXtX(XtX, false);

    assert(approxEqual(XtX[0][0], 4.5));
    assert(approxEqual(XtX[1][1], 5.5));
    assert(approxEqual(XtX[2][2], 6.5));
    assert(approxEqual(XtX[0][1], 1.0)); // Off-diagonal unchanged
}

void testL2RegularizerApplyToXtXWithIntercept() {
    ml::algorithms::L2Regularizer reg(0.5);

    std::vector<std::vector<double>> XtX = {
        {4.0, 1.0, 2.0},
        {1.0, 5.0, 3.0},
        {2.0, 3.0, 6.0}
    };

    // With intercept - don't regularize first element
    reg.applyToXtX(XtX, true);

    assert(approxEqual(XtX[0][0], 4.0)); // Unchanged (intercept)
    assert(approxEqual(XtX[1][1], 5.5));
    assert(approxEqual(XtX[2][2], 6.5));
}

void testL2RegularizerClone() {
    ml::algorithms::L2Regularizer reg(1.5);
    auto clone = reg.clone();

    assert(clone != nullptr);
    assert(approxEqual(clone->getLambda(), 1.5));
    assert(clone->supportsClosedForm());
}

// =============================================================================
// L1Regularizer (Lasso) Tests
// =============================================================================

void testL1RegularizerConstruction() {
    ml::algorithms::L1Regularizer reg(1.0);

    assert(approxEqual(reg.getLambda(), 1.0));
    assert(!reg.supportsClosedForm()); // Requires iterative methods
    assert(reg.getName() == "L1 (Lasso)");
}

void testL1RegularizerInvalidLambda() {
    bool exceptionThrown = false;
    try {
        ml::algorithms::L1Regularizer reg(-1.0);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }
    assert(exceptionThrown);
}

void testL1RegularizerPenalty() {
    ml::algorithms::L1Regularizer reg(2.0);
    std::vector<double> coeffs = {1.0, -2.0, 3.0};

    // Penalty = λ * Σ|βj| = 2 * (1 + 2 + 3) = 12
    assert(approxEqual(reg.penalty(coeffs), 12.0));
}

void testL1RegularizerGradient() {
    ml::algorithms::L1Regularizer reg(2.0);
    std::vector<double> coeffs = {1.0, -2.0, 0.0};

    // Subgradient = λ * sign(β)
    auto grad = reg.gradient(coeffs);

    assert(grad.size() == 3);
    assert(approxEqual(grad[0], 2.0));   // λ * 1
    assert(approxEqual(grad[1], -2.0));  // λ * (-1)
    assert(approxEqual(grad[2], 0.0));   // λ * 0 (at zero)
}

void testL1RegularizerSoftThreshold() {
    ml::algorithms::L1Regularizer reg(1.0);

    // S(z, γ) = sign(z) * max(|z| - γ, 0)
    assert(approxEqual(reg.softThreshold(5.0, 2.0), 3.0));
    assert(approxEqual(reg.softThreshold(-5.0, 2.0), -3.0));
    assert(approxEqual(reg.softThreshold(1.0, 2.0), 0.0));
    assert(approxEqual(reg.softThreshold(-1.0, 2.0), 0.0));
    assert(approxEqual(reg.softThreshold(0.0, 2.0), 0.0));
}

void testL1RegularizerClone() {
    ml::algorithms::L1Regularizer reg(2.5);
    auto clone = reg.clone();

    assert(clone != nullptr);
    assert(approxEqual(clone->getLambda(), 2.5));
    assert(!clone->supportsClosedForm());
}

// =============================================================================
// ElasticNetRegularizer Tests
// =============================================================================

void testElasticNetConstruction() {
    ml::algorithms::ElasticNetRegularizer reg(1.0, 0.5);

    assert(approxEqual(reg.getLambda(), 1.0));
    assert(approxEqual(reg.getAlpha(), 0.5));
    assert(approxEqual(reg.getL1Penalty(), 0.5));
    assert(approxEqual(reg.getL2Penalty(), 0.5));
    assert(!reg.supportsClosedForm()); // Has L1 component
}

void testElasticNetPureL1() {
    ml::algorithms::ElasticNetRegularizer reg(2.0, 1.0); // alpha = 1 is pure L1

    assert(!reg.supportsClosedForm());
    assert(approxEqual(reg.getL1Penalty(), 2.0));
    assert(approxEqual(reg.getL2Penalty(), 0.0));
}

void testElasticNetPureL2() {
    ml::algorithms::ElasticNetRegularizer reg(2.0, 0.0); // alpha = 0 is pure L2

    assert(reg.supportsClosedForm());
    assert(approxEqual(reg.getL1Penalty(), 0.0));
    assert(approxEqual(reg.getL2Penalty(), 2.0));
}

void testElasticNetInvalidParams() {
    bool exceptionThrown = false;

    // Invalid lambda
    try {
        ml::algorithms::ElasticNetRegularizer reg(-1.0, 0.5);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }
    assert(exceptionThrown);

    // Invalid alpha (too high)
    exceptionThrown = false;
    try {
        ml::algorithms::ElasticNetRegularizer reg(1.0, 1.5);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }
    assert(exceptionThrown);

    // Invalid alpha (negative)
    exceptionThrown = false;
    try {
        ml::algorithms::ElasticNetRegularizer reg(1.0, -0.1);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }
    assert(exceptionThrown);
}

void testElasticNetPenalty() {
    ml::algorithms::ElasticNetRegularizer reg(2.0, 0.5);
    std::vector<double> coeffs = {1.0, -2.0, 3.0};

    // L1 = 1 + 2 + 3 = 6
    // L2 = 1 + 4 + 9 = 14
    // Penalty = λ * (α * L1 + (1-α) * L2) = 2 * (0.5 * 6 + 0.5 * 14) = 2 * 10 = 20
    assert(approxEqual(reg.penalty(coeffs), 20.0));
}

void testElasticNetClone() {
    ml::algorithms::ElasticNetRegularizer reg(1.5, 0.7);
    auto clone = reg.clone();

    assert(clone != nullptr);
    assert(approxEqual(clone->getLambda(), 1.5));
}

// =============================================================================
// RegularizedLinearRegression Tests
// =============================================================================

void testRidgeRegressionSimple() {
    auto model = ml::algorithms::createRidgeRegression(0.1);

    // Simple linear data: y = 2x + 1
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

    model->fit(X, y);

    assert(model->isFitted());

    // Ridge should give close to OLS for small lambda
    assert(approxEqual(model->getIntercept(), 1.0, LOOSE_TOLERANCE));
    assert(approxEqual(model->getCoefficients()[0], 2.0, LOOSE_TOLERANCE));
}

void testRidgeRegressionShrinkage() {
    // With high lambda, coefficients should shrink toward zero
    auto modelLow = ml::algorithms::createRidgeRegression(0.001);
    auto modelHigh = ml::algorithms::createRidgeRegression(100.0);

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

    modelLow->fit(X, y);
    modelHigh->fit(X, y);

    // Higher lambda should result in smaller coefficient magnitude
    assert(std::abs(modelHigh->getCoefficients()[0]) < std::abs(modelLow->getCoefficients()[0]));
}

void testRidgeRegressionMultipleFeatures() {
    auto model = ml::algorithms::createRidgeRegression(0.01);

    // y = 1 + 2*x1 + 3*x2
    std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 3.0}
    };
    std::vector<double> y = {6.0, 8.0, 9.0, 11.0, 16.0};

    model->fit(X, y);

    assert(model->isFitted());
    assert(model->getCoefficients().size() == 2);
    assert(approxEqual(model->getIntercept(), 1.0, LOOSE_TOLERANCE));
    assert(approxEqual(model->getCoefficients()[0], 2.0, LOOSE_TOLERANCE));
    assert(approxEqual(model->getCoefficients()[1], 3.0, LOOSE_TOLERANCE));
}

void testLassoRegressionSimple() {
    auto model = ml::algorithms::createLassoRegression(0.01);

    // Simple linear data
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

    model->fit(X, y);

    assert(model->isFitted());
    assert(model->getIterations() > 0); // Should use coordinate descent
}

void testLassoRegressionSparsity() {
    auto model = ml::algorithms::createLassoRegression(1.0);

    // Data where one feature is irrelevant (random noise approximated as zero-contributing)
    // y = 2*x1 + 0*x2 (x2 is noise)
    std::vector<std::vector<double>> X = {
        {1.0, 0.5},
        {2.0, 0.3},
        {3.0, 0.8},
        {4.0, 0.1},
        {5.0, 0.9}
    };
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0}; // y = 2*x1

    model->fit(X, y);

    assert(model->isFitted());
    // With high lambda, Lasso should shrink irrelevant coefficient toward zero
    assert(std::abs(model->getCoefficients()[1]) < std::abs(model->getCoefficients()[0]));
}

void testElasticNetRegressionSimple() {
    auto model = ml::algorithms::createElasticNetRegression(0.01, 0.5);

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

    model->fit(X, y);

    assert(model->isFitted());
    assert(model->getIterations() > 0); // Uses coordinate descent (has L1 component)
}

void testRegularizedRegressionPredict() {
    auto model = ml::algorithms::createRidgeRegression(0.01);

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {2.0, 4.0, 6.0}; // y = 2x

    model->fit(X, y);

    std::vector<std::vector<double>> XTest = {{4.0}, {5.0}};
    auto predictions = model->predict(XTest);

    assert(predictions.size() == 2);
    assert(approxEqual(predictions[0], 8.0, LOOSE_TOLERANCE));
    assert(approxEqual(predictions[1], 10.0, LOOSE_TOLERANCE));
}

void testRegularizedRegressionScore() {
    auto model = ml::algorithms::createRidgeRegression(0.001);

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};

    model->fit(X, y);

    double r2 = model->score(X, y);
    assert(r2 > 0.99); // Should be nearly perfect fit
}

void testRegularizedRegressionPredictBeforeFit() {
    auto model = ml::algorithms::createRidgeRegression(0.1);

    std::vector<std::vector<double>> X = {{1.0}};

    bool exceptionThrown = false;
    try {
        model->predict(X);
    } catch (const std::logic_error&) {
        exceptionThrown = true;
    }
    assert(exceptionThrown);
}

void testRegularizedRegressionEmptyInput() {
    auto model = ml::algorithms::createRidgeRegression(0.1);

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    bool exceptionThrown = false;
    try {
        model->fit(X, y);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }
    assert(exceptionThrown);
}

void testRegularizedRegressionNoIntercept() {
    auto model = ml::algorithms::createRidgeRegression(0.01, false);

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0}; // y = 2x (passes through origin)

    model->fit(X, y);

    assert(model->isFitted());
    assert(approxEqual(model->getIntercept(), 0.0));
    assert(approxEqual(model->getCoefficients()[0], 2.0, LOOSE_TOLERANCE));
}

void testRegularizedRegressionSetRegularizer() {
    auto model = ml::algorithms::createRidgeRegression(0.1);

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {2.0, 4.0, 6.0};

    model->fit(X, y);
    assert(model->isFitted());

    // Change regularizer - should reset fitted state
    model->setRegularizer(std::make_unique<ml::algorithms::L1Regularizer>(0.5));
    assert(!model->isFitted());

    // Re-fit with new regularizer
    model->fit(X, y);
    assert(model->isFitted());
    assert(model->getIterations() > 0); // Now using coordinate descent
}

void testPolymorphismWithIRegressor() {
    std::unique_ptr<ml::algorithms::IRegressor> regressor =
        ml::algorithms::createRidgeRegression(0.1);

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {2.0, 4.0, 6.0};

    regressor->fit(X, y);
    auto predictions = regressor->predict(X);

    assert(predictions.size() == 3);
}

void testPolymorphismWithILinearModel() {
    std::unique_ptr<ml::algorithms::RegularizedLinearRegression> model =
        ml::algorithms::createRidgeRegression(0.01);

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {3.0, 5.0, 7.0}; // y = 2x + 1

    model->fit(X, y);

    ml::algorithms::ILinearModel* linearModel = model.get();
    auto coeffs = linearModel->getCoefficients();
    double intercept = linearModel->getIntercept();

    assert(coeffs.size() == 1);
    assert(approxEqual(coeffs[0], 2.0, LOOSE_TOLERANCE));
    assert(approxEqual(intercept, 1.0, LOOSE_TOLERANCE));
}

void testMoveSemantics() {
    auto model1 = ml::algorithms::createRidgeRegression(0.1);

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {2.0, 4.0, 6.0};

    model1->fit(X, y);
    assert(model1->isFitted());

    auto model2 = std::move(model1);
    assert(model2->isFitted());
}

void testLargeDataset() {
    auto model = ml::algorithms::createRidgeRegression(0.01);

    // Generate larger dataset: y = 1 + 2*x1 + 3*x2 + 0.5*x3
    const std::size_t numSamples = 100;
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    for (std::size_t i = 0; i < numSamples; ++i) {
        double x1 = static_cast<double>(i % 10);
        double x2 = static_cast<double>(i / 10);
        double x3 = static_cast<double>(i % 5);

        X.push_back({x1, x2, x3});
        y.push_back(1.0 + 2.0 * x1 + 3.0 * x2 + 0.5 * x3);
    }

    model->fit(X, y);

    assert(model->isFitted());
    assert(model->getCoefficients().size() == 3);

    double r2 = model->score(X, y);
    assert(r2 > 0.95);
}

} // anonymous namespace

int main() {
    TestRunner runner;

    // NoRegularizer tests
    runner.run("NoRegularizer penalty", testNoRegularizerPenalty);
    runner.run("NoRegularizer gradient", testNoRegularizerGradient);

    // L2Regularizer tests
    runner.run("L2Regularizer construction", testL2RegularizerConstruction);
    runner.run("L2Regularizer invalid lambda", testL2RegularizerInvalidLambda);
    runner.run("L2Regularizer penalty", testL2RegularizerPenalty);
    runner.run("L2Regularizer gradient", testL2RegularizerGradient);
    runner.run("L2Regularizer applyToXtX", testL2RegularizerApplyToXtX);
    runner.run("L2Regularizer applyToXtX with intercept", testL2RegularizerApplyToXtXWithIntercept);
    runner.run("L2Regularizer clone", testL2RegularizerClone);

    // L1Regularizer tests
    runner.run("L1Regularizer construction", testL1RegularizerConstruction);
    runner.run("L1Regularizer invalid lambda", testL1RegularizerInvalidLambda);
    runner.run("L1Regularizer penalty", testL1RegularizerPenalty);
    runner.run("L1Regularizer gradient", testL1RegularizerGradient);
    runner.run("L1Regularizer soft threshold", testL1RegularizerSoftThreshold);
    runner.run("L1Regularizer clone", testL1RegularizerClone);

    // ElasticNetRegularizer tests
    runner.run("ElasticNet construction", testElasticNetConstruction);
    runner.run("ElasticNet pure L1", testElasticNetPureL1);
    runner.run("ElasticNet pure L2", testElasticNetPureL2);
    runner.run("ElasticNet invalid params", testElasticNetInvalidParams);
    runner.run("ElasticNet penalty", testElasticNetPenalty);
    runner.run("ElasticNet clone", testElasticNetClone);

    // RegularizedLinearRegression tests
    runner.run("Ridge regression simple", testRidgeRegressionSimple);
    runner.run("Ridge regression shrinkage", testRidgeRegressionShrinkage);
    runner.run("Ridge regression multiple features", testRidgeRegressionMultipleFeatures);
    runner.run("Lasso regression simple", testLassoRegressionSimple);
    runner.run("Lasso regression sparsity", testLassoRegressionSparsity);
    runner.run("ElasticNet regression simple", testElasticNetRegressionSimple);
    runner.run("Regularized regression predict", testRegularizedRegressionPredict);
    runner.run("Regularized regression score", testRegularizedRegressionScore);
    runner.run("Regularized regression predict before fit", testRegularizedRegressionPredictBeforeFit);
    runner.run("Regularized regression empty input", testRegularizedRegressionEmptyInput);
    runner.run("Regularized regression no intercept", testRegularizedRegressionNoIntercept);
    runner.run("Regularized regression set regularizer", testRegularizedRegressionSetRegularizer);
    runner.run("Polymorphism with IRegressor", testPolymorphismWithIRegressor);
    runner.run("Polymorphism with ILinearModel", testPolymorphismWithILinearModel);
    runner.run("Move semantics", testMoveSemantics);
    runner.run("Large dataset", testLargeDataset);

    return runner.summary();
}
