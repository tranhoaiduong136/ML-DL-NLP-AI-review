#include "../algorithms/regression/LinearRegression.hpp"

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

// Test: Simple linear relationship y = 2x + 1
void testSimpleLinearFit() {
    ml::algorithms::LinearRegression model;
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0}; // y = 2x + 1

    model.fit(X, y);

    assert(model.isFitted());
    assert(approxEqual(model.getIntercept(), 1.0));
    assert(model.getCoefficients().size() == 1);
    assert(approxEqual(model.getCoefficients()[0], 2.0));
}

// Test: Prediction after fitting
void testPredict() {
    ml::algorithms::LinearRegression model;

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {2.0, 4.0, 6.0}; // y = 2x

    model.fit(X, y);

    std::vector<std::vector<double>> XTest = {{4.0}, {5.0}};
    auto predictions = model.predict(XTest);

    assert(predictions.size() == 2);
    assert(approxEqual(predictions[0], 8.0));
    assert(approxEqual(predictions[1], 10.0));
}

// Test: Multiple features
void testMultipleFeatures() {
    ml::algorithms::LinearRegression model;

    // y = 1 + 2*x1 + 3*x2
    std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 3.0}
    };
    std::vector<double> y = {6.0, 8.0, 9.0, 11.0, 16.0};

    model.fit(X, y);

    assert(model.isFitted());
    assert(model.getCoefficients().size() == 2);
    assert(approxEqual(model.getIntercept(), 1.0, 0.01));
    assert(approxEqual(model.getCoefficients()[0], 2.0, 0.01));
    assert(approxEqual(model.getCoefficients()[1], 3.0, 0.01));
}

// Test: R² score for perfect fit
void testScorePerfectFit() {
    ml::algorithms::LinearRegression model;

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0};

    model.fit(X, y);
    double r2 = model.score(X, y);

    assert(approxEqual(r2, 1.0));
}

// Test: R² score for imperfect fit
void testScoreImperfectFit() {
    ml::algorithms::LinearRegression model;

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    std::vector<double> y = {2.1, 3.9, 6.2, 7.8, 10.1}; // Noisy data

    model.fit(X, y);
    double r2 = model.score(X, y);

    // R² should be high but not perfect
    assert(r2 > 0.95);
    assert(r2 < 1.0);
}

// Test: Fit without intercept
void testNoIntercept() {
    ml::algorithms::LinearRegression model(false); // No intercept

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}};
    std::vector<double> y = {3.0, 6.0, 9.0, 12.0}; // y = 3x (passes through origin)

    model.fit(X, y);

    assert(approxEqual(model.getIntercept(), 0.0));
    assert(approxEqual(model.getCoefficients()[0], 3.0));
}

// Test: Predict before fit throws exception
void testPredictBeforeFit() {
    ml::algorithms::LinearRegression model;

    std::vector<std::vector<double>> X = {{1.0}, {2.0}};

    bool exceptionThrown = false;
    try {
        model.predict(X);
    } catch (const std::logic_error&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Empty input throws exception
void testEmptyInput() {
    ml::algorithms::LinearRegression model;

    std::vector<std::vector<double>> emptyX;
    std::vector<double> y = {1.0, 2.0};

    bool exceptionThrown = false;
    try {
        model.fit(emptyX, y);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Mismatched dimensions throw exception
void testMismatchedDimensions() {
    ml::algorithms::LinearRegression model;

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {1.0, 2.0}; // Different size

    bool exceptionThrown = false;
    try {
        model.fit(X, y);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Inconsistent feature count throws exception
void testInconsistentFeatures() {
    ml::algorithms::LinearRegression model;

    std::vector<std::vector<double>> X = {
        {1.0, 2.0},
        {3.0},       // Missing feature
        {4.0, 5.0}
    };
    std::vector<double> y = {1.0, 2.0, 3.0};

    bool exceptionThrown = false;
    try {
        model.fit(X, y);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: Wrong feature count in prediction throws exception
void testWrongPredictionFeatures() {
    ml::algorithms::LinearRegression model;

    // Use linearly independent features
    std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 3.0},
        {3.0, 2.0},
        {4.0, 5.0}
    };
    std::vector<double> y = {4.0, 11.0, 10.0, 19.0}; // y = 1 + 2*x1 + x2

    model.fit(X, y);

    std::vector<std::vector<double>> XTest = {{1.0}}; // Wrong feature count

    bool exceptionThrown = false;
    try {
        model.predict(XTest);
    } catch (const std::invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test: isFitted returns correct state
void testIsFitted() {
    ml::algorithms::LinearRegression model;

    assert(!model.isFitted());

    std::vector<std::vector<double>> X = {{1.0}, {2.0}};
    std::vector<double> y = {1.0, 2.0};

    model.fit(X, y);

    assert(model.isFitted());
}

// Test: Model can be move-constructed
void testMoveConstruction() {
    ml::algorithms::LinearRegression model1;

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {2.0, 4.0, 6.0};

    model1.fit(X, y);

    ml::algorithms::LinearRegression model2(std::move(model1));

    assert(model2.isFitted());
    assert(approxEqual(model2.getCoefficients()[0], 2.0));
}

// Test: Larger dataset
void testLargerDataset() {
    ml::algorithms::LinearRegression model;

    // Generate data: y = 0.5 + 1.5*x1 - 0.5*x2 + 2.0*x3
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    for (int i = 0; i < 100; ++i) {
        double x1 = static_cast<double>(i % 10);
        double x2 = static_cast<double>((i / 10) % 10);
        double x3 = static_cast<double>(i / 50);

        X.push_back({x1, x2, x3});
        y.push_back(0.5 + 1.5 * x1 - 0.5 * x2 + 2.0 * x3);
    }

    model.fit(X, y);

    assert(model.isFitted());
    assert(approxEqual(model.getIntercept(), 0.5, 0.01));
    assert(approxEqual(model.getCoefficients()[0], 1.5, 0.01));
    assert(approxEqual(model.getCoefficients()[1], -0.5, 0.01));
    assert(approxEqual(model.getCoefficients()[2], 2.0, 0.01));

    double r2 = model.score(X, y);
    assert(approxEqual(r2, 1.0, 0.001));
}

// Test: Interface polymorphism (IRegressor)
void testPolymorphism() {
    ml::algorithms::IRegressor* regressor = new ml::algorithms::LinearRegression();

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {2.0, 4.0, 6.0};

    regressor->fit(X, y);
    auto predictions = regressor->predict(X);

    assert(predictions.size() == 3);
    assert(approxEqual(predictions[0], 2.0));

    delete regressor;
}

// Test: ILinearModel interface
void testLinearModelInterface() {
    ml::algorithms::LinearRegression model;

    std::vector<std::vector<double>> X = {{1.0}, {2.0}};
    std::vector<double> y = {3.0, 5.0};

    model.fit(X, y);

    ml::algorithms::ILinearModel* linearModel = &model;

    assert(!linearModel->getCoefficients().empty());
    assert(approxEqual(linearModel->getIntercept(), 1.0));
}

// Test: Negative coefficients
void testNegativeCoefficients() {
    ml::algorithms::LinearRegression model;

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}};
    std::vector<double> y = {10.0, 8.0, 6.0, 4.0}; // y = 12 - 2x

    model.fit(X, y);

    assert(approxEqual(model.getIntercept(), 12.0));
    assert(approxEqual(model.getCoefficients()[0], -2.0));
}

// Test: Zero coefficients scenario
void testConstantTarget() {
    ml::algorithms::LinearRegression model;

    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}};
    std::vector<double> y = {5.0, 5.0, 5.0, 5.0}; // Constant y

    model.fit(X, y);

    // Coefficient should be ~0, intercept should be ~5
    assert(approxEqual(model.getCoefficients()[0], 0.0, 0.01));
    assert(approxEqual(model.getIntercept(), 5.0, 0.01));
}

} // anonymous namespace

int main() {
    std::cout << "========================================\n";
    std::cout << "   LinearRegression Unit Tests\n";
    std::cout << "========================================\n\n";

    TestRunner runner;

    runner.run("Simple Linear Fit", testSimpleLinearFit);
    runner.run("Predict", testPredict);
    runner.run("Multiple Features", testMultipleFeatures);
    runner.run("Score Perfect Fit", testScorePerfectFit);
    runner.run("Score Imperfect Fit", testScoreImperfectFit);
    runner.run("No Intercept", testNoIntercept);
    runner.run("Predict Before Fit", testPredictBeforeFit);
    runner.run("Empty Input", testEmptyInput);
    runner.run("Mismatched Dimensions", testMismatchedDimensions);
    runner.run("Inconsistent Features", testInconsistentFeatures);
    runner.run("Wrong Prediction Features", testWrongPredictionFeatures);
    runner.run("isFitted State", testIsFitted);
    runner.run("Move Construction", testMoveConstruction);
    runner.run("Larger Dataset", testLargerDataset);
    runner.run("Polymorphism (IRegressor)", testPolymorphism);
    runner.run("ILinearModel Interface", testLinearModelInterface);
    runner.run("Negative Coefficients", testNegativeCoefficients);
    runner.run("Constant Target", testConstantTarget);

    return runner.summary();
}
