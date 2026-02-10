/**
 * @file main.cpp
 * @brief Demo application for ML/DL metrics
 * @details Demonstrates SOLID-compliant metric evaluation
 */

#include "MetricEvaluator.hpp"
#include <iostream>
#include <vector>

using namespace metrics;

void demonstrateClassificationMetrics() {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Classification Metrics Demonstration" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // Sample binary classification data
    std::vector<int> y_true = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    std::vector<int> y_pred = {1, 0, 1, 0, 0, 1, 1, 0, 1, 0};
    std::vector<double> y_prob = {0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.6, 0.3, 0.85, 0.35};

    // Method 1: Using individual metrics
    std::cout << "\n--- Individual Metric Usage ---\n";
    Accuracy accuracy;
    Precision precision;
    Recall recall;
    F1Score f1;

    std::cout << "Accuracy:  " << accuracy.calculate(y_true, y_pred) << std::endl;
    std::cout << "Precision: " << precision.calculate(y_true, y_pred) << std::endl;
    std::cout << "Recall:    " << recall.calculate(y_true, y_pred) << std::endl;
    std::cout << "F1 Score:  " << f1.calculate(y_true, y_pred) << std::endl;

    // Method 2: Using evaluator with all metrics
    std::cout << "\n--- Using ClassificationEvaluator ---";
    ClassificationEvaluator evaluator;
    evaluator.addAllDefaultMetrics();
    evaluator.evaluateAndPrint(y_true, y_pred);

    // Method 3: Using factory
    std::cout << "\n--- Using MetricFactory ---\n";
    auto& factory = getMetricFactory();
    auto mcc = factory.createClassificationMetric("mcc");
    std::cout << mcc->name() << ": " << mcc->calculate(y_true, y_pred) << std::endl;

    // Method 4: Probabilistic metrics
    std::cout << "\n--- Probabilistic Metrics ---";
    ProbabilisticEvaluator probEval;
    probEval.addAllDefaultMetrics();
    probEval.evaluateAndPrint(y_true, y_prob);

    // Show confusion matrix
    std::cout << "\n--- Confusion Matrix ---\n";
    auto cm = ConfusionMatrixData::compute(y_true, y_pred);
    std::cout << "True Positives:  " << cm.truePositives << std::endl;
    std::cout << "True Negatives:  " << cm.trueNegatives << std::endl;
    std::cout << "False Positives: " << cm.falsePositives << std::endl;
    std::cout << "False Negatives: " << cm.falseNegatives << std::endl;
}

void demonstrateRegressionMetrics() {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Regression Metrics Demonstration" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // Sample regression data
    std::vector<double> y_true = {3.0, -0.5, 2.0, 7.0, 4.5, 6.0, 2.5, 1.0, 8.0, 5.5};
    std::vector<double> y_pred = {2.5, 0.0, 2.1, 7.8, 4.2, 5.5, 3.0, 1.5, 7.5, 5.0};

    // Method 1: Using individual metrics
    std::cout << "\n--- Individual Metric Usage ---\n";
    MeanSquaredError mse;
    RootMeanSquaredError rmse;
    MeanAbsoluteError mae;
    RSquared r2;

    std::cout << "MSE:  " << mse.calculate(y_true, y_pred) << std::endl;
    std::cout << "RMSE: " << rmse.calculate(y_true, y_pred) << std::endl;
    std::cout << "MAE:  " << mae.calculate(y_true, y_pred) << std::endl;
    std::cout << "R^2:  " << r2.calculate(y_true, y_pred) << std::endl;

    // Method 2: Using evaluator with all metrics
    std::cout << "\n--- Using RegressionEvaluator ---";
    RegressionEvaluator evaluator;
    evaluator.addAllDefaultMetrics();
    evaluator.evaluateAndPrint(y_true, y_pred);

    // Method 3: Using parameterized metrics
    std::cout << "\n--- Parameterized Metrics ---\n";
    HuberLoss huber05(0.5);
    HuberLoss huber20(2.0);
    QuantileLoss q25(0.25);
    QuantileLoss q75(0.75);
    AdjustedRSquared adjR2(3);

    std::cout << "Huber (delta=0.5): " << huber05.calculate(y_true, y_pred) << std::endl;
    std::cout << "Huber (delta=2.0): " << huber20.calculate(y_true, y_pred) << std::endl;
    std::cout << "Quantile (q=0.25): " << q25.calculate(y_true, y_pred) << std::endl;
    std::cout << "Quantile (q=0.75): " << q75.calculate(y_true, y_pred) << std::endl;
    std::cout << "Adjusted R^2 (p=3): " << adjR2.calculate(y_true, y_pred) << std::endl;
}

void demonstrateExtensibility() {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Extensibility Demonstration (OCP)" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // Custom metric: Balanced Accuracy
    class BalancedAccuracy : public IMetric<int> {
    public:
        double calculate(const std::vector<int>& y_true,
                         const std::vector<int>& y_pred) const override {
            Recall recall;
            Specificity specificity;
            return (recall.calculate(y_true, y_pred) +
                    specificity.calculate(y_true, y_pred)) / 2.0;
        }

        std::string name() const override { return "BalancedAccuracy"; }
        bool higherIsBetter() const override { return true; }
    };

    // Register custom metric with factory
    auto& factory = getMetricFactory();
    factory.registerClassificationMetric("balanced_accuracy",
        []() { return std::make_unique<BalancedAccuracy>(); });

    // Use the custom metric
    std::vector<int> y_true = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    std::vector<int> y_pred = {1, 0, 1, 0, 0, 1, 1, 0, 1, 0};

    auto balAcc = factory.createClassificationMetric("balanced_accuracy");
    std::cout << "\nCustom metric registered and used:\n";
    std::cout << balAcc->name() << ": " << balAcc->calculate(y_true, y_pred) << std::endl;

    // List all available metrics
    std::cout << "\nAvailable classification metrics:\n";
    for (const auto& name : factory.listClassificationMetrics()) {
        std::cout << "  - " << name << std::endl;
    }

    std::cout << "\nAvailable regression metrics:\n";
    for (const auto& name : factory.listRegressionMetrics()) {
        std::cout << "  - " << name << std::endl;
    }
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║     ML/DL Metrics Library - SOLID Design         ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";

    demonstrateClassificationMetrics();
    demonstrateRegressionMetrics();
    demonstrateExtensibility();

    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "SOLID Principles Applied:" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "S - Single Responsibility: Each metric class has one job\n";
    std::cout << "O - Open/Closed: Add metrics via factory, no modifications\n";
    std::cout << "L - Liskov Substitution: All metrics are interchangeable\n";
    std::cout << "I - Interface Segregation: IMetric, IProbabilisticMetric\n";
    std::cout << "D - Dependency Inversion: Depend on abstractions (IMetric)\n";

    return 0;
}
