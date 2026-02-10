/**
 * @file ClassificationMetrics.hpp
 * @brief Classification metrics following SOLID principles
 * @details Single Responsibility: Each class handles one specific metric
 */

#ifndef CLASSIFICATION_METRICS_HPP
#define CLASSIFICATION_METRICS_HPP

#include "IMetric.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace metrics {

/**
 * @brief Confusion matrix components (SRP - Single data structure responsibility)
 */
struct ConfusionMatrixData {
    int truePositives = 0;
    int trueNegatives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;

    static ConfusionMatrixData compute(const std::vector<int>& y_true,
                                        const std::vector<int>& y_pred,
                                        int posLabel = 1) {
        ConfusionMatrixData cm;
        for (size_t i = 0; i < y_true.size(); ++i) {
            bool actualPos = (y_true[i] == posLabel);
            bool predPos = (y_pred[i] == posLabel);

            if (actualPos && predPos) cm.truePositives++;
            else if (!actualPos && !predPos) cm.trueNegatives++;
            else if (!actualPos && predPos) cm.falsePositives++;
            else cm.falseNegatives++;
        }
        return cm;
    }

    int total() const {
        return truePositives + trueNegatives + falsePositives + falseNegatives;
    }
};

/**
 * @brief Accuracy metric (SRP)
 * Accuracy = (TP + TN) / (TP + TN + FP + FN)
 */
class Accuracy : public IMetric<int> {
public:
    double calculate(const std::vector<int>& y_true,
                     const std::vector<int>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        int correct = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == y_pred[i]) correct++;
        }
        return static_cast<double>(correct) / y_true.size();
    }

    std::string name() const override { return "Accuracy"; }
    bool higherIsBetter() const override { return true; }
};

/**
 * @brief Precision metric (SRP)
 * Precision = TP / (TP + FP)
 */
class Precision : public IMetric<int> {
    int posLabel_;
public:
    explicit Precision(int posLabel = 1) : posLabel_(posLabel) {}

    double calculate(const std::vector<int>& y_true,
                     const std::vector<int>& y_pred) const override {
        auto cm = ConfusionMatrixData::compute(y_true, y_pred, posLabel_);
        int denominator = cm.truePositives + cm.falsePositives;

        if (denominator == 0) return 0.0;
        return static_cast<double>(cm.truePositives) / denominator;
    }

    std::string name() const override { return "Precision"; }
    bool higherIsBetter() const override { return true; }
};

/**
 * @brief Recall (Sensitivity/TPR) metric (SRP)
 * Recall = TP / (TP + FN)
 */
class Recall : public IMetric<int> {
    int posLabel_;
public:
    explicit Recall(int posLabel = 1) : posLabel_(posLabel) {}

    double calculate(const std::vector<int>& y_true,
                     const std::vector<int>& y_pred) const override {
        auto cm = ConfusionMatrixData::compute(y_true, y_pred, posLabel_);
        int denominator = cm.truePositives + cm.falseNegatives;

        if (denominator == 0) return 0.0;
        return static_cast<double>(cm.truePositives) / denominator;
    }

    std::string name() const override { return "Recall"; }
    bool higherIsBetter() const override { return true; }
};

/**
 * @brief Specificity (TNR) metric (SRP)
 * Specificity = TN / (TN + FP)
 */
class Specificity : public IMetric<int> {
    int posLabel_;
public:
    explicit Specificity(int posLabel = 1) : posLabel_(posLabel) {}

    double calculate(const std::vector<int>& y_true,
                     const std::vector<int>& y_pred) const override {
        auto cm = ConfusionMatrixData::compute(y_true, y_pred, posLabel_);
        int denominator = cm.trueNegatives + cm.falsePositives;

        if (denominator == 0) return 0.0;
        return static_cast<double>(cm.trueNegatives) / denominator;
    }

    std::string name() const override { return "Specificity"; }
    bool higherIsBetter() const override { return true; }
};

/**
 * @brief F1 Score metric (SRP)
 * F1 = 2 * (Precision * Recall) / (Precision + Recall)
 */
class F1Score : public IMetric<int> {
    int posLabel_;
public:
    explicit F1Score(int posLabel = 1) : posLabel_(posLabel) {}

    double calculate(const std::vector<int>& y_true,
                     const std::vector<int>& y_pred) const override {
        Precision precMetric(posLabel_);
        Recall recMetric(posLabel_);

        double prec = precMetric.calculate(y_true, y_pred);
        double rec = recMetric.calculate(y_true, y_pred);

        if (prec + rec == 0.0) return 0.0;
        return 2.0 * (prec * rec) / (prec + rec);
    }

    std::string name() const override { return "F1Score"; }
    bool higherIsBetter() const override { return true; }
};

/**
 * @brief F-Beta Score metric (OCP - configurable via beta parameter)
 * F_beta = (1 + beta^2) * (Precision * Recall) / (beta^2 * Precision + Recall)
 */
class FBetaScore : public IMetric<int> {
    double beta_;
    int posLabel_;
public:
    explicit FBetaScore(double beta = 1.0, int posLabel = 1)
        : beta_(beta), posLabel_(posLabel) {}

    double calculate(const std::vector<int>& y_true,
                     const std::vector<int>& y_pred) const override {
        Precision precMetric(posLabel_);
        Recall recMetric(posLabel_);

        double prec = precMetric.calculate(y_true, y_pred);
        double rec = recMetric.calculate(y_true, y_pred);

        double betaSq = beta_ * beta_;
        double denominator = betaSq * prec + rec;

        if (denominator == 0.0) return 0.0;
        return (1.0 + betaSq) * (prec * rec) / denominator;
    }

    std::string name() const override { return "FBetaScore"; }
    bool higherIsBetter() const override { return true; }
};

/**
 * @brief Matthews Correlation Coefficient (SRP)
 * MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
 */
class MatthewsCorrelationCoefficient : public IMetric<int> {
    int posLabel_;
public:
    explicit MatthewsCorrelationCoefficient(int posLabel = 1) : posLabel_(posLabel) {}

    double calculate(const std::vector<int>& y_true,
                     const std::vector<int>& y_pred) const override {
        auto cm = ConfusionMatrixData::compute(y_true, y_pred, posLabel_);

        double numerator = static_cast<double>(cm.truePositives) * cm.trueNegatives -
                           static_cast<double>(cm.falsePositives) * cm.falseNegatives;

        double denominator = std::sqrt(
            static_cast<double>(cm.truePositives + cm.falsePositives) *
            (cm.truePositives + cm.falseNegatives) *
            (cm.trueNegatives + cm.falsePositives) *
            (cm.trueNegatives + cm.falseNegatives)
        );

        if (denominator == 0.0) return 0.0;
        return numerator / denominator;
    }

    std::string name() const override { return "MCC"; }
    bool higherIsBetter() const override { return true; }
};

/**
 * @brief Cohen's Kappa coefficient (SRP)
 */
class CohensKappa : public IMetric<int> {
public:
    double calculate(const std::vector<int>& y_true,
                     const std::vector<int>& y_pred) const override {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        double n = static_cast<double>(y_true.size());

        // Observed agreement
        int correct = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == y_pred[i]) correct++;
        }
        double p_o = correct / n;

        // Expected agreement by chance
        std::vector<int> labels;
        for (int val : y_true) {
            if (std::find(labels.begin(), labels.end(), val) == labels.end())
                labels.push_back(val);
        }
        for (int val : y_pred) {
            if (std::find(labels.begin(), labels.end(), val) == labels.end())
                labels.push_back(val);
        }

        double p_e = 0.0;
        for (int label : labels) {
            double p_true = std::count(y_true.begin(), y_true.end(), label) / n;
            double p_pred = std::count(y_pred.begin(), y_pred.end(), label) / n;
            p_e += p_true * p_pred;
        }

        if (p_e == 1.0) return 1.0;
        return (p_o - p_e) / (1.0 - p_e);
    }

    std::string name() const override { return "CohensKappa"; }
    bool higherIsBetter() const override { return true; }
};

/**
 * @brief Binary Cross-Entropy / Log Loss (SRP)
 */
class BinaryCrossEntropy : public IProbabilisticMetric<double> {
    double epsilon_;
public:
    explicit BinaryCrossEntropy(double epsilon = 1e-15) : epsilon_(epsilon) {}

    double calculate(const std::vector<int>& y_true,
                     const std::vector<double>& y_prob) const override {
        if (y_true.size() != y_prob.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        double loss = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            double p = std::clamp(y_prob[i], epsilon_, 1.0 - epsilon_);
            loss -= y_true[i] * std::log(p) + (1 - y_true[i]) * std::log(1.0 - p);
        }
        return loss / y_true.size();
    }

    std::string name() const override { return "BinaryCrossEntropy"; }
};

/**
 * @brief ROC-AUC Score (SRP)
 */
class RocAucScore : public IProbabilisticMetric<double> {
public:
    double calculate(const std::vector<int>& y_true,
                     const std::vector<double>& y_prob) const override {
        if (y_true.size() != y_prob.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        // Create sorted indices by probability (descending)
        std::vector<size_t> indices(y_true.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&y_prob](size_t a, size_t b) { return y_prob[a] > y_prob[b]; });

        int nPos = std::count(y_true.begin(), y_true.end(), 1);
        int nNeg = static_cast<int>(y_true.size()) - nPos;

        if (nPos == 0 || nNeg == 0) return 0.0;

        // Calculate AUC using trapezoidal rule
        double auc = 0.0;
        int tp = 0, fp = 0;
        double tprPrev = 0.0, fprPrev = 0.0;

        for (size_t idx : indices) {
            if (y_true[idx] == 1) {
                tp++;
            } else {
                fp++;
            }

            double tpr = static_cast<double>(tp) / nPos;
            double fpr = static_cast<double>(fp) / nNeg;

            auc += (fpr - fprPrev) * (tpr + tprPrev) / 2.0;
            tprPrev = tpr;
            fprPrev = fpr;
        }

        return auc;
    }

    std::string name() const override { return "ROC-AUC"; }
};

/**
 * @brief Average Precision Score (SRP)
 */
class AveragePrecisionScore : public IProbabilisticMetric<double> {
public:
    double calculate(const std::vector<int>& y_true,
                     const std::vector<double>& y_prob) const override {
        if (y_true.size() != y_prob.size() || y_true.empty()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        std::vector<size_t> indices(y_true.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&y_prob](size_t a, size_t b) { return y_prob[a] > y_prob[b]; });

        int nPos = std::count(y_true.begin(), y_true.end(), 1);
        if (nPos == 0) return 0.0;

        double ap = 0.0;
        int tp = 0;

        for (size_t i = 0; i < indices.size(); ++i) {
            if (y_true[indices[i]] == 1) {
                tp++;
                double precisionAtK = static_cast<double>(tp) / (i + 1);
                ap += precisionAtK;
            }
        }

        return ap / nPos;
    }

    std::string name() const override { return "AveragePrecision"; }
};

} // namespace metrics

#endif // CLASSIFICATION_METRICS_HPP
