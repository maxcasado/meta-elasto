# Meta-Elasto Skin Dataset: Classification & Analysis

This repository contains the research and development workflow for predicting skin-related classes using the **Meta-Elasto Skindataset**. The project explores various modeling techniques, ranging from traditional machine learning to deep learning architectures, to identify the most effective way to classify elastography-based data.

## 🎯 Project Overview
The primary goal of this project is to develop a robust classifier for skin data. Given the "Elasto" nature of the dataset, the models focus on interpreting tissue stiffness, temporal sequences, or spatial features to distinguish between different biological or diagnostic classes.

## 🔬 Modeling Processes & Experiments
We have explored several iterations and architectures to find the optimal balance between accuracy and complexity. Below is a summary of the processes tried:

### 1. Traditional Machine Learning (`ML models`)
*   **Approach:** We initially implemented standard machine learning baselines including Random Forest, Support Vector Machines (SVM), and Gradient Boosting.
*   **Result:** These models served as our baseline. However, they demonstrated **low accuracy** and were unable to capture the complex, non-linear patterns inherent in the skin elastography data.
*   **Status:** Deprecated in favor of deep learning approaches.

### 2. Refined ML Models (`Best ML models`)
*   **Approach:** A targeted attempt to optimize machine learning performance through rigorous feature engineering and hyperparameter tuning.
*   **Focus:** Identifying which specific features (e.g., statistical moments of the elasto-signals) provide the most predictive power.
*   **Result:** While performance improved significantly over the initial baselines, these models still lagged behind hybrid deep learning architectures.

### 3. Temporal Convolutional Networks (`Past TCN Work`)
*   **Approach:** Given that the dataset may contain sequential or time-dependent information, we experimented with **TCNs**.
*   **Goal:** To capture long-range dependencies in the data without the computational overhead of RNNs or LSTMs.
*   **Outcome:** These experiments provided valuable insights into the temporal nature of the dataset but ultimately led us toward a more spatial/feature-heavy approach.

### 4. Hybrid CNN + Feature Approach (`CNN + feature`) 🚀
*   **Approach:** This is our **best-performing model** to date. It utilizes a dual-input architecture:
    *   **CNN Branch:** Extracts spatial or high-level latent features directly from the raw data.
    *   **Feature Branch:** Integrates manually extracted domain-specific features (e.g., specific elasticity metrics).
*   **Result:** By fusing deep-learned features with hand-crafted domain knowledge, this model achieves the highest accuracy and most stable performance.
*   **Status:** **Current primary focus** for deployment and further refinement.

---

## 📂 Repository Structure

| Directory | Description |
| :--- | :--- |
| `📂 CNN + feature` | Notebooks and scripts for the top-performing hybrid model. |
| `📂 Best ML models` | Optimized traditional ML scripts and feature engineering logic. |
| `📂 ML models` | Early-stage baseline experiments and legacy ML code. |
| `📂 Past TCN Work` | Legacy code exploring temporal sequence modeling (TCNs). |
| `📂 ML reports` | Performance metrics, confusion matrices, and comparison reports. |

---

## 📈 Current Conclusion
The experimental results indicate that traditional machine learning is insufficient for the high-dimensional nature of this specific dataset. The complexity of skin-elasto signals requires the feature-extraction power of **Convolutional Neural Networks (CNNs)**. Furthermore, performance is significantly boosted when these automated features are combined with **hand-crafted domain features**, proving that a hybrid approach is the most effective for skin elastography classification.

---
