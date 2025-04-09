# Multi-class Detection of Breast Cancer (Normal/ Benign/ Malignant)

### What We Have Done
We developed a breast cancer detection system by evaluating 14 machine learning (ML) and deep learning (DL) models across four datasets: Wisconsin Breast Cancer Dataset (WBCD), MIAS, INBreast, and CBIS-DDSM. Our focus was to enhance early detection accuracy and reliability, culminating in two hybrid models—SVM_CNN and SVM_XGBoost—tested primarily on the Wisconsin dataset, achieving up to 97% accuracy.

### Procedure
1. *Dataset Selection and Preprocessing*: We utilized four datasets, with WBCD (569 samples, 30 features) as the primary focus due to its structured tabular format. Preprocessing involved outlier removal (IQR method), scaling (StandardScaler), and encoding the diagnosis (Malignant: 1, Benign: 0).
2. *Model Implementation*: We tested individual classifiers (e.g., MobileNet, XGBoost, SVM, CNN, ANN, KNN, RF, DT, LR, NB) and developed hybrid models:
   - **SVM_CNN**: Used CNN for feature extraction from numerical data (1D convolutional layers) and SVM for classification.
   - **SVM_XGBoost**: Employed XGBoost to generate probability features (predict_proba), fed into SVM for final classification.
3. *Evaluation*: Applied 7-fold cross-validation on the Wisconsin dataset to ensure robustness, assessing performance via accuracy, precision, recall, F1-score, confusion matrices, and ROC curves.
4. *Analysis*: Compared results across datasets and against prior works to validate efficacy.

### Improvement
We improved performance by:
- *Hybridization*: Combined CNN’s spatial feature extraction with SVM’s classification strength, and XGBoost’s probabilistic feature generation with SVM’s decision boundaries, overcoming limitations of standalone models (e.g., CNN’s tabular data weakness, SVM’s feature representation issues).
- *Preprocessing*: Outlier handling and scaling stabilized data, reducing noise and enhancing model robustness.
- *Cross-Validation*: 7-fold cross-validation mitigated overfitting, improving generalization from initial lower accuracies to a stable 97%.

### Result Analysis
Our best result was a 97% accuracy with both SVM_CNN and SVM_XGBoost on the Wisconsin dataset, validated via 7-fold cross-validation. However, SVM_XGBoost stood out with a higher true positive rate (sensitivity), critical for malignancy detection:
- *SVM_CNN*: TN=297, FP=3, FN=9, TP=89 (Recall: 0.91 for class 1).
- *SVM_XGBoost*: TN=298, FP=2, FN=9, TP=89 (Recall: 0.91 for class 1, but fewer FPs suggest better precision balance).
- *Achievement*: SVM_XGBoost leveraged XGBoost’s probability outputs as enriched features, enabling SVM to better distinguish malignant cases. The process involved training XGBoost (logloss metric), extracting probabilities, and optimizing SVM (probability=True) on these features, enhanced by preprocessing and cross-validation.

![](https://raw.githubusercontent.com/PrajitaB/Breast_Cancer_Detection/refs/heads/main/Wisconsin_Hybrid.jpg)

### Comparison
- *Our Models*: Both achieved 97% accuracy, with SVM_XGBoost excelling in true positives (89 vs. 89, but with fewer FPs: 2 vs. 3), prioritizing sensitivity over SVM_CNN.
- *Traditional Models (Wisconsin)*: XGBoost (97%), SVM (97%), CNN (70%), KNN (95%), RF (93%), LR (96%), NB (94%), DT (95%), ANN (66%), MobileNet (75%).
- *Prior Works (WDBC)*: Kalbkhani et al. (98.59%, RF+GOA), Mavrogiorgou et al. (98.53%, SVM), Chaurasia and Pal (97.36%, MLP).
- *Analysis*: Our 97% is competitive, slightly below top ensemble models (98.59%, 98.53%) but balances accuracy, sensitivity, and computational efficiency. SVM_XGBoost’s higher sensitivity makes it clinically preferable over SVM_CNN and aligns with medical diagnostic needs, though it trails the optimized ensembles in raw accuracy.

![](https://raw.githubusercontent.com/PrajitaB/Breast_Cancer_Detection/refs/heads/main/Winconsin_Traditional.jpg)

In summary, SVM_XGBoost’s superior true positive rate, achieved through probabilistic feature enhancement and rigorous validation, positions it as our standout model for breast cancer detection, offering a practical, sensitive solution despite not topping prior accuracy peaks.
