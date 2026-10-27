# HCL Technical Assessment: Interview Preparation Guide
## Credit Card Fraud Detection Using Anomaly Detection Techniques

---

## 🎙️ 1. The 60-Second Elevator Pitch
> *"In this project, I developed an end-to-end Machine Learning anomaly detection system specifically tailored to identify fraudulent credit card transactions under extreme class imbalance. In financial datasets like the European credit card benchmark, fraud represents less than 0.2% of all transactions. Supervised classifiers fail here because a naive model predicting 100% legitimate transactions gets 99.8% accuracy while missing every single fraud—a phenomenon known as the **Accuracy Paradox**.*
>
> *To overcome this, I reframed fraud detection into an **unsupervised and semi-supervised anomaly detection** problem. I implemented and benchmarked three algorithms: **Isolation Forest** as our primary sublinear detector, **Local Outlier Factor (LOF)** for density-based neighborhood modeling, and **One-Class SVM** for kernel boundary estimation. To guarantee zero data leakage, I built a `RobustScaler` pipeline that scales skewed transaction amounts and timestamps using interquartile ranges strictly on training splits. Finally, I deployed an interactive **Streamlit dashboard** that provides real-time transaction risk scoring, dynamic threshold tuning for Precision-Recall optimization, and batch CSV screening with enriched audit exports."*

---

## 🧠 2. Simple, Intuitive Explanations of All Algorithms

### A. Isolation Forest (Primary Model)
- **Analogy**: Imagine playing a guessing game where you try to isolate an animal using "yes/no" questions. If you are trying to isolate a common animal (like a domestic cat), it takes dozens of specific questions because there are many similar pets. But if someone says "a flying blue whale", you can isolate it with just one question: *"Is it over 50 feet long and flies?"*
- **Mechanism**: Isolation Forest builds an ensemble of random binary decision trees (iTrees). At each step, it picks a random feature and a random split value between that feature's minimum and maximum.
- **Core Truth**: Normal points sit in dense clusters and require many cuts to isolate (deep in the tree). Anomalies are few and far away, so they get isolated in very few cuts (near the root of the tree, small path length).
- **Why It’s Best**: It has a time complexity of $O(n \log n)$, requiring very little memory and running fast enough for high-throughput banking systems.

### B. Local Outlier Factor (LOF)
- **Analogy**: Imagine people living in residential apartment complexes. A normal apartment building has high density; everyone's nearest neighbor is 10 feet away. But if someone builds a lone cabin 5 miles away in the woods, their nearest neighbor is 5 miles away. LOF compares your distance to your neighbors against your neighbors' distance to their neighbors.
- **Mechanism**: For every transaction, LOF calculates the **Local Reachability Density (LRD)** based on its $k$-nearest neighbors.
- **Core Truth**: If a point has a significantly lower density than the average density of its $k$ neighbors, its LOF score is significantly greater than 1, marking it as a local outlier.
- **Production Note**: By default, standard LOF is non-parametric and transductive (only scores training points). We configured LOF with `novelty=True` in Scikit-Learn so it fits a training baseline and can evaluate unseen test transactions in production.

### C. One-Class SVM (OC-SVM)
- **Analogy**: Imagine drawing a tight fence around a herd of sheep in a field. Anything inside the fence is considered part of the herd. Any wolf or stranger wandering outside the fence is flagged as an intruder.
- **Mechanism**: Uses a non-linear kernel (specifically Radial Basis Function or RBF) to map input features into a high-dimensional feature space. It then constructs a maximum-margin hyperplane that separates the normal data points from the origin.
- **Hyperparameter $\nu$ (nu)**: Represents an upper bound on the fraction of training errors (outliers) and a lower bound on the fraction of support vectors.

---

## 📋 3. Top 20 Likely HCL Interview Questions & Answers

### Q1: Why did you choose unsupervised anomaly detection instead of supervised algorithms like Random Forest or XGBoost?
**Answer**:
1. **Severe Imbalance**: In fraud detection, fraud accounts for ~0.17% of data. Supervised algorithms heavily optimize for majority-class accuracy and tend to ignore the minority class unless artificial oversampling (SMOTE) is applied.
2. **Zero-Day / Novel Fraud**: Supervised models only detect patterns they have seen in labeled historical data. Fraudsters constantly change tactics (new merchant categories, velocity patterns). Anomaly detection flags anything deviating from legitimate behavior, catching zero-day fraud without needing prior examples.
3. **Delayed Ground Truth**: In real banks, fraud labels often take 30 to 90 days to confirm due to chargeback investigations. Anomaly detection functions immediately on unlabelled streaming transactions.

### Q2: What is the "Accuracy Paradox", and why is accuracy useless in fraud detection?
**Answer**:
The Accuracy Paradox occurs when a model achieves high accuracy by simply guessing the majority class, while failing completely at its business objective.
- If a dataset has 99,828 legitimate transactions and 172 fraudulent transactions, a naive model that predicts "Legitimate" for every transaction achieves **99.83% accuracy**.
- However, its **Recall is 0%** and **F1-Score is 0.0**. It missed every single fraud event, causing catastrophic financial losses.
- In fraud detection, business success is measured by **Recall** (catching fraud), **Precision** (avoiding false alarms), and **PR-AUC**.

### Q3: Explain Precision, Recall, and the F1-Score in the context of credit card transactions.
**Answer**:
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$
  - *Definition*: Out of all actual fraudulent transactions, what percentage did our system catch?
  - *Business impact*: High recall minimizes financial fraud losses. Low recall means fraudsters get away with stolen money.
- **Precision**: $\frac{TP}{TP + FP}$
  - *Definition*: Out of all transactions flagged as fraud by the model, how many were truly fraud?
  - *Business impact*: High precision ensures customer satisfaction. Low precision results in blocked legitimate cards, embarrassing cardholders at checkouts.
- **F1-Score**: $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
  - Harmonic mean balancing the trade-off between fraud prevention and customer friction.

### Q4: Why is PR-AUC (Precision-Recall AUC) preferred over ROC-AUC in fraud detection?
**Answer**:
- **ROC-AUC** plots True Positive Rate vs False Positive Rate ($FPR = \frac{FP}{FP + TN}$).
- Because the number of True Negatives ($TN$) in credit card data is enormous (hundreds of thousands), the denominator in $FPR$ is huge. Even if the model generates 1,000 false alarms ($FP = 1,000$), $FPR$ remains tiny (~0.003). This produces an artificially inflated ROC-AUC of 0.95+, giving a false sense of security.
- **PR-AUC (Average Precision)** does not include True Negatives in its calculation. It focuses exclusively on $TP$, $FP$, and $FN$. If a model generates excessive false positives, PR-AUC drops sharply. Therefore, PR-AUC is the gold standard for imbalanced fraud evaluation.

### Q5: What is Data Leakage, and how did you prevent it in your preprocessing pipeline?
**Answer**:
- **Data Leakage** occurs when information from outside the training dataset (such as the test set or future data) inadvertently influences model training.
- **How I Prevented It**:
  1. **Strict Temporal/Stratified Split First**: We partitioned the dataset into 80% train and 20% test *before* any feature transformation took place.
  2. **Pipeline Encapsulation**: Our `FraudPreprocessor` was fitted strictly on `X_train`. The median and Interquartile Range (IQR) learned from `X_train` were saved and reused to transform `X_test` and live inference queries without re-fitting.
  3. **Target Exclusion**: The target `Class` column was explicitly stripped prior to transformation, ensuring zero target leakage.

### Q6: Why did you use `RobustScaler` instead of `StandardScaler` or `MinMaxScaler`?
**Answer**:
- Transaction `Amount` is heavily right-skewed with extreme outliers (e.g., $1 to $25,000).
- `StandardScaler` relies on the mean and standard deviation. A single $25,000 transaction pulls the mean upward and inflates the standard deviation, compressing 99% of normal transactions into a microscopic range near zero.
- `MinMaxScaler` maps values between 0 and 1 using min and max. An extreme outlier pushes almost all typical transactions into 0.001.
- `RobustScaler` scales data using the **Median** and the **Interquartile Range** ($IQR = Q_3 - Q_1$). Because median and IQR are order-based statistics, they are mathematically immune to extreme outliers.

### Q7: What is the Contamination parameter, and how did you set it?
**Answer**:
- **Contamination** is the expected proportion of outliers / fraudulent transactions in the dataset.
- In Scikit-Learn's `IsolationForest` and `LocalOutlierFactor`, contamination determines the default threshold for the internal decision function.
- In our project, historical credit card datasets exhibit a fraud rate of ~0.17% to 0.20%. We set default contamination to `0.002` (0.2%).
- In our architecture, we decoupled binary classification from the fixed internal threshold by converting raw decision values into **continuous normalized anomaly scores [0.0, 1.0]**, allowing the risk team to tune the decision threshold dynamically in the Streamlit UI.

### Q8: Why is an anomaly score not necessarily proof of fraud?
**Answer**:
An anomaly detector flags **statistical and geometric deviations**, not criminal intent.
- A cardholder purchasing an international business class flight or shopping at 3:00 AM while on vacation is mathematically an outlier, but completely legitimate.
- In production banking systems, an anomaly score does not immediately cancel the card. Instead, it triggers **tiered risk actions**:
  - Score < 0.50 (LOW): Frictionless approval.
  - Score 0.50 – 0.75 (MEDIUM): Step-up authentication (SMS OTP or mobile push approval).
  - Score 0.75 – 0.85 (HIGH): Automated call verification.
  - Score > 0.85 (CRITICAL): Temporary hold and fraud analyst review.

### Q9: What are features V1 through V28 in the European dataset?
**Answer**:
Features V1 through V28 are numerical components obtained via **Principal Component Analysis (PCA)**. Due to European banking privacy and GDPR regulations, sensitive cardholder data (Card Number, CVV, Cardholder Name, Merchant Category Code, IP Address, Device Fingerprint) was transformed into orthogonal principal components. V1 to V28 have zero mean and unit variance for typical transactions.

### Q10: How do you identify which features contributed most to a specific transaction being flagged?
**Answer**:
1. **Domain Feature Correlation**: In the European dataset, statistical analysis shows that fraudulent transactions consistently exhibit extreme negative deviations in `V14`, `V12`, `V10`, and `V17`, and elevated positive values in `V4` and `V11`.
2. **Deviation Analysis**: In our `predict.py` and `app.py`, we compare the individual transaction's feature values against the baseline legitimate distribution and highlight components exceeding 3 standard deviations.
3. **Advanced Explainability**: In production, we can attach **SHAP (SHapley Additive exPlanations)** or **TreeSHAP** to Isolation Forest to compute exact Shapley value contributions for each feature.

### Q11: How does Isolation Forest handle high-dimensional data?
**Answer**:
Isolation Forest handles high dimensions remarkably well because each isolation tree selects random 1D split dimensions independently. However, if irrelevant noisy features exist, isolation efficiency drops. Because our features were either PCA-condensed (V1-V28) or robustly scaled (Time and Amount), the feature space was compact and dense (30 features total).

### Q12: How does the Local Outlier Factor (LOF) algorithm differ in training mode vs testing mode?
**Answer**:
- Standard LOF is an unsupervised transductive algorithm: it computes density for the dataset provided and has no `predict()` method for new points.
- To use LOF in an assessment-ready, production-grade system, we instantiated it with `novelty=True`. This enables LOF to fit a neighborhood representation on the training data and then compute `predict()` and `decision_function()` / `score_samples()` on unseen test transactions.

### Q13: What is the computational complexity of the three models?
**Answer**:
- **Isolation Forest**:
  - Training: $O(t \cdot \psi \log \psi)$, where $t$ is number of trees (e.g. 150) and $\psi$ is subsample size (e.g. 256).
  - Inference: $O(t \cdot \log \psi)$ — extremely fast (< 1ms per query).
- **Local Outlier Factor**:
  - Training: $O(n^2)$ without tree indexing; $O(n \log n)$ with KD-tree / Ball-tree.
  - Inference: Requires $k$-nearest neighbor search against reference points.
- **One-Class SVM**:
  - Training: Scales between $O(n^2)$ and $O(n^3)$ due to quadratic optimization of the kernel matrix.
  - Inference: Depends on the number of support vectors.

### Q14: How does dynamic thresholding work in your Streamlit application?
**Answer**:
Rather than relying on a hardcoded 0.5 cutoff, our system computes a normalized continuous score $S(x) \in [0.0, 1.0]$. The Streamlit slider allows an operator to adjust the cutoff:
- Lowering threshold (e.g., 0.35) increases **Recall** (captures 90%+ of frauds) at the expense of more False Positives. Suitable for high-value transactions or heightened security periods.
- Raising threshold (e.g., 0.70) increases **Precision** (only flags indisputable anomalies), suitable when customer support teams have limited bandwidth to process alerts.

### Q15: How did you structure the project to follow professional software engineering best practices?
**Answer**:
1. **Decoupled Architecture**: Separated concerns into `src/dataset.py` (data loading), `src/preprocessing.py` (scaling), `src/models.py` (algorithm wrappers), and `src/evaluation.py` (metrics).
2. **Artifact Serialization**: Scalers and models are saved via `joblib` so inference runs identically in CLI and UI.
3. **Automated Unit Tests**: Implemented comprehensive tests in `tests/` covering dataset schema validation, missing column handling, outlier scaling, model score bounds, and inference endpoints.
4. **Resilient Fallback**: Built an autonomous synthetic data generator so the application works out-of-the-box even if the 150MB Kaggle dataset is not downloaded yet.

### Q16: How would you deploy this project into a real banking microservices architecture?
**Answer**:
1. **Streaming Ingestion**: Transactions arrive via **Apache Kafka** or **AWS Kinesis** event streams.
2. **Inference Microservice**: Package the trained model in a containerized **FastAPI** service deployed on **Kubernetes (EKS/GKE)**.
3. **Low-Latency Serialization**: Convert the Isolation Forest model to **ONNX** or **Treelite** to achieve sub-millisecond (< 5ms) inference times required by payment gateways.
4. **Feature Store**: Use a feature store like **Feast** or Redis to retrieve real-time velocity aggregates (e.g., number of transactions in the last 10 minutes).
5. **Human-in-the-Loop Audit**: Alerts populate an analyst queue (similar to our Streamlit dashboard) for Tier-2 verification.

### Q17: What is Concept Drift, and how would you monitor it in this fraud system?
**Answer**:
- **Concept Drift**: The statistical relationship between transaction features and fraud changes over time as fraudsters devise new techniques and legitimate purchasing patterns evolve (e.g., holiday seasons).
- **Monitoring Strategy**:
  1. Monitor Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test statistics on incoming feature distributions weekly.
  2. Track the ratio of flagged transactions and customer dispute rates over time.
  3. Implement automated re-training pipelines using Airflow or Kubeflow.

### Q18: What is the difference between Unsupervised and Semi-Supervised anomaly detection?
**Answer**:
- **Unsupervised Anomaly Detection**: The model trains on raw, unlabelled data that contains an unknown mixture of legitimate transactions and rare fraud cases, guided only by the `contamination` parameter.
- **Semi-Supervised (Novelty Detection)**: The model is trained exclusively on confirmed clean, legitimate transactions (`Class == 0`). It learns a pure profile of normal behavior. Any deviation in testing is considered an anomaly. Our `train.py` supports both paradigms via the `--normal-only-train` flag.

### Q19: What are the primary business trade-offs when optimizing a fraud detection model?
**Answer**:
The trade-off is between **Chargeback / Direct Financial Loss** vs. **Customer Insult Rate**:
- Missing a fraud event ($FN$) incurs direct chargeback liability, interchange fines, and brand damage.
- Generating a false alarm ($FP$) causes customer friction (declined card at a restaurant), customer service call center costs, and churn to competing credit cards.
- The threshold must be tuned based on transaction amount: high-dollar transactions justify more false positive friction, while micro-transactions require high precision.

### Q20: If you had more time and resources, what would you improve in this project?
**Answer**:
1. **Graph Neural Networks (GNNs)**: Model transactions as a bipartite graph (Cardholders $\leftrightarrow$ Merchants $\leftrightarrow$ IP Addresses) to detect coordinated mule rings.
2. **Sequential Modeling**: Add LSTM / Temporal Transformers to capture sequential velocity (e.g., transaction in New York followed 15 minutes later by a transaction in London).
3. **SHAP Integration**: Provide interactive visual waterfall plots in the Streamlit UI explaining each individual prediction.
4. **Dockerization**: Provide a `Dockerfile` and `docker-compose.yml` for instant containerized deployment.

### Q21: Why might a synthetic fraud dataset show ROC-AUC = 1.00, and how did you diagnose and fix this?
**Answer**:
- **Diagnosis**: In simplistic synthetic generators, fraud samples are created by shifting several PCA features (e.g., $V_{14}, V_{12}, V_{10}$) by 5 or 6 standard deviations simultaneously. In 30-dimensional space, this creates a clean geometric hyper-separation where every single fraud instance has a higher continuous anomaly score than all normal instances, resulting in a mathematically trivial ROC-AUC of 1.00 and PR-AUC of 1.00.
- **The Problem**: Real-world fraud is subtle, multi-modal, and camouflaged. A model evaluated on trivially separated data gives a false sense of security.
- **The Solution**: We upgraded `data/generate_data.py` to generate **4 realistic fraud typologies**:
  1. *Micro-authorization / Card Testing*: Very low amounts ($0.50 - $4.00) with slight negative deviations on $V_{12}$ and $V_{14}$.
  2. *Account Takeover*: High-value ($800 - $3,500) bursts with extreme shifts on $V_{17}$ and $V_{14}$.
  3. *Velocity Bursts*: Clustered night-time activity with moderate shifts on $V_{4}$ and $V_{11}$.
  4. *Camouflaged Fraud (25%)*: Subtle anomalies whose feature distributions overlap with the tail of legitimate transactions ($1.2\sigma$ shift).
- **Result**: Isolation Forest achieves a realistic ROC-AUC of **0.8878** and Average Precision of **0.3718** (a 185x lift over the 0.002 random baseline), proving true discriminative capability under realistic overlap.

### Q22: Why do you use a 3-way split (Train / Validation / Test) and calibrate the decision threshold on Validation?
**Answer**:
- **Why a 3-way split**: In standard supervised learning, people often split 80/20 into Train/Test. But in unsupervised anomaly detection, raw anomaly scores are continuous distances, not calibrated probabilities. Selecting an operational cutoff threshold (e.g., 0.695) requires optimization.
- **Preventing Optimism Bias**: If we tune the decision threshold on the test set to maximize F1 or Recall, the test set is no longer an unbiased evaluation of unseen performance—it suffers from **threshold leakage**.
- **Our Methodology**: We split 60% Train (15,000 samples) to fit the models, 20% Validation (5,000 samples) to calibrate the optimal $F_2$-score threshold ($\beta=2.0$ to prioritize fraud recall over precision), and 20% Held-Out Test (5,000 samples) strictly evaluated once at that calibrated cutoff without any adjustments.

### Q23: What is the difference between Average Precision (AP) and Trapezoidal PR-AUC in Scikit-Learn?
**Answer**:
- **Trapezoidal PR-AUC (`auc(recall, precision)`)**: Uses the trapezoidal rule to compute the area under the PR curve. In extreme class imbalance, the PR curve is not monotonic; using linear interpolation between recall points systematically overestimates the area because precision drops sharply as recall increases. Scikit-Learn explicitly recommends avoiding `auc()` on PR curves for this reason.
- **Average Precision (`average_precision_score`)**: Computes the area as a weighted step-function sum of precisions multiplied by the increase in recall at each threshold ($\sum_n (R_n - R_{n-1}) P_n$). It represents the exact area under the interpolated precision-recall curve and prevents optimistic distortion.

---

## 🎯 Final Checklist for HCL Technical Interview
- [x] Able to clearly explain the **Accuracy Paradox** in 30 seconds.
- [x] Understand why **RobustScaler** handles skewed Amount features better than StandardScaler.
- [x] Can explain how **Isolation Forest** isolates outliers using random splits.
- [x] Know the mathematical definitions of **Precision**, **Recall**, and **PR-AUC**.
- [x] Familiar with why an anomaly does **not** always equal fraud (false alarms & 2FA step-up).
- [x] Confident demonstrating the live Streamlit dashboard, single transaction scoring, and batch CSV export.
