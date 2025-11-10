#  DA5401 – Assignment 8: Ensemble Learning on Bike Sharing Dataset

##  Overview
This project demonstrates the use of **ensemble learning techniques** — Bagging, Boosting, and Stacking — to improve predictive performance on the **Bike Sharing (hourly)** dataset.  
The objective is to predict the total count of bike rentals (`cnt`) using various machine learning models and ensemble methods.

---

##  Project Structure


---

##  Part A – Data Preprocessing & Baseline Models
- Loaded and cleaned the dataset.
- Dropped irrelevant columns: `instant`, `dteday`, `casual`, and `registered`.
- Converted categorical features (`season`, `weathersit`, `mnth`, `hr`) using **One-Hot Encoding**.
- Split data into **train/test (70/30)**.
- Trained and evaluated:
  - **Decision Tree Regressor (max_depth=6)**
  - **Linear Regression**
- Used the better performing model as the **baseline**.

| Model | RMSE |
|:--|--:|
| Decision Tree (max_depth=6) | 118.53 |
| **Linear Regression (Baseline)** | **100.42** |

 **Baseline Model:** Linear Regression (RMSE = 100.42)

---

##  Part B – Ensemble Techniques

### B1: Bagging (Variance Reduction)
- Implemented a **Bagging Regressor** using the tuned Decision Tree as the base estimator.
- Fine-tuned parameters:  
  - `n_estimators = 100`  
  - `max_samples = 0.6`  
  - `max_features = 0.9`  
  - `bootstrap = False`

| Model | RMSE |
|:--|--:|
| Decision Tree (Tuned) | 118.47 |
| **Bagging Regressor (Tuned)** | **108.62** |

 Bagging successfully **reduced variance** by averaging predictions across multiple trees.

---

### B2: Boosting (Bias Reduction)
- Implemented a **Gradient Boosting Regressor** to address model bias.
- Fine-tuned parameters:
  - `n_estimators = 300`
  - `learning_rate = 0.1`
  - `max_depth = 5`
  - `min_samples_split = 2`

| Model | RMSE |
|:--|--:|
| Bagging Regressor (Tuned) | 108.62 |
| **Gradient Boosting Regressor (Tuned)** | **48.34** |

 Boosting significantly **reduced bias**, improving prediction accuracy over Bagging and the baseline.

---

##  Part C – Stacking Regressor (Fully Fine-Tuned)

**Base Learners (Level-0):**
- K-Nearest Neighbors (KNN)
- Bagging Regressor (Tuned)
- Gradient Boosting Regressor (Tuned)

**Meta-Learner (Level-1):**
- Ridge Regression

All models were fine-tuned using **GridSearchCV** for optimal performance.

| Model | RMSE |
|:--|--:|
| Baseline (Best of DT/Linear) | 100.42 |
| Decision Tree (Tuned) | 118.47 |
| Bagging Regressor (Tuned) | 108.62 |
| Gradient Boosting Regressor (Tuned) | 48.34 |
| **Stacking Regressor (Fully Tuned)** | **44.71** |

 **Stacking achieved the lowest RMSE (44.71)** — outperforming every individual and ensemble model.

---

##  Part D – Final Analysis

| Model | RMSE | Key Insight |
|:--|--:|:--|
| Baseline (Linear Regression) | 100.42 | Reference performance |
| Decision Tree (Tuned) | 118.47 | Overfits slightly; high variance |
| Bagging Regressor (Tuned) | 108.62 | Reduces variance modestly |
| Gradient Boosting Regressor (Tuned) | 48.34 | Greatly reduces bias |
| **Stacking Regressor (Fully Tuned)** | **44.71** | Best balance of bias and variance |

---

##  Key Insights
- **Bagging:** Reduces variance by averaging multiple Decision Trees trained on bootstrap samples.
- **Boosting:** Reduces bias by sequentially correcting errors from previous learners.
- **Stacking:** Combines both — blending diverse models to optimize overall performance using a meta-learner.

---

##  Conclusion
- **Best Model:** 🏆 **Stacking Regressor (RMSE = 44.71)**  
- **Why It Works:** Combines multiple learners (KNN, Bagging, Boosting) with a Ridge meta-learner to achieve an optimal **bias–variance trade-off**.  
- **Result:** The ensemble effectively captures both global and local patterns, providing the most accurate and stable predictions across all methods.

---

##  Technologies Used
- Python 3.10+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- Environment: Jupyter Notebook / VS Code

---

##  Author
**Name:** V G Masilamani  
**Roll Number:** DA25S005 

---

##  References
- Scikit-learn Documentation – https://scikit-learn.org/  
- UCI Machine Learning Repository – Bike Sharing Dataset  
- Course Material – Bias–Variance Trade-off & Ensemble Methods
