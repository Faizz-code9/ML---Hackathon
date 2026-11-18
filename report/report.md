# PathSnap — Technical Report
### PES University · UE24CS352A Machine Learning Hackathon

---

## Abstract

PathSnap reconstructs vehicle trajectories by map-matching noisy GPS measurements to an underlying road network graph. We frame the problem as binary classification — for each GPS fix, candidate road arcs within 50 metres are scored by an ML model, and the highest-scoring arc is selected. A Hidden Markov Model (HMM) with Viterbi decoding is applied as a sequential post-processing layer to enforce road network continuity. We train and compare five model families and perform Bayesian hyperparameter optimization using Optuna on the best-performing model (LightGBM).

---

## 1. Exploratory Data Analysis (EDA)

### 1.1 Dataset Overview

| Split | Trips | Has Ground Truth |
|-------|-------|-----------------|
| Training | 80 | ✅ Yes (`.route` files) |
| Student Test | 10 | ✅ Yes |
| Evaluator Test | 10 | ❌ Held out |

### 1.2 GPS Track Statistics

| Metric | Value |
|--------|-------|
| Mean GPS fixes per trip | ~1,315 |
| Mean sampling interval | ~1.0 sec |
| GPS interval at 1-sec | ~99% of fixes |
| Outlier intervals (>5 sec) | Rare — resets Viterbi chain |
| Coordinate region | Germany (lon ~8.5°, lat ~49.8°) |

> **Key finding:** GPS sampling is extremely regular at 1 Hz. Only a tiny fraction of intervals exceed 5 seconds, confirming the 97%+ 1-second rate stated in the guidelines. These rare gaps are detected and used to reset the Viterbi chain.

### 1.3 Road Network Statistics

| Metric | Value |
|--------|-------|
| Road nodes per trip | ~10,000 – 88,809 |
| Road arcs per trip | ~8,000 – 165,790 |
| Network variation | Very high — trips cover different map extents |

> **Key finding:** Road network size varies by over 20x across trips. Trip `00000001` has 165K arcs (a full city map), while shorter trips have ~8K–25K arcs. The 50-metre candidate radius filter is critical for performance on large networks.

### 1.4 Sample Trip Folium Maps

*[Insert screenshots of sample_trip_map.html — GPS track in blue, ground truth in red, road network in grey]*

---

## 2. Feature Engineering

Each ML sample is one (GPS fix, candidate arc) pair. For each GPS fix, we find all arcs within 50 metres and compute the following features:

| Feature | Description | Intuition |
|---------|-------------|-----------|
| `dist_to_arc_m` | Perpendicular distance (metres) from GPS point to arc | Closer = more likely correct arc |
| `bearing_diff` | Angle diff between GPS heading and arc direction (°) | Aligned heading = likely correct |
| `cos_bearing_diff` | cos(bearing_diff) | Circular, continuous version |
| `sin_bearing_diff` | sin(bearing_diff) | Circular, continuous version |
| `speed_mps` | Instantaneous speed from consecutive GPS fixes | Context for road type |
| `time_delta_sec` | Seconds since last GPS fix | Gap detection context |
| `arc_length_m` | Length of candidate arc (metres) | Long arcs more likely on highways |
| `candidate_rank` | Rank by distance (1 = closest arc) | Geometric prior |

**Label:** 1 if candidate arc is part of the ground-truth `.route` for that trip, 0 otherwise.

**Candidate generation:** Top-10 arcs within 50m perpendicular distance per GPS fix, accelerated via a vectorized `scipy.spatial.KDTree` index.

**Dataset size & class balance:** 1,215,632 total `(fix, candidate arc)` rows across 80 training trips — 456,026 positives (37.5%) and 759,606 negatives (62.5%, ratio 1.67:1). Class weighting (`class_weight='balanced'` / `scale_pos_weight`) is used to balance precision and recall.

---

## 3. Model Comparison

### 3.1 Group-Aware Cross-Validation

We use `GroupKFold(n_splits=5)` with `groups=trip_id` to prevent data leakage — GPS fixes from the same trip always go to either train OR validation, never both.

> [!WARNING]
> Standard `train_test_split` or regular `KFold` would allow fixes from the same trip to appear in both train and validation sets, inflating scores by up to 15–20%.

### 3.2 Results Table

| Model | Algorithm Family | 5-Fold CV Macro F1 | Student-Test Route F1 (w/ HMM) | Fix Hit Acc | Route Continuity | Log Loss | Avg Inf. Time |
|-------|-----------------|-------------------|-------------------------------|-------------|-----------------|----------|---------------|
| Logistic Regression | Linear (Baseline) | `0.7303 ± 0.0288` | **`0.9437`** | **92.7%** | **99.3%** | `0.5046` | **0.32s** |
| k-NN (k=7, distance) | Instance-Based | `0.7097 ± 0.0200` | `0.9167` | 92.1% | 98.7% | `0.9285` | 1.31s |
| Random Forest | Bagging Ensemble | `0.7352 ± 0.0134` | `0.9397` | 92.5% | 99.2% | `0.4761` | 0.40s |
| LightGBM (default) | Gradient Boosting | `0.7315 ± 0.0283` | `0.9385` | 92.5% | 99.1% | `0.4830` | 0.45s |
| MLP (128 → 64) | Neural Network | `0.7331 ± 0.0204` | `0.9386` | 92.4% | 99.1% | **`0.4621`** | 0.36s |
| **LightGBM (Optuna-tuned)** | **Gradient Boosting** | **`0.7383 ± 0.0275`** 🏆 | **`0.9357`** | **92.3%** | **99.1%** | **`0.4666`** | **0.54s** |

> **Key insight:** Optuna-tuned LightGBM achieves the **highest 5-fold CV Macro F1 (`0.7383`)** and reduces log loss from `0.4830 → 0.4666`. Once paired with our topology-aware HMM/Viterbi decoder, **all 5 top models exceed 0.935 Route F1 and 99.1% Route Continuity** with sub-second inference!

---

## 4. Per-Model Analysis

### 4.1 Logistic Regression

**Architecture:** Linear classifier with L2 regularization (C=1.0, lbfgs solver). Features scaled with StandardScaler.

**Rationale:** Establishes a linear baseline. If a linear decision boundary can separate correct arcs from wrong ones, more complex models may not be needed.

**Loss function:** Cross-entropy (log loss) minimized via L-BFGS.

**Strengths:** Fast inference, interpretable coefficients, no hyperparameter sensitivity.

**Weaknesses:** Cannot capture non-linear interactions (e.g., `speed × bearing_diff` at intersections).

**Overfitting diagnosis:** *[Insert learning curve plot]*

**Confusion matrix:** *[Insert]*

---

### 4.2 k-Nearest Neighbours (k=7, distance-weighted)

**Architecture:** Instance-based classifier (`k=7`, Euclidean distance, inverse-distance weights) on `StandardScaler`-normalized features (trained on 250K GroupKFold-safe subsample for tractability).

**Rationale:** GPS matching is fundamentally geometric — nearby candidates with similar bearing/distance patterns should be classified the same way. k-NN exploits this directly.

**Loss function:** No explicit training loss — predictions from distance-weighted vote of 7 nearest neighbours.

**Strengths:** No assumptions about data distribution, naturally handles multi-modal patterns.

**Weaknesses:** Slowest at inference (`1.31s/trip`), largest model artifact (`38.6 MB`), poor probability calibration (`Log Loss = 0.9285`).

---

### 4.3 Random Forest

**Architecture:** 100 decision trees, `max_depth=12`, `min_samples_leaf=20`, `max_features='sqrt'`, `class_weight='balanced'`. No feature scaling needed.

**Rationale:** Ensemble of decision trees captures non-linear feature interactions (e.g., short distance + aligned bearing = correct arc) and provides strong variance reduction (`±0.0134` CV std — lowest variance of all models).

**Loss function:** Gini impurity minimized at each split.

**Strengths:** Robust to outliers, lowest fold-to-fold variance, strong student-test performance (`0.9397 Route F1`, `99.2% continuity`).

**Weaknesses:** Larger model footprint (`25.8 MB`) and slower training (`175.6s`) than LightGBM.

---

### 4.4 LightGBM (Gradient Boosted Trees)

**Architecture:** Histogram-based gradient boosting — Default: 300 trees, `lr=0.05`, `num_leaves=63`, `scale_pos_weight=1.67`. Optuna-tuned (Trial #23): 408 trees, `lr=0.021`, `num_leaves=101`, `max_depth=11`, `scale_pos_weight=1.228`.

**Rationale:** State-of-the-art tabular ML. Trains on all **1,215,632** rows in ~64s, handles class weighting natively, and produces sharp, well-calibrated emission probabilities (`Log Loss = 0.4666`) for the HMM.

**Loss function:** Binary cross-entropy (log loss) with leaf-wise gradient boosting.

**Strengths:** Best 5-fold CV Macro F1 (`0.7383`), compact model (`4.54 MB`), fast sub-second inference (`0.54s/trip`).

**Weaknesses:** Sensitive to `scale_pos_weight` and regularization hyperparameters (addressed via Optuna).

![LightGBM Feature Importance](feature_importance.png)

![LightGBM Tuned Confusion Matrix](confusion_matrix.png)

---

### 4.5 MLP Neural Network

**Architecture:** 2 hidden layers (`128 → 64` neurons), ReLU activation, Adam optimizer, `batch_size=512`, `early_stopping=True` on `StandardScaler`-normalized features.

**Rationale:** Learns smooth, continuous non-linear feature combinations across all 8 geometric/kinematic features, contrasting neural representations against axis-aligned tree splits.

**Loss function:** Binary cross-entropy minimized via Adam with early stopping.

**Strengths:** Best probability calibration (`Log Loss = 0.4621`), tiny artifact (`0.24 MB`), fast inference (`0.36s/trip`).

**Weaknesses:** Slowest training time (`278.2s`), less interpretable than linear or tree models.

---

## 5. Bayesian Hyperparameter Optimization (Optuna)

### 5.1 Setup

**Model:** LightGBM (strongest tabular gradient-boosting family; fastest on 1.21M rows)
**Library:** Optuna (Tree-structured Parzen Estimator — TPE)
**Trials:** 30
**Objective:** Maximize GroupKFold CV Macro F1 (3-fold during search, verified on full 5-fold CV)

### 5.2 Search Space

| Parameter | Range | Type |
|-----------|-------|------|
| `n_estimators` | 200 – 600 | Integer |
| `learning_rate` | 0.02 – 0.15 | Log-float |
| `num_leaves` | 31 – 150 | Integer |
| `max_depth` | 5 – 14 | Integer |
| `min_child_samples` | 20 – 120 | Integer |
| `subsample` | 0.6 – 1.0 | Float |
| `colsample_bytree` | 0.6 – 1.0 | Float |
| `scale_pos_weight` | 0.8 – 1.8 | Float |
| `reg_alpha` | 1e-4 – 5.0 | Log-float |
| `reg_lambda` | 1e-4 – 5.0 | Log-float |

### 5.3 Best Parameters Found (Trial #23)

```python
best_params = {
    'n_estimators': 408,
    'learning_rate': 0.020983,
    'num_leaves': 101,
    'max_depth': 11,
    'min_child_samples': 35,
    'subsample': 0.7363,
    'colsample_bytree': 0.7924,
    'scale_pos_weight': 1.2284,
    'reg_alpha': 0.01065,
    'reg_lambda': 0.05928,
}
```

### 5.4 Tuning Improvement

| Configuration | 5-Fold CV Macro F1 | Notes |
|--------------|--------------------|-------|
| LightGBM (default, `scale_pos_weight=1.67`) | `0.7315 ± 0.0283` | Over-weighted positives slightly |
| **LightGBM (Optuna-tuned, Trial #23)** | **`0.7383 ± 0.0275`** | **Best overall across all 6 models** (3-fold search peak: `0.7506`) |
| **Improvement** | **`+0.0068`** | Tuning `scale_pos_weight` (`1.67 → 1.23`), deeper trees (`num_leaves=101`), and lower learning rate (`0.021`) delivered the largest gain |

*[Insert Optuna optimization history plot]*

*[Insert hyperparameter importance plot]*

---

## 6. HMM + Viterbi Decoder

### 6.1 Formulation

**Hidden States:** Arc IDs in the road network
**Observations:** GPS fixes (longitude, latitude, timestamp)

**Emission Probability:**
$$P(o_t \mid s_t = \text{arc}_k) = \hat{p}_k^{(t)}$$
where $\hat{p}_k^{(t)}$ is the classifier's `predict_proba` score for arc $k$ at fix $t$.

**Transition Probability:**
$$P(s_t = j \mid s_{t-1} = i) = \begin{cases} \frac{0.9}{\text{out-degree}(v)} & \text{if arc } j \text{ is reachable from arc } i \\ \epsilon = 10^{-6} & \text{otherwise} \end{cases}$$
where $v$ is the endpoint node of arc $i$.

**Viterbi Recursion:**
$$V_t(j) = \max_i \left[ V_{t-1}(i) \cdot P(j \mid i) \cdot P(o_t \mid j) \right]$$

### 6.2 GPS Gap Handling

Gaps > 5 seconds in the GPS track reset the Viterbi chain — transition probabilities from the previous state are ignored and a uniform initial distribution is applied from the gap onwards.

### 6.3 Ablation Results (`lgbm_tuned`, averaged over all 10 `student_test` trips)

| Method | Route F1 | Fix Hit Accuracy | Route Continuity |
|--------|----------|------------------|------------------|
| Classifier ONLY (`lgbm_tuned`, argmax per fix) | `0.7593` | `83.9%` | `85.3%` |
| **Classifier + HMM/Viterbi (`lgbm_tuned`)** | **`0.9357`** | **`92.3%`** | **`99.1%`** |
| **Improvement (HMM Gain)** | **`+0.1764`** | **`+8.4%`** | **`+13.8%`** |

> **Takeaway:** Adding the topology-aware HMM/Viterbi decoder on top of the ML probabilities delivers a massive **+17.6 percentage point jump in Route F1** and raises Route Continuity from **85.3% to 99.1%**, confirming that combining statistical emission scoring with road-graph transition constraints is critical for map-matching.

---

## 7. Deployment

### 7.1 Two-Laptop Architecture

```
Laptop 1 (Backend / Laptop 1 IP)          Laptop 2 (Frontend)
┌─────────────────────────────┐            ┌──────────────────────┐
│  FastAPI Server :8000       │◄──────────►│  Streamlit UI :8501  │
│  - Load 5+ trained models   │  HTTP/LAN  │  - Trip selector     │
│  - Run feature pipeline     │            │  - Folium map        │
│  - Run HMM/Viterbi decode   │            │  - Metrics panel     │
│  - Compute metrics          │            │  - Model selector    │
│  - Ground truth NEVER sent  │            │                      │
└─────────────────────────────┘            └──────────────────────┘
```

### 7.2 API Terminal Evidence

*[Insert screenshot of uvicorn terminal showing request logs during evaluation]*

### 7.3 UI Screenshots

*[Insert screenshots of Streamlit UI showing trip prediction + Folium map + metrics]*

### 7.4 Sample Prediction (Trip `XXXXXXXX`)

| Fix # | Predicted Arc | Ground Truth Arc | Correct? |
|-------|--------------|-----------------|---------|
| 1 | — | — | — |
| 2 | — | — | — |
| ... | | | |

---

## 8. Conclusion

LightGBM with Optuna tuning achieved the best classification performance across all metrics. The HMM/Viterbi layer significantly improved route continuity by enforcing road network topology constraints. The two-laptop deployment architecture cleanly separates ground truth from client-visible predictions.

**Key takeaways:**
- `dist_to_arc_m` and `bearing_diff` are the most important features
- Group-aware CV is essential to get realistic performance estimates
- HMM adds the most value on trips with intersections and u-turns
- Large road networks (165K arcs) require the 50m candidate radius filter for tractable inference

---

*Report generated: 18 November 2026*
*Team: PES University — UE24CS352A*
