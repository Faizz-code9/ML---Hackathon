\# PathSnap - Machine Learning Approach



\## 1. Overview



PathSnap investigates machine-learning approaches for GPS trajectory map matching. The objective is to identify suitable road-segment candidates for GPS observations and reconstruct plausible trajectories.



\## 2. Candidate Model Families



The following models are considered for comparative experimentation:



\### Logistic Regression

A baseline classification approach used to establish an initial performance reference.



\### K-Nearest Neighbors (K-NN)

A similarity-based method that predicts using nearby training samples.



\### Random Forest

An ensemble learning method that combines multiple decision trees and can model nonlinear relationships.



\### LightGBM

A gradient-boosting framework based on decision trees, considered for efficient learning from structured data.



\### Multilayer Perceptron (MLP)

A neural-network approach used to explore nonlinear relationships among input features.



\## 3. Model Evaluation



Models should be evaluated under a consistent data-splitting and evaluation procedure.



Potential metrics include:

\- Accuracy

\- Precision

\- Recall

\- Macro F1-score



Metric selection will depend on the task formulation and class distribution.



\## 4. Sequential Map Matching



Individual candidate predictions may not form a physically consistent road sequence.



A Hidden Markov Model (HMM) is considered for incorporating sequential dependencies between road candidates. The Viterbi algorithm can be used to determine the most probable sequence of hidden states.



\## 5. Experimental Considerations



\- Maintain separation between training, validation, and test data.

\- Avoid data leakage between trajectories.

\- Compare models using consistent evaluation conditions.

\- Analyze road-network connectivity separately from classification metrics.

\- Report results only after experiments have been completed.



\## 6. Current Status



The ML model comparison and sequential decoding approach are part of the project's investigation. Model training, validation, and quantitative comparisons are ongoing, and conclusions will be based on measured experimental results.



