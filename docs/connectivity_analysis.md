\# PathSnap - Preliminary Connectivity Analysis



\## 1. Purpose



This analysis evaluates directed connectivity between consecutive road arcs in a trajectory. It serves as a preliminary diagnostic for understanding road-network consistency.



\## 2. Evaluation Results



| Metric | Value |

|---|---:|

| GPS observations | 1,315 |

| Unique consecutive arc sequence length | 752 |

| Consecutive arc transitions | 751 |

| Connected transitions | 318 |

| Disconnected transitions | 433 |

| Directed connectivity | 42.34% |



\## 3. Connectivity Calculation



Directed connectivity is calculated as:



Connected transitions / Total consecutive transitions × 100



Using the observed values:



318 / 751 × 100 = 42.34% (approximately)



\## 4. Interpretation



A connected transition indicates that consecutive road arcs satisfy the directed connectivity condition used in the evaluation.



A disconnected transition indicates that the consecutive arcs do not satisfy this condition.



The analysis identified disconnected transitions in the examined sequence. This highlights the importance of considering road-network structure and sequence consistency during GPS trajectory reconstruction.



\## 5. Limitations



This is a preliminary connectivity diagnostic and should not be interpreted as classification accuracy or final map-matching performance.



The result is dependent on the trajectory, arc representation, and connectivity-check implementation.



\## 6. Next Steps



\- Investigate disconnected transitions.

\- Validate arc representation and directionality.

\- Analyze trajectory and road-network structure.

\- Compare connectivity before and after map matching.

\- Evaluate model outputs using appropriate performance metrics.



