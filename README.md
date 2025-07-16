# Comparison of Classical ML Models and MLPs 

## 1. Project Overview  
This project evaluates two classical ensemble methods (AdaBoost, Bagging) and two multilayer perceptron (MLP) variants on a binary classification task: predicting airline passenger satisfaction. Using the “Airline Passenger Satisfaction” dataset (129,880 records, 22 features), compare model performance on original, normalized, and standardized data.

## 2. Objectives  
- Apply and tune AdaBoost and Bagging classifiers  
- Build and optimize two MLP architectures  
- Compare accuracy, precision, recall, F1-score on different feature scalings  
- Assess the impact of hyperparameter tuning and data transformation

## 3. Data Preparation  
1. **Dataset:** 129,880 rows, features include gender, customer type, age, travel class, service ratings, delays.  
2. **Cleaning:** Removed features with correlation < 0.1.  
3. **Feature variants:**  
   - Original (Raw)  
   - Min–Max normalized  
   - Standardized (μ = 0, σ = 1)

## 4. Models  

### 4.1 Classic Models  
- **AdaBoostClassifier**  
  - Base and hyperparameter-tuned versions (n_estimators, tree depth)  
- **BaggingClassifier**  
  - Base and tuned versions (n_estimators, max_depth)
 
### 4.2 Multilayer Perceptrons (MLP)  
- **Base architecture:**  
  - Input: 17 neurons  
  - Hidden: 2×256 ReLU layers  
  - Output: 1-neuron sigmoid  
  - Loss: binary_crossentropy, optimizer: SGD, Adam, RMSprop
- **Variants:**  
  - Optimizers: SGD, Adam, RMSprop  
  - Layer sizes: 64–512, dropout 0.2  
  - Learning rates and epochs (50 vs. 200)
 
  ## 5. Results  

| Model                     | Data Variant          | Accuracy |
|---------------------------|-----------------------|----------|
| AdaBoost (base)           | original/std/norm.    | 0.9235   |
| AdaBoost (tuned)          | original/std/norm.    | 0.9576   |
| Bagging (base)            | normalized            | 0.9588   |
| Bagging (tuned)           | original              | 0.961    |
| Best MLP                  | normalized/standard.  | ~0.96    |

- **Top classical model:** Tuned Bagging on original data (Acc ≈ 0.961).  
- **Top MLP:** Adam-optimized network on normalized data (Acc ≈ 0.96, F1 ≈ 0.95–0.97).

## 6. Conclusions  
1. Bagging slightly outperforms MLPs in this binary task. This shows that it’s best to start by understanding the problem and selecting the appropriate tool to solve it, rather than choosing the most advanced tool for simple problems.
2. Feature scaling markedly improves MLP stability and performance.  
3. For simpler tasks, classical methods often suffice; MLPs excel with more complex, high-dimensional data.
