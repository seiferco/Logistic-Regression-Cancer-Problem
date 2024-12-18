# Logistic Regression with Stochastic Gradient Descent

This project implements a Logistic Regression model trained using Stochastic Gradient Descent (SGD). The code includes functionality for training the model, evaluating its performance, and performing k-fold cross-validation.

---

## Prerequisites

Before running this code, make sure you have the following installed:

- Python 3.x
- NumPy
- Matplotlib

You can install the necessary packages using the following commands:

```bash
pip install numpy matplotlib
```

Ensure that the following files are in the same directory as the script:

- `train_cancer.csv`: Training dataset with features and labels.
- `test_cancer_pub.csv`: Test dataset with features only.

---

## Features

### Key Functions

1. **logistic(z)**: Computes the logistic (sigmoid) function for the input vector `z`.

2. **calculateNegativeLogLikelihood(X, y, w)**: Calculates the negative log-likelihood for the Logistic Regression model.

3. **trainLogistic(X, y, max_iters, step_size)**: Implements SGD to optimize the weight vector `w` and minimize the negative log-likelihood.

4. **dummyAugment(X)**: Adds a bias column of ones to the input feature matrix `X`.

5. **kFoldCrossVal(X, y, k)**: Performs k-fold cross-validation and returns the mean and standard deviation of accuracy.

6. **loadData()**: Loads training and test datasets from CSV files.

### Logging and Visualization

- Logs key steps and results during training and validation.
- Plots training loss curves with and without a bias term.

---

## Usage

1. **Set Global Parameters**
   Modify the global parameters for SGD as needed:

   ```python
   step_size = 0.001
   max_iters = 100000
   ```

2. **Run the Script**
   Execute the script using:

   ```bash
   python logistic_regression.py
   ```

3. **Outputs**
   - Training loss curves.
   - Logs of learned weights, training accuracy, and cross-validation results.
   - Test predictions saved to `test_predicted.csv`.

---

## Results

### Training
- Models are trained both with and without a bias term, and their losses are compared.

### Cross-Validation
- The script evaluates model performance using k-fold cross-validation for varying values of `k`.

### Test Submission
- Final predictions for the test set are saved to `test_predicted.csv`.

---

## File Structure

```
|-- logistic_regression.py  # Main script
|-- train_cancer.csv        # Training dataset
|-- test_cancer_pub.csv     # Test dataset
```

---

## Example Logs

```plaintext
2024-12-18 12:00:00 INFO     Loading data
2024-12-18 12:00:01 INFO     Training logistic regression model (No Bias Term)
2024-12-18 12:00:02 INFO     Learned weight vector: [0.1234, 0.5678, ...]
2024-12-18 12:00:02 INFO     Train accuracy: 95.43%
...
2024-12-18 12:05:00 INFO     5-fold Cross Val Accuracy -- Mean (stdev): 94.56% (2.34%)
```

---

## Customization

You can adjust the following parameters to fine-tune the model:

- `step_size`: Learning rate for SGD.
- `max_iters`: Number of iterations for training.
- `k` in `kFoldCrossVal`: Number of folds for cross-validation.

---

## License

This project is provided as-is with no specific license. Feel free to modify and use it for educational purposes.
