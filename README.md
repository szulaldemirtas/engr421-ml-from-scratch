# Machine Learning Algorithms from Scratch

Core machine learning methods implemented in NumPy without ML libraries, for the Introduction to Machine Learning course (ENGR421) at Koç University, Fall 2024.

| Folder | Method | Data |
|---|---|---|
| `01-naive-bayes` | Bernoulli naive Bayes classifier | Pen-based handwritten digits |
| `02-logistic-regression` | Multiclass discrimination by regression with sigmoid outputs, trained by gradient descent | Fashion-MNIST |
| `03-nonparametric-regression` | Regressogram, running mean smoother and Gaussian kernel smoother | 1D regression data |
| `04-decision-tree-regression` | Regression tree with pre-pruning by minimum leaf size | 1D regression data |
| `05-em-clustering` | Gaussian mixture model clustering with the EM algorithm | 2D synthetic data |
| `06-roc-auroc` | ROC curve and area under the curve computed from predicted probabilities | Two classifiers' outputs |

Each folder has the script, its data and the generated plots (PDF). Run a script from inside its folder, e.g. `cd 03-nonparametric-regression && python nonparametric_regression.py`.

The Fashion-MNIST CSV for `02-logistic-regression` (200 MB) is not included; it is the standard [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset flattened to one image per row.
