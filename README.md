# Hotel Booking Demand - Machine Learning Project

## Course Information
* **Course Name:** Machine Learning
* **Course Code:** CO3013
* **Semester:** Semester 252
* **Academic Year:** 2025 - 2026
* **Institution:** Ho Chi Minh City University of Technology (HCMUT)
* **Instructor:** Truong Vinh Lan

## Group Members
| Full Name | Student ID | Email |
| :--- | :--- | :--- |
| Nguyen Ho Quang Khai | 2352538 | khai.nguyen.hcmut@hcmut.edu.vn |
| Nguyen Tran Trong Tuyen | 2353283 | tuyen.nguyen.trong@hcmut.edu.vn |
| Nguyen Quoc Thang | 2353122 | thang.nguyen.quoc@hcmut.edu.vn |
| RISPAL Noa Marc Pierre | 2560118 | pierre.rispalexc@hcmut.edu.vn |
| Quach Hoang Phu | 2352922 | phu.quach.hoang@hcmut.edu.vn |

## Project Objectives
The primary goal of this project is to predict hotel booking cancellations using the "Hotel Booking Demand" dataset. The objectives include:
1.  **Exploratory Data Analysis (EDA):** Identifying patterns and correlations in booking behavior.
2.  **Traditional ML Pipeline:** Implementing Preprocessing, PCA, and evaluating multiple models (SVM, Random Forest, Logistic Regression) across 4 specific Test Cases (TC1-TC4).
3.  **Advanced Deep Learning:** Comparing traditional models with modern architectures specialized for tabular data such as **TabNet**, **MLP**, and **NODE**.
4.  **Performance Optimization:** Maximizing F1-Score and Recall to provide actionable business insights for hotel management.

## Data Loading 

The dataset used is [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Kaggle)

The datasets .csv is in data/ folder, if you have changed its location you need to update the `data_path` variable in the notebook to point to your file location 

## Instructions for Running the Notebook

### 1. Required Libraries
The project leverages GPU acceleration via **NVIDIA RAPIDS (cuML)**. Ensure you have a compatible environment (like Google Colab with GPU T4/A100).

```bash
# Core Libraries
pip install pandas numpy matplotlib seaborn scikit-learn pytorch-tabular pytorch-tabnet

# GPU Accelerated Libraries (RAPIDS)
pip install cudf-cu12 cuml-cu12 cpupy-cu12

```

## Project structure 
```bash
├── data/
│   └── hotel_bookings.csv      
├── notebooks/
│   └── hotelBookingDemand.ipynb             
├── reports/
│   └── HCMUT_Report_ML.pdf      
├── output/
│   ├── processed_X.npy          
│   └── processed_y.npy          
└── README.md                    
```
## Link

[Notebook](notebooks/hotelBookingDemand.ipynb)

[Open In Colab](https://colab.research.google.com/drive/1TYWLOIielweUY4uoOyj6N148QVoRSwPN)

[PDF Report](reports/HCMUT_Report_ML.pdf)

## Configuration & Hyperparameters

The project uses a dictionary-based configuration for transparency and reproducibility.

### 1. General Configuration (`config_task_1`)
* **`data_path`**: Path to the raw `hotel_bookings.csv` file.
* **`feature_output_path`**: Directory where processed NumPy arrays are saved to skip preprocessing in future runs.
* **`test_size`**: The 20% proportion of data held out for final evaluation.
* **`random_state`**: Fixed seed (42) for reproducible data splits and model initialization.

### 2. Preprocessing & Feature Engineering
* **`num_cols_names`**: List of 12 numerical features subjected to standardization.
* **`cat_feature_idx`**: Indices of categorical variables processed via **One-Hot Encoding** (resulting in a 39-dimensional space).
* **`pca_n_components`**: Set to 0.95 to retain **95% of the variance** while reducing dimensionality.

### 3. Model Optimizers & Training
#### Traditional ML
LR = Logistic regression, SVM = Support Vector Machine, RF = Random Forest
* **`kernel (SVM)`**: Which function kernel to use.
* **`n_estimators (RF)`**: Number of trees in Random Forest (150).
* **`C (SVM/LR)` (Regularization)**: Controls the penalty strength. Smaller values (TC3) specify stronger regularization.
* **`class_weight (SVM)`**: Applied in TC4 (for SVM) to handle class imbalance by penalizing minority class errors more heavily.
* **`penalty (LR)` (Regularization)**: Specifies the regularization. If it is equal to L2, it adds a penalty proportional to the square of the magnitude of coefficients, discouraging large weights.
* **`gamma (SVM)`**: Defines the influence of a single training example. The 'scale' option automatically calculates gamma based on the number of features
* **`min_samples_split (RF)`**: The minimum number of samples required to split an internal node.


#### Deep Learning (TabNet, MLP, NODE)
The following optimizers are available in the training pipeline to minimize the loss function:

| Optimizer | Key Parameters | Description |
| :--- | :--- | :--- |
| **Adam** | `lr=1e-3` | Standard adaptive optimizer; excellent for fast initial convergence. |
| **RMSprop** | `lr=1e-3` | Adjusts learning rate based on a moving average of the squared gradients. |
| **SGD** | `lr=1e-2`, `momentum=0.9` | Classic gradient descent with momentum to skip local minima. |
| **AdamW** | `lr=5e-4`, `weight_decay=1e-3` | Advanced Adam version with decoupled weight decay for superior generalization. |

**General details:**
* **Learning Rate (LR):** Controls how much the model weights are adjusted with respect to the loss gradient.
* **Momentum:** Helps the optimizer "keep moving" in consistent directions, reducing noise.
* **Weight Decay:** L2 penalty that prevents weights from growing too large, effectively reducing overfitting.
* **Beta 1 & 2:** Control the decay rates of the moving averages of gradients used in Adam-based optimizers.
* **Batch Size**: Set to 1024 for optimal GPU memory utilization.
* **Patience**: Early stopping threshold to prevent overfitting when validation loss stops improving.
* **dropout**: The percentage of neurons randomly deactivated during training to force the model to learn redundant representations and increase robustness.
* **n_splits=X**: Indicates X-fold Stratified Cross-Validation, meaning the model is trained X times on different 80/20 distributions to ensure the results are statistically sound and not due to a lucky data split.

#### TabNet Specific Parameters (`tabnet_config`)
TabNet uses a sequential attention mechanism to choose which features to learn from at each decision step.
* **`n_d` & `n_a`**: Width of the decision prediction layer and the attention embedding layer. Usually kept equal to balance capacity.
* **`n_steps`**: Number of sequential "decision steps" the model takes (typically between 3 and 10).
* **`gamma`**: This coefficient controls feature re-use. A higher gamma prevents the model from using the same feature across different decision steps.
* **`n_independent` & `n_shared`**: Number of GLU (Gated Linear Unit) blocks that are independent for each step vs. shared across all steps.
* **`momentum`**: Momentum for batch normalization (usually set between 0.01 and 0.1).
* **`lambda_sparse`**: Sparsity regularization strength. It encourages the model to use as few features as possible per step to improve interpretability.

#### NODE Specifi Parameters (`config_node`)
NODE is designed to behave like a forest of Deep Decision Trees but remains fully differentiable.
* **`num_layers`**: Number of sequential layers of oblivious decision trees.
* **`num_trees`**: Total number of decision trees within each layer.
* **`tree_depth`**: Depth of each individual tree (usually 6 for optimal performance).
* **`bin_function`**: A soft-selection function (like Entmax or Softmax) used to split nodes in a differentiable way.