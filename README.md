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
│   └── BTL_ML.ipynb             
├── reports/
│   └── HCMUT_Report_ML.pdf      
├── output/
│   ├── processed_X.npy          
│   └── processed_y.npy          
└── README.md                    
```
## Link

[Notebook](notebooks/BTL_ML.ipynb)

[Open In Colab](https://colab.research.google.com/drive/1TYWLOIielweUY4uoOyj6N148QVoRSwPN)

[PDF Report](reports/HCMUT_Report_ML.pdf)

