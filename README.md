# 🌸 Iris Flower Species Classification

This project is part of my **Summer Training in Advanced Python for Machine Learning & AI**.
The goal is to build a machine learning model that classifies iris flowers into three species (*Setosa, Versicolor, Virginica*) based on their sepal and petal measurements.

---

## 📌 Project Overview

* Implemented a **K-Nearest Neighbours (KNN)** classifier using **Scikit-learn**.
* Performed **Exploratory Data Analysis (EDA)** with pair plots and box plots to understand data distribution.
* Achieved **96.67% accuracy** on the test dataset.
* Visualized results using **confusion matrix heatmaps** for detailed performance insights.

---

## 📂 Dataset

The project uses the **Iris dataset**, which is built into **Scikit-learn** (`load_iris`).

* 150 samples
* 3 species: *Setosa, Versicolor, Virginica*
* 4 features:

  * Sepal Length (cm)
  * Sepal Width (cm)
  * Petal Length (cm)
  * Petal Width (cm)

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy** → Data handling
* **Matplotlib, Seaborn** → Data visualization
* **Scikit-learn** → Machine learning model

---

## 🚀 Implementation Steps

1. Load and explore the Iris dataset.
2. Perform **EDA** with pair plots and box plots.
3. Split dataset into training (80%) and testing (20%).
4. Train a **KNN classifier** with `k=5`.
5. Evaluate model using **accuracy score** and **confusion matrix**.
6. Visualize results with plots.

---

## 📊 Results

* **Accuracy:** 96.67%
* **Confusion Matrix:** Perfect classification for *Setosa* and *Virginica*, with only one misclassification between *Versicolor* and *Virginica*.

---

## 🔮 Future Enhancements

* Apply other ML algorithms (Logistic Regression, SVM, Decision Trees) for comparison.
* Implement **hyperparameter tuning** (GridSearchCV).
* Build a **simple GUI / web app** for interactive predictions.

---

## 📷 Sample Visualizations

* Pair Plot of features by species
* Box Plot of feature distributions
* Confusion Matrix heatmap

*(Plots will be visible when you run the notebook/code.)*

---

## ▶️ How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/saina25/iris-classification.git
   ```
2. Navigate to the project folder:

   ```bash
   cd iris-classification
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *(Or manually install pandas, seaborn, matplotlib, scikit-learn)*
4. Run the script:

   ```bash
   python iris_classification.py
   ```

---

## 📖 References

* Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems*.
* [Scikit-learn Documentation](https://scikit-learn.org/)
* [Seaborn Documentation](https://seaborn.pydata.org/)
