# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

print("Starting Iris Classification ...")

# Loading the dataset
# Scikit-learn has the Iris dataset built-in.
iris = load_iris()
X = iris.data
y = iris.target

# Creating a pandas DataFrame for easier visualization
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['species'] = y

# Mapping target numbers to actual species names for plotting
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
print("Dataset loaded successfully.")
print("First 5 rows of the dataset:")
print(df.head())


# Exploratory Data Analysis (EDA) - Displaying Plots
print("\n--- Displaying Plots ---")

# Displaying the Pair Plot (for Figure 3.1)
print("\nDisplaying Pair Plot...")
sns.pairplot(df, hue='species', palette='viridis')
plt.suptitle('Figure 3.1: Pair Plot of Iris Features', y=1.02) # Add a title above the plot
plt.show() # This command displays the plot in the notebook

# Displaying the Box Plot (for Figure 3.2)
print("\nDisplaying Box Plot...")
plt.figure(figsize=(12, 8))

# Melting the DataFrame to make it suitable for a single boxplot with hue
df_melted = df.melt(id_vars='species', var_name='feature', value_name='measurement')
sns.boxplot(x='species', y='measurement', hue='feature', data=df_melted)
plt.title('Figure 3.2: Box Plot of Iris Features by Species')
plt.xlabel('Species')
plt.ylabel('Measurement (cm)')
plt.legend(title='Feature')
plt.tight_layout()
plt.show() # This command displays the plot


# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nData split into training and testing sets.")
print(f"Training set has {X_train.shape[0]} samples.")
print(f"Testing set has {X_test.shape[0]} samples.")


# Creating and Training the K-Nearest Neighbors (KNN) Model
print("\n--- Model Training ---")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("KNN model trained successfully with K=5.")


# Making Predictions and Evaluating the Model
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix (Text):")
print(conf_matrix)

# Displaying the Confusion Matrix Plot
print("\nDisplaying Confusion Matrix Plot...")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for KNN Model')
plt.show() # This command displays the plot in the notebook


print("Iris Classification finished successfully...")
