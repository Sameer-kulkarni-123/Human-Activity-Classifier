from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # or any model
from sklearn.datasets import load_iris  # example dataset
import matplotlib.pyplot as plt

# Step 1: Load Data (you can replace this with your own dataset)
# X, y = load_iris(return_X_y=True)


# Step 2: Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train a Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Predict on Test Set
y_pred = model.predict(X_test)

# Step 5: Generate and Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()