import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import coremltools as ct

# Step 1: Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

data = {
    'amount': np.random.uniform(10, 1000, n_samples),
    'day_of_week': np.random.randint(0, 7, n_samples),  # 0-6 (Sunday-Saturday)
    'month': np.random.randint(1, 13, n_samples),       # 1-12 (January-December)
    'category': np.random.choice(['Groceries', 'Dining', 'Transportation', 'Shopping', 'Other'], n_samples)
}

df = pd.DataFrame(data)

# Step 2: Prepare features and target
X = df[['amount', 'day_of_week', 'month']]
y = df['category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model (optional)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy on test set: {accuracy:.2f}")

# Step 5: Convert to Core ML model
# Define the input and output features
feature_descriptions = {
    'amount': ct.features.FeatureDescription(type=ct.features.DoubleTensor, shape=(1,)),
    'day_of_week': ct.features.FeatureDescription(type=ct.features.IntegerTensor, shape=(1,)),
    'month': ct.features.FeatureDescription(type=ct.features.IntegerTensor, shape=(1,))
}
output_description = ct.features.FeatureDescription(type=ct.features.StringTensor, shape=(1,))

# Convert the model
mlmodel = ct.converters.sklearn.convert(
    model,
    input_features=[
        ct.features.TensorType(name='amount', shape=(1,), dtype=np.float64),
        ct.features.TensorType(name='day_of_week', shape=(1,), dtype=np.int64),
        ct.features.TensorType(name='month', shape=(1,), dtype=np.int64)
    ],
    output_features=[ct.features.TensorType(name='category', shape=(1,), dtype=np.str_)]
)

# Set metadata
mlmodel.author = 'Your Name'
mlmodel.short_description = 'Budget Classifier using CoreML for EGK BudgetAI'
mlmodel.version = '1.0'

# Save as .mlpackage
mlmodel.save('BudgetClassifier.mlpackage')

print("BudgetClassifier.mlpackage has been generated successfully!")
