import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = r"C:\Users\Deepanshu Tewatia\Downloads\Clean_Dataset[1].csv"
df = pd.read_csv(file_path)

# Drop unnecessary column if needed
df = df.drop(columns=['Unnamed: 0'])  # Optional: drops the index column

# Define target and features
target_column = 'price'
X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert categorical variables to dummy/encoded variables
X = pd.get_dummies(X, drop_first=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Show feature coefficients
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print("\nFeature Coefficients:")
print(coeff_df)
