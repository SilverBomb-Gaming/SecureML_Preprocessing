from preprocessing import scale_features, encode_categorical, select_features
from preprocessing import save_preprocessed_data
import pandas as pd

# Example dataset loading (replace with actual dataset loading)
data = {
    "feature1": [10, 20, 30, 40, 50],
    "feature2": [100, 200, 300, 400, 500],
    "category": ["A", "B", "A", "C", "B"]
}
df = pd.DataFrame(data)

# Apply feature scaling (choose "standard" or "minmax")
df_scaled = scale_features(df.drop(columns=["category"]), method="standard")

# Apply categorical coding
df_encoded = encode_categorical(df, method="label") # Change to "label" for Label Encoding

# Display results
print("Original Data:\n", df)
print("\nScaled Data:\n", df_scaled)

# Example dataset with categorical values
data = {
    "feature1": [10, 20, 30, 40, 50],
    "feature2": [100, 200, 300, 400, 500],
    "category": ["A", "B", "A", "C", "B"]
}
df = pd.DataFrame(data)

# Apply feature scaling
df_scaled = scale_features(df.drop(columns=["category"]), method="standard")

# Apply categorical encoding
df_encoded = encode_categorical(df, method="label")  # Change to "label" for Label Encoding
df_selected = select_features(df_encoded)

# Display results
print("Original Data:\n", df)
print("\nScaled Data:\n", df_scaled)
print("\nEncoded Data:\n", df_encoded)
print("\nSelected Features:\n", df_selected.head())

# Save the final processed data
save_preprocessed_data(df_selected)
