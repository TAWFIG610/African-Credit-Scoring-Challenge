from code.data_preprocessing import load_data, preprocess_data
from code.feature_engineering import add_features
from code.model import create_model
from code.train_model import train_model
from code.utils import calculate_class_weights
from sklearn.model_selection import train_test_split

# Load data
train_data, test_data, economic_data = load_data('data/Train.csv', 'data/Test.csv', 'data/economic_indicators.csv')

# Feature Engineering
train_data, test_data = add_features(train_data, test_data)

# Preprocessing
X, y, test_data = preprocess_data(train_data, test_data)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class weights
class_weights = calculate_class_weights(y_train)

# Create and train the model
model = create_model(X_train.shape[1])
trained_model = train_model(model, X_train, y_train, X_val, y_val, class_weights)

# Save final model
trained_model.save('final_model.keras')
