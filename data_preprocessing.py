import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def load_data(train_path, test_path, economic_data_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    economic_data = pd.read_csv(economic_data_path)
    return train_data, test_data, economic_data

def preprocess_data(train_data, test_data):
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
    test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

    # Preprocessing: separate features and target
    X = train_data.drop(columns=['ID', 'target'])
    y = train_data['target']
    
    return X, y, test_data
