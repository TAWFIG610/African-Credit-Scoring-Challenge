def add_features(train_data, test_data):
    # Loan to repay ratio
    train_data['loan_to_repay_ratio'] = train_data['Total_Amount'] / (train_data['Total_Amount_to_Repay'] + 1)
    test_data['loan_to_repay_ratio'] = test_data['Total_Amount'] / (test_data['Total_Amount_to_Repay'] + 1)

    # Amount-duration interaction feature
    train_data['amount_duration_interaction'] = train_data['Total_Amount'] * train_data['duration']
    test_data['amount_duration_interaction'] = test_data['Total_Amount'] * test_data['duration']

    return train_data, test_data
