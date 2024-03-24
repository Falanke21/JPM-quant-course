import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_logistic_regression(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print("Start standardizing features")
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the logistic regression model
    model = LogisticRegression(random_state=42)
    print("Training logistic regression model...")
    model.fit(X_train_scaled, y_train)

    # Make predictions
    print("Predicting on test data...")
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Display classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # See which feature is the most important one for customer default prediction
    # Get feature importance from logistic regression coefficients
    feature_importance = pd.Series(model.coef_[0], index=X.columns)
    # Sort feature importance values in descending order
    sorted_feature_importance = feature_importance.abs().sort_values(ascending=False)
    print(f"\nFeature importance in descending order:\n{sorted_feature_importance}")

    return model, scaler


def predict_default_probability(model, scaler, X: pd.DataFrame):
    # Standardize the input data
    X_scaled = scaler.transform(X)
    # Predict the probability of default
    proba_default = model.predict_proba(X_scaled)[:, 1]
    return proba_default


def calculate_expected_loss(loan_amount, proba_default, recovery_rate = 0.1):
    # Calculate expected loss
    expected_loss = loan_amount * proba_default * (1 - recovery_rate)
    return expected_loss


if __name__ == "__main__":
    # read csv data for task 3
    data = pd.read_csv("Task_3_and_4_Loan_Data.csv")
    print("Data read successfully!")
    # remove default column and customer_id column
    X = data.drop(['default', 'customer_id'], axis=1)
    y = data['default']

    model, scaler = train_logistic_regression(X, y)

    # Use an example from our dataset
    example_data = X.iloc[0:1]
    print(f"Example data:\n{example_data}")

    proba_default = predict_default_probability(model, scaler, example_data)[0]

    expected_loss = calculate_expected_loss(1000, proba_default)
    print(f"Expected loss of an example loan with amount 1000: {expected_loss:.2f}")
