from flask import Flask, jsonify  
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

try:
    from utils import status_calc
except ImportError:
    print("Error: 'utils.py' is missing or 'status_calc' function is not defined.")
    status_calc = None  # Prevent crashes

app = Flask(__name__)  

# Define backtest function
def backtest():
    try:
        data_df = pd.read_csv("keystats.csv", index_col="Date")  # Read CSV file
    except FileNotFoundError:
        return {"error": "Error: 'keystats.csv' file not found."}

    if status_calc is None:
        return {"error": "Error: 'status_calc' function is not available."}

    data_df.dropna(axis=0, how="any", inplace=True)  # Drop missing values

    features = data_df.columns[6:]
    X = data_df[features].values
    y = list(status_calc(data_df["stock_p_change"], data_df["SP500_p_change"], outperformance=10))
    z = np.array(data_df[["stock_p_change", "SP500_p_change"]])

    # Train-test split
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=0.2, random_state=0)

    # Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    # Model Predictions
    y_pred = clf.predict(X_test)
    accuracy_score = clf.score(X_test, y_test)
    
    # Handle case where precision_score might fail (e.g., only one class in y_test)
    try:
        precision_score_value = precision_score(y_test, y_pred, zero_division=0)
    except Exception:
        precision_score_value = 0.0  # Set to 0 if an error occurs

    num_positive_predictions = int(sum(y_pred))  # Convert np.int64 to standard int
    if num_positive_predictions == 0:
        return {
            "error": "No stocks predicted!",
            "accuracy": round(float(accuracy_score), 2),
            "precision": round(float(precision_score_value), 2)
        }

    # Compute Returns
    stock_returns = 1 + z_test[y_pred, 0] / 100
    market_returns = 1 + z_test[y_pred, 1] / 100
    avg_predicted_stock_growth = sum(stock_returns) / num_positive_predictions
    index_growth = sum(market_returns) / num_positive_predictions

    percentage_stock_returns = 100 * (avg_predicted_stock_growth - 1)
    percentage_market_returns = 100 * (index_growth - 1)
    total_outperformance = percentage_stock_returns - percentage_market_returns

    return {
        "accuracy": round(float(accuracy_score), 2),
        "precision": round(float(precision_score_value), 2),
        "total_trades": num_positive_predictions,
        "stock_return": round(float(percentage_stock_returns), 1),
        "market_return": round(float(percentage_market_returns), 1),
        "strategy_outperformance": round(float(total_outperformance), 1)
    }

# Flask Routes
@app.route('/')
def home():
    return "Flask is running"

@app.route("/stocks_prediction")
def stock_prediction():
    result = backtest()  # Call backtest function
    return jsonify(result)  

if __name__ == "__main__":
    app.run(debug=True)
