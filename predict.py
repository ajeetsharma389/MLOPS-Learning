import mlflow
import numpy as np

if __name__ == "__main__":
    # 1. Point to your registered model
    model_uri = "models:/BasicLinearModel/1"
    print(f"Fetching {model_uri} from MLflow Registry...")
    
    # 2. Load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # 3. Define brand new data points to test the model (e.g., X = 6 and X = 10)
    # Based on the trained line (y = 2x + 1), we expect outputs near 13 and 21
    new_data = np.array([[6], [10]])
    
    # 4. Generate predictions
    raw_predictions = loaded_model.predict(new_data)
    
    print("\n--- Raw Debug Info ---")
    print(f"Type of raw predictions: {type(raw_predictions)}")
    print(f"Content of raw predictions: {raw_predictions}")
    
    # 5. Convert explicitly to a flat numpy array to guarantee clean printing
    flat_predictions = np.array(raw_predictions).flatten()
    
    print("\n--- Final Inference Results ---")
    for x_val, y_pred in zip(new_data.flatten(), flat_predictions):
        print(f"Input X: {x_val} -> Predicted Y: {y_pred:.2f}")
