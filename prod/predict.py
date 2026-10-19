import mlflow
import numpy as np

if __name__ == "__main__":
    # Point directly to the alias contract instead of a volatile version number
    # Format: "models:/<model_name>@<alias>"
    model_uri = "models:/BasicLinearModel@champion"
    
    print(f"Connecting to live endpoint: Fetching current '{model_uri}'...")
    
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        new_data = np.array([[6], [10]])
        raw_predictions = loaded_model.predict(new_data)
        flat_predictions = np.array(raw_predictions).flatten()
        
        print("\n--- Live Production Traffic Output ---")
        for x_val, y_pred in zip(new_data.flatten(), flat_predictions):
            print(f"Input X: {x_val} -> Predicted Y: {y_pred:.2f}")
            
    except Exception as e:
        print(f"\n[Deployment Error]: Could not pull the champion model. Reason: {e}")
