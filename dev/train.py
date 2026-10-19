import warnings

warnings.filterwarnings("ignore", message=".*OpenSSL.*")

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    
    # 1. Start an MLflow experiment run
    with mlflow.start_run():

        # 2. Generate basic dummy data ( y = 2x + 1 )
        X = np.array([[1], [2], [3], [4], [5]])
        #Y = np.array([3, 5, 7, 9, 11])
        Y = np.array([2, 7, 12, 17, 22])
        
        # 3. Train a basic model
        model = LinearRegression()
        model.fit(X, Y)

        # 4. Log a dummy parameter and metric for experiment tracking
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("score", float(model.score(X, Y)))
        
        # 5. Infer the signature using training data features (X) and predictions
        signature = infer_signature(X, model.predict(X))

        # 6. Log and register the model artifact into MLflow Registry (Single corrected call)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",  # The folder name inside MLflow artifacts
            registered_model_name="BasicLinearModel",  # Registers it to the Model Registry
            signature=signature,
            input_example=X[:5]  # Automatically maps the schema signature
        )
        
        print("Model training complete. Logged and registered to MLflow successfully")
