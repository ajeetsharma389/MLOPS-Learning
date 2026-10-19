import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from mlflow import MlflowClient

if __name__ == "__main__":
    MODEL_NAME = "BasicLinearModel"
    
    with mlflow.start_run() as run:
        X = np.array([[1],[2],[3],[4],[5]])
        Y = np.array([3,5,7,9,11])
        
        model = LinearRegression()
        model.fit(X, Y)

        # Calculate performance metric
        accuracy_score = float(model.score(X, Y))
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("score", accuracy_score)
        
        signature = infer_signature(X, model.predict(X))

        # 1. Log and register the model artifact
        model_info = mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model",       
            registered_model_name=MODEL_NAME,  
            signature=signature,
            input_example=X[:5]          
        )
        
        # 2. DEVOPS GATE: Automated Quality Assessment
        PRODUCTION_THRESHOLD = 0.90
        
        if accuracy_score >= PRODUCTION_THRESHOLD:
            print(f"\n[CI/CD Gate Passed]: Model score ({accuracy_score:.2f}) meets production requirements.")
            
            # Initialize MLflow Client to manipulate governance markers
            client = MlflowClient()
            model_version = model_info.registered_model_version
            
            # Dynamically assign the production alias to this new version
            # This instantly strips the alias from the old version and gives it to the new one
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="champion",
                version=str(model_version)
            )
            print(f"[CI/CD Route]: Promoted Version {model_version} to 'champion' alias status.")
        else:
            print(f"\n[CI/CD Gate Failed]: Score ({accuracy_score:.2f}) too low. Not promoting.")
