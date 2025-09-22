# Import MLflow for experiment tracking
import mlflow
# MLflow module for scikit-learn models (provides log_model, load_model, etc.)
import mlflow.sklearn
# Import Iris dataset
from sklearn.datasets import load_iris
# RandomForestClassifier for classification
from sklearn.ensemble import RandomForestClassifier
# Utility to split dataset into training and testing sets
from sklearn.model_selection import train_test_split
# Evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix
# Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import dagshub for MLflow + DVC integration
import dagshub
# Initialize connection to a DagsHub repository
# - repo_owner: account/organization name on DagsHub
# - repo_name: repository where metrics, artifacts, and models will be logged
# - mlflow=True: enables MLflow logging integration
mlflow.set_tracking_uri("https://dagshub.com/DeepuML/mlflow-dagshub.mlflow")
# Set the tracking URI so MLflow logs data to the DagsHub MLflow server
dagshub.init(repo_owner='DeepuML', repo_name='mlflow-dagshub', mlflow=True)


# Load Dataset

iris = load_iris()        # Load Iris dataset (features + labels)
X = iris.data             # Feature matrix
y = iris.target           # Target labels (species classes)

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define Model Parameters

max_depth = 1         # Limit maximum depth of trees (helps prevent overfitting)
n_estimators = 100    # Number of trees in the forest

# Set MLflow Experiment

# Creates a new experiment (if not already present) called "iris-rf"
mlflow.set_experiment('iris-rf')


# Start MLflow Run

with mlflow.start_run():

    # Initialize the Random Forest classifier with defined hyperparameters
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

    # Train the model on training data
    rf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf.predict(X_test)

    
    # Evaluate Model

    accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy of predictions

    
    # Log Metrics & Params
    
    mlflow.log_metric('accuracy', accuracy)       # Log accuracy metric
    mlflow.log_param('max_depth', max_depth)      # Log hyperparameter: max depth
    mlflow.log_param('n_estimators', n_estimators) # Log hyperparameter: number of trees

    
    # Confusion Matrix Plot
    
    cm = confusion_matrix(y_test, y_pred)         # Compute confusion matrix
    plt.figure(figsize=(6,6))                     # Set figure size
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=iris.target_names,            # Predicted labels
        yticklabels=iris.target_names             # Actual labels
    )
    plt.ylabel('Actual')                          # Y-axis label
    plt.xlabel('Predicted')                       # X-axis label
    plt.title('Confusion Matrix')                 # Plot title
    
    # Save confusion matrix plot locally
    plt.savefig("confusion_matrix.png")

    
    # Log Artifacts & Model
    
    mlflow.log_artifact("confusion_matrix.png")   # Log confusion matrix image
    mlflow.log_artifact(__file__)                 # Log this script as an artifact
    mlflow.sklearn.log_model(rf, "random_forest") # Log trained Random Forest model

    
    # Add Metadata Tags
    
    mlflow.set_tag('author','rahul')              # Tag: experiment author
    mlflow.set_tag('model','random forest')       # Tag: model type

    
    # Print Result in Console

    print('accuracy:', accuracy)
