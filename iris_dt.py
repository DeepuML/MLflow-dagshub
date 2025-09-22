# Import MLflow for experiment tracking
import mlflow
# MLflow module for scikit-learn models (provides log_model, load_model, etc.)
import mlflow.sklearn
# Load the Iris dataset from scikit-learn
from sklearn.datasets import load_iris
# DecisionTreeClassifier for classification
from sklearn.tree import DecisionTreeClassifier
# Utility to split dataset into training and testing sets
from sklearn.model_selection import train_test_split
# Metrics for evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
# Matplotlib and Seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# DagsHub integration for MLflow tracking
import dagshub
# Initialize DagsHub connection (links repo to MLflow tracking)
dagshub.init(repo_owner='Deepu', repo_name='MLFlow-DagsHub', mlflow=True)

# Set MLflow tracking URI to DagsHub server
mlflow.set_tracking_uri("https://github.com/DeepuML/MLflow-dagshub.git")

# Load the iris dataset
iris = load_iris()          # Load built-in Iris dataset (features and labels)
X = iris.data               # Feature matrix
y = iris.target             # Target labels (species of iris)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the parameter for the Decision Tree model
max_depth = 1  # Limit tree depth to avoid overfitting

# Set the experiment name in MLflow (creates experiment if it doesn’t exist)
mlflow.set_experiment('iris-dt')

# Start a new MLflow run (all logs/artifacts will be grouped under this run)
with mlflow.start_run():

    # Initialize Decision Tree classifier with defined max depth
    dt = DecisionTreeClassifier(max_depth=max_depth)

    # Train the Decision Tree on training data
    dt.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = dt.predict(X_test)

    # Calculate accuracy of predictions
    accuracy = accuracy_score(y_test, y_pred)

    # Log accuracy metric to MLflow
    mlflow.log_metric('accuracy', accuracy)

    # Log the hyperparameter (max_depth) used for this model
    mlflow.log_param('max_depth', max_depth)

    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)   # Generate confusion matrix
    plt.figure(figsize=(6,6))               # Define plot size
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=iris.target_names,      # X-axis = predicted class labels
        yticklabels=iris.target_names       # Y-axis = actual class labels
    )
    plt.ylabel('Actual')                    # Label for y-axis
    plt.xlabel('Predicted')                 # Label for x-axis
    plt.title('Confusion Matrix')           # Title of the plot
    
    # Save the confusion matrix plot locally
    plt.savefig("confusion_matrix.png")

    # Log confusion matrix image as an artifact in MLflow
    mlflow.log_artifact("confusion_matrix.png")

    # Log this script itself as an artifact (helps reproduce results)
    mlflow.log_artifact(__file__)

    # Log the trained Decision Tree model to MLflow
    mlflow.sklearn.log_model(dt, "decision tree")

    # Add tags to the run (metadata for better organization)
    # mlflow.set_tag('author','nitish')          # Who created the run
    # mlflow.set_tag('model','decision tree')    # Model type used

    # Print the accuracy in console
    print('accuracy', accuracy)
