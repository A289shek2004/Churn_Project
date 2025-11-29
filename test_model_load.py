import pickle, os

# Path setup
basedir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(basedir, 'churn_model.pkl')

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Print what we got
print("Loaded model type:", type(model))
