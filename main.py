from featureExtractor import featureExtraction
from pycaret.classification import load_model, predict_model
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the pre-trained phishing detection model
model = load_model('model/phishingdetection')

# Sample synthetic dataset with labeled URLs (1 = Phishing, 0 = Legitimate)
sample_data = {
    'url': [
        'https://bafybeifqd2yktzvwjw5g42l2ghvxsxn76khhsgqpkaqfdhnqf3kiuiegw4.ipfs.dweb.link/',
        'http://about-ads-microsoft-com.o365.frc.skyfencenet.com',
        'https://chat.openai.com',
        'https://github.com/',
        'http://paypal-secure-login.com',
        'https://www.google.com',
        'http://bankofamerica-secure-login.com',
        'https://www.wikipedia.org',
    ],
    'label': [1, 1, 0, 0, 1, 0, 1, 0]  # Corresponding labels
}

# Convert to DataFrame
df = pd.DataFrame(sample_data)

# Extract features for each URL
features = []
for url in df['url']:
    extracted_features = featureExtraction(url)
    print(f"Extracted Features for {url}:{extracted_features}")  # Detailed Debugging
    features.append(extracted_features)

# Convert features to DataFrame
features_df = pd.DataFrame(features)

# Predict using the model
predictions = predict_model(model, data=features_df)
print("Raw Predictions:", predictions)  # Detailed Debugging

# Calculate and print the accuracy
try:
    y_true = df['label']
    y_pred = predictions['prediction_label'].astype(int)  # Ensure correct dtype
    print(f"True Labels: {y_true.tolist()}")  # Debugging
    print(f"[INFO] Predicted Labels: {y_pred.tolist()}")  # Debugging
    accuracy = accuracy_score(y_true, y_pred)
except Exception as e:
    print("[ERROR] Accuracy Calculation Failed:", e)
    accuracy = None
print(f"Accuracy: {accuracy * 100:.2f}%")
if accuracy is None or pd.isna(accuracy):
    print("Error: Accuracy calculation failed. Check labels and predictions.")