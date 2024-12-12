import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight  
import matplotlib.pyplot as plt

# Load the preprocessed training and test datasets
train_data = pd.read_csv('processed_training_data.csv')
test_data = pd.read_csv('processed_test_data.csv')
test_labels = pd.read_csv('test_labels.csv')

# Ensure the test_data DataFrame has the expected 'processed_text' column
if 'processed_text' in test_data.columns:
    test_text_column = 'processed_text'
else:
    raise KeyError("Expected 'processed_text' column not found in test_data.csv")

# Convert the processed text into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_full = vectorizer.fit_transform(train_data['processed_text'])
y_train_full = train_data['category']
X_test = vectorizer.transform(test_data[test_text_column])

# Encode labels from strings into numerical format
label_encoder = LabelEncoder()
label_encoder.fit(y_train_full)
y_train_full_encoded = label_encoder.transform(y_train_full)
y_test_encoded = label_encoder.transform(test_labels['category'])

# Define which categories belong to which model
nb_categories = ['business', 'sport', 'entertainment']  # Categories for the Naive Bayes model
svm_categories = ['politics', 'tech']                   # Categories for the SVM model
all_categories = list(set(nb_categories + svm_categories))
all_classes = label_encoder.transform(label_encoder.classes_)

# Compute class weights to handle class imbalances
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=all_classes,
    y=y_train_full_encoded
)
class_weights = dict(zip(all_classes, class_weights_array))
print("\nComputed Class Weights:")
for cls, weight in class_weights.items():
    print(f"Class {cls} ({label_encoder.inverse_transform([cls])[0]}): Weight = {weight:.4f}")

# Display an overview of the dataset
print("\n--- Dataset Overview ---")
print(f"Total number of articles in the dataset: {len(train_data)}")
print("Number of articles per category:")
print(train_data['category'].value_counts())

# Separate the training data for each model based on their categories
nb_data = train_data[train_data['category'].isin(nb_categories)]
svm_data = train_data[train_data['category'].isin(svm_categories)]

# Display how many articles each model gets for its categories
print("\n--- Model-Specific Data ---")
print(f"Number of articles for Naive Bayes categories ({nb_categories}): {len(nb_data)}")
print(nb_data['category'].value_counts())
print(f"\nNumber of articles for SVM categories ({svm_categories}): {len(svm_data)}")
print(svm_data['category'].value_counts())

# Split each model's data into training and held-out sets
nb_indices = nb_data.index
svm_indices = svm_data.index

nb_train_indices, nb_held_indices = train_test_split(nb_indices, test_size=0.33, random_state=42)
svm_train_indices, svm_held_indices = train_test_split(svm_indices, test_size=0.33, random_state=42)

# Show distribution of categories in the split sets
print("\nNaive Bayes Training Set Distribution:")
print(nb_data.loc[nb_train_indices]['category'].value_counts())
print("Naive Bayes Held-Out Set Distribution:")
print(nb_data.loc[nb_held_indices]['category'].value_counts())

print("\nSVM Training Set Distribution:")
print(svm_data.loc[svm_train_indices]['category'].value_counts())
print("SVM Held-Out Set Distribution:")
print(svm_data.loc[svm_held_indices]['category'].value_counts())

# Prepare training data for Naive Bayes and SVM
nb_X_train = vectorizer.transform(nb_data.loc[nb_train_indices]['processed_text'])
nb_y_train = label_encoder.transform(nb_data.loc[nb_train_indices]['category'])

svm_X_train = vectorizer.transform(svm_data.loc[svm_train_indices]['processed_text'])
svm_y_train = label_encoder.transform(svm_data.loc[svm_train_indices]['category'])

# Initialize the models
nb_model = MultinomialNB()

# Using an SGDClassifier for the SVM approach. Setting 'max_iter' to 1 for partial_fit and enabling warm_start to allow iterative training.
sgd_model = SGDClassifier(
    loss='log_loss',          
    penalty='l2',
    alpha=1e-4,               
    max_iter=1,               
    tol=None,                 
    random_state=42,
    learning_rate='optimal',
    n_iter_no_change=5,       
    warm_start=True
)

# Function to perform early stopping training for Naive Bayes
# Note: partial_fit is being used here for demonstration, but MultinomialNB doesn't inherently support early stopping
def early_stopping_training(model, X_train, y_train, model_name):
    best_score = 0
    no_improve_count = 0
    max_no_improve = 5  # Stop if no improvement after a certain number of iterations
    print(f"\nTraining {model_name} with Early Stopping...")
    iteration = 0
    while True:
        iteration += 1
        model.partial_fit(X_train, y_train, classes=all_classes)
        y_train_pred = model.predict(X_train)
        score = accuracy_score(y_train, y_train_pred)
        print(f"Iteration {iteration}: Training Accuracy = {score * 100:.2f}%")
        
        if score > best_score + 1e-4:
            best_score = score
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= max_no_improve:
            print(f"Early stopping triggered after {iteration} iterations. Best Score = {best_score * 100:.2f}%")
            break

# Similar early stopping function for SVM, but incorporating sample weights derived from class weights
def early_stopping_training_sgd(model, X_train, y_train, model_name, class_weights):
    best_score = 0
    no_improve_count = 0
    max_no_improve = 5
    print(f"\nTraining {model_name} with Early Stopping...")
    iteration = 0
    while True:
        iteration += 1
        sample_weights = [class_weights[label] for label in y_train]
        model.partial_fit(X_train, y_train, classes=all_classes, sample_weight=sample_weights)
        y_train_pred = model.predict(X_train)
        score = accuracy_score(y_train, y_train_pred)
        print(f"Iteration {iteration}: Training Accuracy = {score * 100:.2f}%")
        
        if score > best_score + 1e-4:
            best_score = score
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= max_no_improve:
            print(f"Early stopping triggered after {iteration} iterations. Best Score = {best_score * 100:.2f}%")
            break

# Phase 1: Train each model on its own 2/3 training split with early stopping
early_stopping_training(nb_model, nb_X_train, nb_y_train, "Naive Bayes")
early_stopping_training_sgd(sgd_model, svm_X_train, svm_y_train, "SGDClassifier (SVM)", class_weights)

# Evaluate both models on the full test set after Phase 1
print("\n--- Phase 1 Evaluation on Full Test Dataset ---")
y_test_pred_nb_phase1 = nb_model.predict(X_test)
y_test_pred_svm_phase1 = sgd_model.predict(X_test)
print("\nNaive Bayes:")
print(classification_report(y_test_encoded, y_test_pred_nb_phase1, target_names=label_encoder.classes_, zero_division=0))
print(f"Naive Bayes Test Accuracy: {accuracy_score(y_test_encoded, y_test_pred_nb_phase1) * 100:.2f}%")

print("\nSGDClassifier (SVM):")
print(classification_report(y_test_encoded, y_test_pred_svm_phase1, target_names=label_encoder.classes_, zero_division=0))
print(f"SVM Test Accuracy: {accuracy_score(y_test_encoded, y_test_pred_svm_phase1) * 100:.2f}%")

# Phase 2: Each model labels its own held-out 1/3 of data using its current model
print("\n--- Phase 2: Labeling Own Held-Out Data ---")

# Naive Bayes applies predictions to NB-held-out data
nb_X_held = vectorizer.transform(nb_data.loc[nb_held_indices]['processed_text'])
nb_held_preds = nb_model.predict(nb_X_held)
nb_held_probs = nb_model.predict_proba(nb_X_held)
nb_held_pred_labels = label_encoder.inverse_transform(nb_held_preds)

# Create a DataFrame of predictions and save to CSV
nb_held_labeled_data = pd.DataFrame({
    'processed_text': nb_data.loc[nb_held_indices]['processed_text'].values,
    'category': nb_held_pred_labels,
    'probability': nb_held_probs.max(axis=1)
})
nb_held_labeled_data.to_csv('nb_phase2_predictions.csv', index=False)

# SVM applies predictions to SVM-held-out data
svm_X_held = vectorizer.transform(svm_data.loc[svm_held_indices]['processed_text'])
svm_held_preds = sgd_model.predict(svm_X_held)
svm_held_probs = sgd_model.predict_proba(svm_X_held)
svm_held_pred_labels = label_encoder.inverse_transform(svm_held_preds)

# Create a DataFrame of predictions and save to CSV
svm_held_labeled_data = pd.DataFrame({
    'processed_text': svm_data.loc[svm_held_indices]['processed_text'].values,
    'category': svm_held_pred_labels,
    'probability': svm_held_probs.max(axis=1)
})
svm_held_labeled_data.to_csv('svm_phase2_predictions.csv', index=False)

# Phase 3: Combine each model’s original training data with the other model’s newly labeled held-out data
print("\n--- Phase 3: Continuing Training with Combined Data using Early Stopping ---")

# Naive Bayes gets its original training data plus SVM's newly labeled held-out data
nb_combined_training = pd.concat([
    nb_data.loc[nb_train_indices][['processed_text', 'category']],
    svm_held_labeled_data[['processed_text', 'category']]
], ignore_index=True)

# SVM gets its original training data plus NB's newly labeled held-out data
svm_combined_training = pd.concat([
    svm_data.loc[svm_train_indices][['processed_text', 'category']],
    nb_held_labeled_data[['processed_text', 'category']]
], ignore_index=True)

# Save the final combined training sets
nb_combined_training.to_csv('nb_final_training.csv', index=False)
svm_combined_training.to_csv('svm_final_training.csv', index=False)

# Check the class distributions in the combined datasets
print("\nNaive Bayes Final Training Set Class Distribution:")
print(nb_combined_training['category'].value_counts())

print("\nSVM Final Training Set Class Distribution:")
print(svm_combined_training['category'].value_counts())

# Transform the combined training data into TF-IDF features and encode labels
nb_X_final = vectorizer.transform(nb_combined_training['processed_text'])
nb_y_final = label_encoder.transform(nb_combined_training['category'])

svm_X_final = vectorizer.transform(svm_combined_training['processed_text'])
svm_y_final = label_encoder.transform(svm_combined_training['category'])

# Continue training both models with early stopping on the combined data
print("\nContinuing training Naive Bayes model with Early Stopping...")
early_stopping_training(nb_model, nb_X_final, nb_y_final, "Naive Bayes")

print("\nContinuing training SVM model with Early Stopping...")
early_stopping_training_sgd(sgd_model, svm_X_final, svm_y_final, "SGDClassifier (SVM)", class_weights)

# Final evaluation after the models have learned from each other’s held-out data
print("\n--- Final Evaluation After Mutual Learning with Early Stopping ---")
y_test_pred_nb_final = nb_model.predict(X_test)
y_test_pred_svm_final = sgd_model.predict(X_test)

print("\nNaive Bayes:")
print(classification_report(y_test_encoded, y_test_pred_nb_final, target_names=label_encoder.classes_, zero_division=0))
print(f"Naive Bayes Final Accuracy on Test Data: {accuracy_score(y_test_encoded, y_test_pred_nb_final) * 100:.2f}%")

print("\nSGDClassifier (SVM):")
print(classification_report(y_test_encoded, y_test_pred_svm_final, target_names=label_encoder.classes_, zero_division=0))
print(f"SVM Final Accuracy on Test Data: {accuracy_score(y_test_encoded, y_test_pred_svm_final) * 100:.2f}%")

# Generate and display confusion matrices for both models
print("\n--- Confusion Matrices ---")
nb_cm = confusion_matrix(y_test_encoded, y_test_pred_nb_final)
disp = ConfusionMatrixDisplay(confusion_matrix=nb_cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Naive Bayes Confusion Matrix')
plt.show()

svm_cm = confusion_matrix(y_test_encoded, y_test_pred_svm_final)
disp = ConfusionMatrixDisplay(confusion_matrix=svm_cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('SVM Confusion Matrix')
plt.show()

# Generate a micro-averaged Precision-Recall curve to compare both models
print("\n--- Precision-Recall Curve (Micro-Averaged) ---")
y_test_binarized = label_binarize(y_test_encoded, classes=all_classes)

nb_y_score = nb_model.predict_proba(X_test)
svm_y_score = sgd_model.predict_proba(X_test)

precision_nb_micro, recall_nb_micro, _ = precision_recall_curve(y_test_binarized.ravel(), nb_y_score.ravel())
average_precision_nb_micro = average_precision_score(y_test_binarized, nb_y_score, average="micro")

precision_svm_micro, recall_svm_micro, _ = precision_recall_curve(y_test_binarized.ravel(), svm_y_score.ravel())
average_precision_svm_micro = average_precision_score(y_test_binarized, svm_y_score, average="micro")

plt.figure()
plt.plot(recall_nb_micro, precision_nb_micro, label=f'Naive Bayes (AP = {average_precision_nb_micro:.2f})')
plt.plot(recall_svm_micro, precision_svm_micro, label=f'SVM (AP = {average_precision_svm_micro:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Micro-Averaged Precision-Recall Curve')
plt.legend()
plt.show()

