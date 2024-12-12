# Mutual Learning for News Classification

## Project Description
This project explores a mutual learning approach for news classification, leveraging specialized machine learning models to handle distinct news categories. The goal was to demonstrate how models trained on smaller, category-specific datasets could share knowledge and achieve robust accuracy across all categories.

## Key Features
- **Mutual Learning Approach**: Combines specialized models (Naive Bayes and SVM) trained on distinct categories and enables knowledge sharing to improve overall performance.
- **Data Preprocessing**:
  - **Filtered Stemming**: Used Snowball Stemmer with a corpus filter to ensure accurate root word extraction without over-stemming.
  - **Lemmatization**: Context-aware word reduction using SpaCy to retain meaningful base forms.
- **Dataset**:
  - **Training Data**: 1,726 articles categorized into Business, Entertainment, Politics, Sport, and Tech.
  - **Testing Data**: 499 articles, derived from a separate subset.
- **Model Specialization**:
  - **Naive Bayes**: Optimized for Business, Sport, and Entertainment.
  - **SVM**: Focused on Politics and Tech.

## Phased Algorithm
1. **Phase 1: Initial Training**:
   - Trained models on their specialized categories using early stopping.
   - Naive Bayes: 60.72% accuracy.
   - SVM: 39.08% accuracy.
2. **Phase 2: Labeling Held-Out Data**:
   - Models labeled additional data for categories outside their specialization.
3. **Phase 3: Mutual Learning**:
   - Combined labeled data with original training data for cross-category learning.
   - Achieved final accuracies of 95.99% (Naive Bayes) and 94.99% (SVM).

## Final Results
- **Improvements**:
  - Naive Bayes: 58% accuracy improvement.
  - SVM: 144% accuracy improvement.
- Mutual learning enabled significant cross-category generalization, achieving high performance with smaller, specialized datasets.

## Technologies Used
- Python (SpaCy, Snowball Stemmer, scikit-learn, Pandas, Matplotlib)
- Machine Learning Models:
  - Naive Bayes
  - Stochastic Gradient Descent (SVM)

## Lessons Learned
- Mutual learning fosters collaboration between specialized models, addressing data limitations.
- Smaller, specialized datasets can deliver robust performance through knowledge sharing.

## Team Members
Brandon Lucero, Erica Macedo, Archit Malik, and Jonathan Rodriguez.
