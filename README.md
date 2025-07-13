# Mental-Health-Classifier
A machine learning-based project that classifies emotional text inputs into "Mental Health Concern" or "Normal" using NLP techniques and Logistic Regression. Developed using Python in Google Colab with TF-IDF vectorization, the system aims to support early detection of mental health issues from user-generated content.
# AI-Powered Mental Health Classifier from Text

This project builds an AI-based system that classifies emotional text inputs into two categories: **"Mental Health Concern"** or **"Normal."** Using Natural Language Processing (NLP) and supervised machine learning, the system helps identify early signs of emotional distress from user-generated text.

---

## Problem Statement

Mental health conditions often go unrecognized in their early stages, especially when expressed through everyday language. There is a growing need for intelligent systems that can analyze textual input and detect early warning signs of emotional or psychological issues. This project addresses this challenge using AI-based text classification.

---

## Proposed System / Solution

**1. Data Collection:**
- Use a labeled dataset containing emotional text (e.g., joy, sadness, anger, fear).
- Map emotional labels into two classes: **Mental Health Concern** and **Normal**.

**2. Data Preprocessing:**
- Clean the text (remove stopwords, punctuation, irrelevant characters).
- Convert text into numerical format using **TF-IDF vectorization**.
- Split the dataset into training and testing sets.

**3. Machine Learning Model:**
- Train a supervised ML model (e.g., **Logistic Regression** or **SVM**).
- Classify text into either “Normal” or “Mental Health Concern.”
- Evaluate the model using accuracy, precision, recall, and confusion matrix.

**4. Model Testing (in Google Colab):**
- The model is trained, tested, and evaluated entirely within **Google Colab**.
- Users can manually input text to receive predictions in real time during notebook execution.

**5. Evaluation:**
- Metrics such as **accuracy score**, **F1 score**, and **confusion matrix** are used to measure performance.
- Results are visualized directly in Colab using matplotlib/seaborn.

---

## System Development Approach (Technology Used)

### System Requirements:
- **Hardware:** Minimum 4GB RAM and stable internet connection
- **Software Stack:**
  - Python 3.x
  - Google Colab (for model development and execution)

### Required Libraries:
- **Pandas** – Data handling  
- **NumPy** – Numerical operations  
- **Scikit-learn** – ML algorithms  
- **NLTK** – Text preprocessing  
- **TF-IDF Vectorizer** – Feature extraction  
- **Matplotlib / Seaborn** – Visualizations  

### Methodology Overview:
- Load and label the dataset  
- Preprocess and clean the text  
- Extract features using TF-IDF  
- Train and evaluate a **Logistic Regression** model  
- Test model predictions in Colab notebook environment  

---

## Algorithm & Training

### Algorithm Selection:
The algorithm chosen is **Logistic Regression**, a supervised classification model. It is efficient, interpretable, and well-suited for binary text classification tasks like this one.

### Data Input:
- Emotional text (user-generated)
- Feature vectors generated using TF-IDF

### Training Process:
- Dataset is split into training and testing sets (e.g., 80/20 split)
- Model is trained using `sklearn.linear_model.LogisticRegression`
- Evaluation includes accuracy, F1 score, and confusion matrix

### Prediction Process:
- Users enter text directly into the Colab interface
- Model transforms input using TF-IDF and returns the prediction (Normal / Mental Health Concern)
- Predictions and metrics are displayed within the notebook output

---

## Result

The model was trained and tested successfully in Google Colab. It classifies user-input text with strong accuracy and interpretable results.

**Sample Metrics:**

- **Accuracy**: 88%  
- **Precision**: 87%  
- **Recall**: 83%  
- **F1 Score**: 85%
- <img width="567" height="435" alt="download55" src="https://github.com/user-attachments/assets/5e5546ea-8cc4-40a5-bf45-029b4b689b31" />


---

## Conclusion

This AI-powered system offers a practical way to identify mental health concerns through text classification. It demonstrates the value of combining NLP and machine learning to support mental health awareness in a digital-first world. Though not a substitute for professional diagnosis, it can act as an early-warning tool to encourage individuals to seek help.

---

## Future Scope

- Expand dataset with real-world anonymized data  
- Support multiple languages and regional text inputs  
- Enhance performance using deep learning models like BERT or LSTM  
- Create a web/mobile interface in future for broader accessibility  
- Continuously update the model based on user feedback and new data


---

## References

1. Scikit-learn documentation – https://scikit-learn.org  
2. NLTK Toolkit – https://www.nltk.org  
3. TF-IDF Vectorizer – https://scikit-learn.org/stable/modules/feature_extraction.html  
4. Streamlit – https://streamlit.io  
5. Microsoft Azure ML documentation – https://learn.microsoft.com/en-us/azure/machine-learning/  
6. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.  
7. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.  
8. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523.  
9. GitHub Repository: https://github.com/GopalGhosh55/mental-health-classifier

---

## Author

**Gopal Ghosh**  
B.Tech CSE | Microsoft AICTE Internship by Edunet Project | July 2025
