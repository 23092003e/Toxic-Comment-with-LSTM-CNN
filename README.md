# Toxic-Comment-with-LSTM-CNN
Detecting harassment in social media discussions is crucial, as well as assessing the degree of negativity present in comments. Automating the classification of such comments can significantly reduce the time and effort required for manual moderation, enabling organizations to manage online platforms more efficiently. The primary goal is to analyze the toxicity and negativity in online comments, facilitating the identification of individuals who engage in abusive behavior. This, in turn, supports the enforcement of rules and penalties for violations, contributing to a reduction in toxicity in online discussions. By leveraging LSTM and CNN models, we aim to develop a multi-label classification system that categorizes comments into six toxicity levels: 
- **toxic**
- **severe toxic**
- **obscene**
- **threat**
- **insult**
- **identity hate**

# Overview
Online platforms face challenges in moderating user-generated content. Manual moderation is time-consuming and often inconsistent. This project offers an automated solution to:
- **Reduce moderation workload:** By automatically flagging toxic comments.
- **Improve user experience:** By helping platforms quickly identify and act upon harmful content.
- **Support community guidelines enforcement:** By accurately categorizing different types of toxic behavior.

# Dataset

The dataset consists of comments collected from social media platforms, annotated for six different toxicity categories. You may need to download or update the dataset as per your project requirements. Ensure the dataset is placed in the `dataset/` folder.

# Model Architecture

This project employs a hybrid deep learning model that integrates:
- **LSTM (Long Short-Term Memory):** To capture sequential dependencies in text.
- **CNN (Convolutional Neural Networks):** To extract local features and patterns from comments.

The combination of LSTM and CNN provides a robust framework for handling the complexity of language and detecting nuanced forms of toxicity.

# Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/23092003e/Toxic-Comment-with-LSTM-CNN.git
   cd Toxic-Comment-with-LSTM-CNN

