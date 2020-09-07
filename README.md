# 19th Century English Fiction1 books Genre Identification on Gutenberg Corpus

## The task contains 2 main parts:
  1. Extracting ficiton books related features by using feature engineering techniques
  2. Apply supervised learning technique for classification of books to one genre class
  
### 1. Feature Extraction related to fiction book:
#### Features:
* Sentiment analysis - Beginning of book
* Sentiment analysis - End of book
* Sentence count
* Average sentence length
* Flesch reading score
* Word count
* Proper noun count

### 2. Supervised Learning techniques used:
* SVM
* Naive Bayes
* Random Forest

## Implementation:
### 1. Feature Extarction:
* Run feature_extractor.py to create features from the books
* Input: books folder containing html format books
* Output: features.csv file containing extracted features

### 2. Supervised Learning Algorithms:
* Simple_NB_SVM.py file containing SVM and NB method
* Leave_One_Out_SVM_NB.py file containing SVM and NB with leave one out method

