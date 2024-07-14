## Overview
Breast cancer (BC) is one of the most common cancers among women worldwide, representing the majority of new cancer cases and cancer-related deaths.
The early diagnosis of BC can improve the prognosis and chance of survival significantly, as it can promote timely clinical treatment to patients. Further accurate classification of benign tumors can prevent patients undergoing unnecessary treatments.
There are two types of classification of tumors : benign and malignant. Benign tumors are those that stay in their primary location without invading other sites of the body.Benign tumors are not usually problematic. Malignant tumors have cells that grow uncontrollably and spread locally and/or to distant sites. Malignant tumors are cancerous (ie, they invade other sites).

## Description
The goal of this project is to develop a computer program that can help doctors detect breast cancer early. By analyzing medical data about patients' breast cells, the program can predict whether a tumor is likely to be benign (non-cancerous) or malignant (cancerous). Early and accurate detection of breast cancer can lead to better treatment outcomes and save lives.

## What is done in the Project?
I gathered a dataset and explored the Data to understand the types of information it contains and identified any missing or inconsistent data that needed to be addressed. Then
cleaned the data by filling in missing values and normalizing it to ensure that all the characteristics were on a similar scale, making it easier for the algorithms to process.

I used several advanced machine learning algorithms, including Random Forest, Support Vector Machine (SVM), and XGBoost, to train models on the dataset and fine-tuned the models by adjusting their settings (hyperparameters) using a method called GridSearchCV, which helps find the best combination of parameters for the highest accuracy.

Stored the trained models using a tool called Pickle so they could be easily reused for making predictions on new data in the future.

## TechStack and Frameworks Requirements
- Hardware: personal computer, RAM is 1.47 GB and Disk is 22.56 GB
- Tools: Jupyter Notebook, Pickle
- Programming Language: Python
- Libraries:  NumPy, Pandas, Seaborn, Matplotlib, Scikit-Learn, XGBoost, XGBClassifier, Random Forest classifier, SVC

## DataSet Description
The dataset used for project is the Breast Cancer Wisconsin (Diagnostic) dataset, which is widely utilized for breast cancer research and predictive modeling and the Source is publicly available from the UCI Machine Learning Repository. It also uploaded in my above repository as data.csv
- Number of Instances: 569
- Number of Attributes: 32 (including the target variable)
- Attribute Information:
1.) ID number
2.) Diagnosis (target variable): Indicates whether the tumor is benign (B) or malignant (M).
4.) 30 Numerical Features: These features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass and describe characteristics of the cell nuclei present in the image. These include:
Radius (mean of distances from center to points on the perimeter)
Texture (standard deviation of gray-scale values)
Perimeter
Area
Smoothness (local variation in radius lengths)
Compactness (perimeter^2 / area - 1.0)
Concavity (severity of concave portions of the contour)
Concave points (number of concave portions of the contour)
Symmetry
Fractal dimension ("coastline approximation" - 1)

- Data Characteristics:
- Missing Values: None
- Class Distribution: 357 benign samples and 212 malignant samples

![image](https://github.com/user-attachments/assets/13358ea9-e406-47ae-ac8a-7dbcc51ec1ec)

![image](https://github.com/user-attachments/assets/7d5c0b0b-2b3e-4b45-8c30-c0bf8f7e1f5e)

![image](https://github.com/user-attachments/assets/a2dbfc91-683e-4575-a865-337efd58b8b2)

## Results
- Model Performance: Random Forest Classifier achieved accuracy of 97.37% and both Support Vector Machine(SVM) & XGBoost Classifier Acheived 99.12%.
- Classification Metrics (Precision and Recall):
  For Benign Tumors (Class 0) it is 0.99 and 0.97 resp.
  For Malignant Tumors (Class 1) it is 0.95 and 0.97 resp.
