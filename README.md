## Abstract 
<p> 
  Now days, Heart disease is the most common disease. But, unfortunately the treatment of heart
disease is somewhat costly that is not affordable by common man. Hence, we can reduce this
problem in some amount just by predicting heart disease before it becomes dangerous
using Heart Disease Prediction System Using Machine Learning and Data mining. If we can
find out heart disease problem in early stages then it becomes very helpful for
treatment. Machine Learning and Data Mining techniques are used for the construction
of Heart Disease Prediction System. In healthcare biomedical field, there is large use of heath
care data in the form of text, images, etc but, that data is hardly visited and is not mined. So,
we can avoid this problem by introducing Heart Disease Prediction System. This system will
help us reduce the costs and to enhance the quality treatment of heart patients. This system can
able to identify complex problems and can able to take intelligent medical decisions. The
system can predict likelihood of patients of getting heart problems by their profiles such as
blood pressure, age, sex, cholesterol and blood sugar. Also, the performance will be compared
by calculation of confusion matrix. This can help to calculate accuracy, precision, and recall.
The overall system provides high performance and better accuracy. 
</p>

## Introduction
<p>
  The health care industries collect huge amounts of data that contain some hidden information,
which is useful for making effective decisions. For providing appropriate results and making
effective decisions on data, some advanced data mining techniques are used. In this study, a
Heart Disease Prediction System (HDPS) is developed using Naives Bayes and Decision Tree
algorithms for predicting the risk level of heart disease. The system uses 13 medical parameters
such as age, sex, blood pressure, cholesterol, and obesity for prediction. The HDPS predicts
the likelihood of patients getting heart disease. It enables significant knowledge. E.g.
Relationships between medical factors related to heart disease and patterns, to be established.
We have employed the multilayer perceptron neural network with back propagation as the
training algorithm. The obtained results have illustrated that the designed diagnostic system
can effectively predict the risk level of heart diseases.
</p>

### Aim
<p> 
  To predict heart disease according to input parameter values provided by user and dataset
stored in database.
</p>

### Objective
<p>
  The main objective of this project is to develop a heart disease prediction system. The system
can discover and extract hidden knowledge associated with diseases from a historical heart data
set Heart disease prediction system aims to exploit data mining techniques on medical data set
to assist in the prediction of the heart diseases.
</p>

### Project Scope
<p>
  The project has a wide scope, as it is not intended to a particular organization. This project is
going to develop generic software, which can be applied by any businesses organization.
Moreover it provides facility to its users. Also the software is going to provide a huge amount
of summary data.
</p>
  
### Technology Used:
- #### Machine-Learning Algorithms:
  - <a href="https://en.wikipedia.org/wiki/Gradient_boosting">**GRADIENT BOOSTING ALGORITHM**</a>
  - <a href="https://en.wikipedia.org/wiki/Logistic_regression">**LOGISTIC REGRESSION**</a>
- #### ML/DL:
  - ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
  - ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
  - ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
- Database:
  - ![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)
- #### Data-Set for training:
  - <a href="https://github.com/Kumar-laxmi/Heart-Disease-Prediction-System/blob/main/Machine_Learning/heart.csv">Click here for DATA-SET</a>
- #### IDE:
  - ![VS Code](https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
- #### OS used for testing:
  - ![MacOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=apple&logoColor=white)
  - ![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
  - ![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)

## Run Locally

Clone the project

```bash
  git clone https://github.com/Kumar-laxmi/Heart-Disease-Prediction-System
```

## Model Training(Machine Learning)

```javascript
def prdict_heart_disease(list_data):
    csv_file = Admin_Helath_CSV.objects.get(id=1)
    df = pd.read_csv(csv_file.csv_file)

    X = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
    nn_model = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1, random_state=0)
    nn_model.fit(X_train, y_train)
    pred = nn_model.predict([list_data])
    print("Neural Network Accuracy: {:.2f}%".format(nn_model.score(X_test, y_test) * 100))
    print("Prdicted Value is : ", format(pred))
    dataframe = str(df.head())
    return (nn_model.score(X_test, y_test) * 100),(pred)
```

