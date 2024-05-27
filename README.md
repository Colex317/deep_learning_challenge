# Deep Learning Challenge

## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.


From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

- **EIN** and **NAME** — Identification columns
- **APPLICATION_TYPE** — Alphabet Soup application type
- **AFFILIATION** — Affiliated sector of industry
- **CLASSIFICATION** — Government organization classification
- **USE_CASE** — Use case for funding
- **ORGANIZATION** — Organization type
- **STATUS** — Active status
- **INCOME_AMT** — Income classification
- **SPECIAL_CONSIDERATIONS** — Special considerations for application
- **ASK_AMT** — Funding amount requested
- **IS_SUCCESSFUL** — Was the money used effectively

## Files Included

deep_learning.ipynb


## Steps:
1. **Preprocess the Data**

      Pandas and Scikit-learn’s StandardScaler were used to preprocess the dataset in preparation for the next step.
   
2. **Compile, Train, and Evaluate the Model**

      TensorFlow was used to design a neural network model that can predict, based on the features in the dataset, whether an Alphabet Soup-funded organization will be successful.
   
4. **Optimize the Model**

      TensorFlow was used to optimize the model to achieve a target predictive accuracy higher than 75%.



# Report on the Neural Network Model

## Overview of the Analysis

The purpose of the analysis was to develop a neural network model based on the features in the dataset to predict whether the Alphabet Soup-funded organization will be successful.

## Results

### Data Preprocessing

**Target Variable(s):**

The target variable for the model is typically the outcome or the variable we are trying to predict. In this analysis, the target variable is:

- IS_SUCCESSFUL

<img width="879" alt="image" src="https://github.com/Colex317/deep_learning_challenge/assets/148498483/e81aa202-c98b-4de3-a97a-1242fea218d8">

----------------------------------------------------------------------------------------------------------------------------------------------

**Features**:

The features are the inputs to the model that help predict the target variable. In this analysis, the features are all the columns in the application_dummies DataFrame except for the IS_SUCCESSFUL column, such as:

- APPLICATION_TYPE	
            
- AFFILIATION	
            
- CLASSIFICATION	
            
- USE_CASE	
            
- INCOME_AMT
            
- ORGANIZATION	
            
- INCOME_AMT
            
- SPECIAL_CONSIDERATIONS	
            
- ASK_AMT



**Variable(s) to Remove:** 

The variables to remove are those that are neither targets nor features to avoid unnecessary complexity in the model. These variables do not contribute to the prediction task. 

In this analysis, the variables [EIN and NAME] were removed.


<img width="1029" alt="image" src="https://github.com/Colex317/deep_learning_challenge/assets/148498483/b70aa3e8-1a67-48d8-a4df-d0efe1fae691">

----------------------------------------------------------------------------------------------------------------------------------------------


### Compiling, Training, and Evaluating the Model

**INITIAL MODEL:**

The model was built with two hidden layers.

***First Hidden Layer:***

Number of neurons: 80

Activation function: ReLU (Rectified Linear Unit)

***Second Hidden Layer:***

Number of neurons: 30

Activation function: ReLU (Rectified Linear Unit)

***Output Layer:***

The output layer consists of a single neuron, as indicated by units = 1.

Activation function: Sigmoid


<img width="780" alt="image" src="https://github.com/Colex317/deep_learning_challenge/assets/148498483/d84ad98b-a981-4d34-84de-ae6d62dd4faa">

----------------------------------------------------------------------------------------------------------------------------------------------



**Were you able to achieve the target model performance?**


**Steps taken in an attempt to increase the model performance included:**

Using a systematic approach to tuning various aspects of the model, the neural network model was optimized to achieve a predictive accuracy of more than 75%.

- Adding more neurons to a hidden layer.
  
- Adding more hidden layers.
  
- Using different activation functions for the hidden layers.
  
- Reducing the number of epochs in the training regimen.


## Summary
Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
