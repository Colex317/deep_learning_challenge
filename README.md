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
- Number of neurons: 80
- Activation function: ReLU (Rectified Linear Unit)

***Second Hidden Layer:***
- Number of neurons: 30
- Activation function: ReLU (Rectified Linear Unit)

***Output Layer:***
- The output layer consists of a single neuron
- Activation function: Sigmoid

**Epochs**
- Number of epochs to the training regimen: 100


<img width="766" alt="image" src="https://github.com/Colex317/deep_learning_challenge/assets/148498483/f690200b-4ed0-4d32-a76c-f8ed8fa990e6">

----------------------------------------------------------------------------------------------------------------------------------------------

The result was **a predictive accuracy of 72.8%** 

<img width="716" alt="image" src="https://github.com/Colex317/deep_learning_challenge/assets/148498483/0afb3ac3-36a8-4ad9-9e89-09ad36fa4293">

----------------------------------------------------------------------------------------------------------------------------------------------


**OPTIMIZATION - FIRST MODEL:**

The model was built with two hidden layers, and the number of neurons for each was reduced, as was the number of epochs.

***First Hidden Layer:***
- Number of neurons: 40
- Activation function: ReLU (Rectified Linear Unit)

***Second Hidden Layer:***
- Number of neurons: 15
- Activation function: ReLU (Rectified Linear Unit)

***Output Layer:***
- The output layer consists of a single neuron
- Activation function: Sigmoid

**Epochs**
- Number of epochs to the training regimen: 10

<img width="710" alt="image" src="https://github.com/Colex317/deep_learning_challenge/assets/148498483/4398d9f5-0f68-41e7-b182-29563898d47e">

----------------------------------------------------------------------------------------------------------------------------------------------

The result was **a predictive accuracy of 73 %** 

<img width="716" alt="image" src="https://github.com/Colex317/deep_learning_challenge/assets/148498483/c34ff125-3b00-4e77-8d20-38e416e85a2e">


----------------------------------------------------------------------------------------------------------------------------------------------


**OPTIMIZATION - SECOND MODEL:**

The model was built with three hidden layers; the third hidden layer uses the sigmoid activation function. The number of neurons for the first two hidden layers and the number of epochs remained the same as in the first optimization attempt.

***First Hidden Layer:***
- Number of neurons: 40
- Activation function: ReLU (Rectified Linear Unit)

***Second Hidden Layer:***
- Number of neurons: 15
- Activation function: ReLU (Rectified Linear Unit)

***Third Hidden Layer:***
- Number of neurons: 10
- Activation function: Sigmoid

***Output Layer:***
- The output layer consists of a single neuron
- Activation function: Sigmoid

**Epochs**
- Number of epochs to the training regimen: 10

<img width="765" alt="image" src="https://github.com/Colex317/deep_learning_challenge/assets/148498483/d6ee61d0-df40-4e75-b596-6c51b0cbef70">

----------------------------------------------------------------------------------------------------------------------------------------------

The result was **a predictive accuracy of 73 %** 

<img width="737" alt="image" src="https://github.com/Colex317/deep_learning_challenge/assets/148498483/364b20c8-00c7-4c94-9c75-63319e263870">

----------------------------------------------------------------------------------------------------------------------------------------------

**OPTIMIZATION - THIRD MODEL:**

The model was built with three hidden layers. The number of neurons and epochs was increased (higher than the second attempt).

***First Hidden Layer:***
- Number of neurons: 50
- Activation function: ReLU (Rectified Linear Unit)

***Second Hidden Layer:***
- Number of neurons: 25
- Activation function: Tanh

***Third Hidden Layer:***
- Number of neurons: 15
- Activation function: Sigmoid

***Output Layer:***
- The output layer consists of a single neuron
- Activation function: Sigmoid

**Epochs**
- Number of epochs to the training regimen: 20

<img width="794" alt="image" src="https://github.com/Colex317/deep_learning_challenge/assets/148498483/690e1f76-39cc-4a23-a4dd-35134b53c4de">


----------------------------------------------------------------------------------------------------------------------------------------------

The result was **a predictive accuracy of 73 %** 

<img width="768" alt="image" src="https://github.com/Colex317/deep_learning_challenge/assets/148498483/4f40005c-66ee-4a3b-adba-0f5718a151ad">


----------------------------------------------------------------------------------------------------------------------------------------------

**Steps taken in an attempt to increase the model performance included:**

To optimize the neural network model to get to a predictive accuracy of more than 75%, the following were done:

- Reducing more neurons to a hidden layer.
  
- Adding more hidden layers.
  
- Using different activation functions for the hidden layers.
  
- Reducing the number of epochs in the training regimen.


**Were you able to achieve the target model performance?**

No, the optimization was done several times with few improvements; only a slight increase in accuracy was observed. It did not reach a predictive accuracy of 75%, the highest score reached was 73%.

## Summary

The deep learning model was optimized through various iterations to achieve a predictive accuracy of more than 75%. Several strategies were employed, including adjusting the number of neurons in hidden layers, adding more hidden layers, experimenting with different activation functions, and reducing the number of epochs in the training regimen. Despite these efforts, the model performance plateaued, with the highest achieved accuracy reaching only 73%.

Considering the challenges encountered in optimizing the deep learning model, it's advisable to explore alternative approaches. One potential solution could involve leveraging ensemble learning techniques, such as Random Forest and boosting (Khan, Chaundhari & Chandra, 2024). Ensemble learning methods combine multiple models to obtain a comprehensive and robust model and produce better results than other algorithms for datasets with class imbalance problems. 

### References

Khan, A.A., Chaundhari.O., & Chandra, R. (2024). A review of ensemble learning and data augmentation models for class imbalanced problems: Combination, implementation, and evaluation. ***Expert Systems with Applications***. https://doi.org/10.1016/j.eswa.2023.122778



