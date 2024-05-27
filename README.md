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

**Overview of the Analysis**
Explain the purpose of this analysis.

**Results**
Using bulleted lists and images to support your answers, address the following questions:

**Data Preprocessing**

- What variable(s) are the target(s) for your model?
- What variable(s) are the features for your model?
- What variable(s) should be removed from the input data because they are neither targets nor features?

**Compiling, Training, and Evaluating the Model**

- How many neurons, layers, and activation functions did you select for your neural network model, and why?
- Were you able to achieve the target model performance?
- What steps did you take in your attempts to increase model performance?

**Summary**
Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
