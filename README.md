# Term Deposit Marketing - An Apziva Project (#2)
By Samuel Alter
Apziva: G3SuQYZYrFt9dwF3

## Summary<a name='summary'></a>
Designing a model that predicts which customers will most likely purchase a term deposit loan.

## Overview<a name='overview'></a>
I produced two notebooks for this project, one for the [EDA](project2_eda.ipynb) and one for the [Modeling](project2_modeling.ipynb). This being the ReadMe, you can jump to those sections that are found below.
* [EDA](#eda)
* [Modeling](#modeling)

### Table of Contents
* [Summary](#summary)
* [Overview](#overview)
* [The dataset](#the-dataset)
* [Goals](#goals)
* [EDA](#eda)
  * [Figure 1]: Barplots of count of customers between successful and and failed campaigns(#figure-1)
  * [Figure 2]: Boxplots of numerical columns in dataset, separated by successful and failed campaigns(#figure-2)
  * [Figure 3]: Correlation of feature variables with target](#figure-3)
  * [What about Scatterplots?](#scat)
* [Modeling](#modeling)
  * [Notes on project setup](#notes-setup)
  * [Layer 1](#l1): Using only the demographic and banking data to simulate customers that haven't been contacted by the bank yet.
### The dataset<a name='the-dataset'></a>
I am working with a phone call dataset that also has demographic information about the recipients:
| Column | Data Type | Comments |
|---|---|---|
| `age` | Numeric | The age of the customer |
| `job` | Categorical | The job category of the customer |
| `marital` | Categorical | Whether the customer is married |
| `education` | Categorical | The customer's level of education |
| `default` | Binary | If the customer has credit in default or not |
| `balance` | Numeric | Average yearly balance in Euros |
| `housing` | Binary | If the customer has a housing loan or not |
| `loan` | Binary | If the customer has a personal loan |
| `contact` | Categorical | The type of contact communication |
| `day` | Numeric | Last contact day of the month |
| `month` | Categorical | Last contact month of the year |
| `duration` | Numeric | Duration of the last phone call with the customer |
| `campaign` | Numeric | The number of contacts performed during this campaign and for this client<br>including the last contact |
 
The final column, `y`, is the target of the dataset and shows whether the client subscribed to a term deposit.

### Goals <a name='goals'></a>
The startup is hoping that I can **achieve ≥81% accuracy** using a 5-fold cross validation strategy, taking the average performance score.

Bonus goals are:
* Determine which customers are most likely to buy the term deposit loan
  * Which segments of customers should the client prioritize?
* Determine what makes the customer buy the loan
  * Which feature should the startup focus on?

## EDA <a name='eda'></a>
There are 40000 rows and 14 columns in the datset, and it arrived to me clean, with no null values.

Of all 40000 customers, a little more than 7% received loans. This points to a very large class-imbalance in the datsaet.

With 13 columns, there was a lot of data to go through. We'll look at barplots of the amount of customers within each categorical column, separated into successful and failed campaigns [Figure 1](#figure-1), boxplots of the continuous columns [Figure 2](#figure-2), and a figure showing the correlatoin between each OneHotEncoded column against the target, `y` [Figure 3](#figure-3). Note: the columns were OneHotEncoded so that each column as shown in the figure refers to one category within a column. For example, there are four categories for highest level of education attained (primary, secondary, tertiary) and a category for customers with unknown education level. The OneHotEncoded version of this column would have a separate column for education_primary, with those that only possess that level of education getting encoded as a 1 and the rest getting a 0.

### Figure 1<a name='figure-1'></a>
![Barplots of count of customers between successful and and failed campaigns](figures/2_countcategorical.jpg)
Although the raw numbers differ drastically between successful and failed campaigns, the patterns are similar for most of the features. Also notable is that there were no calls made to customers in the month of September.

### Figure 2<a name='figure-2'></a>
![Boxplots of numerical columns in dataset, separated by successful and failed campaigns](figures/2_boxplots.jpg)
Duration does indeed seem different, though recall that this feature is describing how long the last phone call was with the customer. It may not tell us that much.

### Figure 3<a name='figure-3'></a>
![Correlation of feature variables with target](figures/2_corr_y.jpg)
Duration has the highest correlation with the target variable at over 0.4.

### Scatterplots?<a name='scat'></a>
**What about scatterplots?** you may ask. **My response**: Scatterplots did not seem to give us much insight. The data are very dispersed and a pattern does not readily emerge:
![Scatterplots are not helpful for this project](figures/2_pairplot.jpg)

## Modeling<a name='modeling'></a>
_**`AutoSklearn` to  `Optuna` to `scikit-learn`: the Modeling Workflow**_

I first used [`AutoSklearn`](#https://automl.github.io/auto-sklearn/master/#) to help me explore the ML algorithm landscape to identify the best-performing models for this particular dataset.
Next, In order to find the best hyperparameters for our modeling, used [`Optuna`](#https://optuna.readthedocs.io/en/stable/index.html). This is similar to other hyperparameter search frameworks like [`Hyperopt`](#http://hyperopt.github.io/hyperopt/), which are designed to quickly and efficiently find the best hyperparameters for your dataset.
Finally, we will use `sklearn` to build the final, optimized model.

### Notes on project setup<a name='notes-setup'></a>
We want to help the bank understand which customers are most likely to purchase the financial product. Knowing this would save the bank time and money. The dataset that we were given consists of demographic (and banking) data (like `age`,`job`,`marital`,and `balance`) as well as campaign-specific information (like `contact`,`day`,and `duration`).

| Demographic and Banking Data | Campaign-Specific Data | Target Feature |
|---|---|---|
| `age` | `contact` | `y` |
| `job` | `day` |  |
| `marital` | `month` |  |
| `education` | `duration` |  |
| `default` | `campaign` |  |
| `balance` |  |  |
| `housing` |  |  |
| `loan` |  |  |

We want to build a three-layered ML system that helps answer the project goals:
1. Understand which kinds of customers that they should call
 1. I will **not** give the model access to the campaign call data
1. After the initial calls, understand which customers the company should keep calling
 1. Give the model access to the campaign call data
1. Build a model using unsupervised learning to learn about clusters of customers in the dataset

**Layer 1**:  
Use `X_1` to model which customers to make calls to. We are training a model that does not know any call data, so this is *before* making calls.

**Layer 2**:  
Use the full `X` dataset (for clarity in its use in the layer flow, we'll be using `X_2` to model which customers the company should keep calling.

**Layer 3**:  
Use unsupervised learning to uncover how the customers are grouped.

### L1<a name='l1'></a>
I wrote a function that utilized AutoSklearn to spend 60 minutes perfoming a fitting and evaluation of the models. The function then returned a list of models that achieved a high accuracy.

However, with our balanced dataset, we needed more control, as we had to tune for recall. I decided that the best course of action was to run a [grid search](#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) of sorts. I created a list of scaling techniques, like `StandardScaler`(#), a list of sampling techniques, like `RandomOverSampler`(#) or `SMOTETomek`(#), and a list of classifiers to test, like `RandomForestClassifier` or [`LGBMClassifier`](#)
