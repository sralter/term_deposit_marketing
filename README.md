# Term Deposit Marketing - An Apziva Project (#2)
By Samuel Alter
Apziva: G3SuQYZYrFt9dwF3

## Summary
Designing a model that predicts which customers will most likely purchase a term deposit loan.

## Overview

### Table of Contents
* [Summary](#summary)
* [Overview](#overview)
* [The dataset](#the-dataset)
* [Goals](#goals)

### The dataset
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

### Figure 1
![Barplots of count of customers between successful and and failed campaigns](figures/2_countcategorical.jpg)
Although the raw numbers differ drastically between successful and failed campaigns, the patterns are similar for most of the features. Also notable is that there were no calls made to customers in the month of September.

### Figure 2
![Boxplots of numerical columns in dataset, separated by successful and failed campaigns](figures/2_boxplots.jpg)
