# Diagnosing Erythemato-Squamous Diseases

## Resources
**Python Version:** 3.8
**Packages:** Pandas, sklearn

## The Dataset
**Data Source:** [UCI Machine Learning Repository Dermatology Data Set](https://archive.ics.uci.edu/ml/datasets/Dermatology)

The dataset has 366 observations of 34 variables. Each observation is labeled from 1:5 with the label corresponding to a Erythemato-Squamous Disease:

1. psoriasis (111 observations)
2. seboreic dermatitis (60 111 observations)
3. lichen planus (71 111 observations)
4. pityriasis rosea (48 111 observations)
5. cronic dermatitis (48 111 observations)
6. pityriasis rubra pilaris (20 111 observations)

The 34 variables consist of:
* 32 ordinal attributes on a 0-3 scale
* 1 discrete numerical variable (age)
* 1 binary variable (family history)

## Data Preparation
This dataset is very clean and mostly complete. The only missing data is 8 instances of missing age variables.

For this I chose to delete the rows with missing values. Imputation is not a good option in this case because we don't have a good method for choosing values. It wouldn't make sense to impute the mean, median or mode age and regression techniques are generally not accurate when using ordinal data as the predictor variables. It is unclear if there is data is MCAR, MAR or MNAR but with only 8 observations missing this data point, deleteing the rows won't add a great deal of bias.

## Model Performance

I fit several models to compare performance. Each model was evaluated with K=5 K fold cross validation. Results are as follows:

* Decision Tree: 0.944
* Random Forest: 0.969
* Naive Bayes: 0.974
* Logistic Regression: 0.963

Naive Bayes and Random Forests are the two best performing models.
