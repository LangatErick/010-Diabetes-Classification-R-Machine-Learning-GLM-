# R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-
Extensive Guide for Health Care Data Analysis using R(Machine Learning Algorithms, GLM)
This article is all about detailed Base Model analysis of the Diabetes Data which includes the following analysis:
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/f700cf91-548d-40c0-9101-38e2493f5b1d)

1. Data exploration (Data distribution inferences, Univariate Data analysis, Two-sample t-test)
2. Data Correlation Analysis
3. Feature Selection (using Logistic regression)

4. Outlier Detection (using principal component graph)

5. Basic Parameter Tuning(CV, complexity parameter)

6. Data modeling

## Basic GLM (With all Features and eliminating a few features based on AIC)

- Logistic Regression

- Decision Tree

- Naïve Bayes

```
{r warnings=F, message=F}
#import Libraries
library(bookdown)
library(tidyverse)
library(rmarkdown)
library(flexdashboard)
```
```
{r  warnings=F, message=F}
# {r warnings=F, message=F}
#IMPORT DATASET
diabetes <- read_csv("C:/Users/langa/OneDrive/Desktop/R PROGRAMMING PRACTICE/Extensive Guide for Health Care Data Analysis using R(Machine Learning Algorithms)#/diabetes.csv")
diabetes <- diabetes %>% 
     mutate(
       Outcome= ifelse(Outcome==1, )
     )

head(diabetes)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/bdbd002f-8117-4b8b-9a84-7ebe3fd46d91)

## Basic EDA
```
{r  warnings=F, message=F}
summary(diabetes)
```
### Uni-variate analysis
```
{r  warnings=F, message=F}
#library(patchwork)
par(mfrow=c(2,2))
p1 <- hist(diabetes$Pregnancies)
p2 <- hist(diabetes$Glucose)
p3 <- hist(diabetes$BMI)
p4 <- hist(diabetes$Age)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/6a724652-1d1b-4a89-a4ab-9232af5df5c0)
From these distribution graphs, Age and number of times pregnant are not in normal distributions as expected since the underlying population should not be normally distributed either.

Glucose level and BMI are following a normal distribution.
```
{r warning=FALSE, message=FALSE}
boxplot(diabetes$BloodPressure, ylab="BloodPressure")
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/9069e8b5-ab6a-482e-b4c2-0bf94da005ea)

## Impact of Glucose on Diabetes
```
{r  warning=FALSE, message=FALSE}
ggplot(diabetes, aes(x=Glucose))+
     geom_histogram(fill='deepskyblue', col='red') +
      facet_grid(Outcome~.)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/f038f3f3-749d-42c0-a6be-037e688c5c92)
Formulate a hypothesis to assess the mean difference of glucose levels between the positive and negative groups.

## Conditions

- Individuals are independent of each other

- Here distributions are skewed but the sample >30

- Both the groups are independent of each other and the sample size is lesser than 10% of the population,
```
{r warning=FALSE, message=FALSE}
library(report)
t <- t.test(Glucose~Outcome, data = diabetes) 
t
report(t)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/b2080aba-a333-446e-9d81-860003181c82)



