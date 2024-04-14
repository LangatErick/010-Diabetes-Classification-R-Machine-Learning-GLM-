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

p-value is \< critical values of 0.05, so we reject the null hypothesis for the alternate hypothesis. We can say that we are, 95 % confident, that the average glucose levels for individuals with diabetes is \> the people without diabetes.

```{r warning=FALSE, message=FALSE}
t1 <- t.test(Age~Outcome, data = diabetes)
t1
report(t1)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/9bfa30ab-0eb3-4d0c-a8d2-98042b6f1e18)

p_value\<0.001, suggest that diabetic people is mostly old.

```{r}
theme_set(theme_test())
diabetes %>% 
  ggplot(aes(x=cut(Age, breaks = 5))) +
        geom_boxplot(aes(y=DiabetesPedigreeFunction), col='deepskyblue')+
  labs(
    x = "Age Breaks =5",
    y = "DiabetesPedigreeFunction",
    colour = " ",
    shape = " "
   ) 

```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/a851ce21-3c6c-4f56-98f8-d09d4358b0d8)

## Insulin Vs Glucose based on Outcome as diabetes
```
diabetes %>% 
  ggplot(aes(x=Insulin,y=Glucose)) +
    geom_point() +
  geom_smooth()
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/3be6425c-e1b6-495a-b5f6-8cfd8bd33c90)
```{r warning=FALSE, message=FALSE}
# par(mfrow=c(1,2))
#boxplot
library(patchwork)
p0 <- diabetes %>% 
  ggplot(aes(x=DiabetesPedigreeFunction))+
  geom_boxplot() +
  facet_wrap(~Outcome) + coord_flip() +
  ggtitle('Boxplot')

p1 <- diabetes %>% 
  ggplot(aes(x=Glucose, col=Outcome))+
    geom_density()+
    facet_wrap(~Outcome) +
    ggtitle('Density plot of Glucose')

(p0 + p1)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/7bd7f676-e71b-4c54-b236-30ad0ba570ba)
From the Density Plot, the distribution is shifted towards the left for those without diabetes.

This indicates those **without diabetes generally have a lower blood glucose level**.

```{r warning=FALSE, message=FALSE}
#two sample t-test
t2 <- t.test(DiabetesPedigreeFunction~Outcome, data = diabetes) #%>% report()
t2
report(t2)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/5eb56f29-e411-4969-abcc-8c80e5c99f0b)

## Correlation between each variable
Scatter matrix of all columns
```{r warning=FALSE, message=FALSE}
library(GGally)
diabetes %>% select(-Outcome) %>% 
  ggcorr(
    name = "corr", label = TRUE
  ) +
  theme(legend.position = 'none') +
  labs(title = 'Correlation Plot of Variance') +
  theme(plot.title = element_text(
    face = 'bold',                                color = 'deepskyblue',
    hjust = 0.5, size = 11)) 
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/d3f92b5e-99c0-44a1-8e84-3fd8444c5252)

Pregnancy, Age, Insulin, and skin thickness have had higher correlations.

### Fitting a logistic regression to assess the importance of predictors
### Fitting a GLM (General Linear Model) with link function ‘probit’

- Target variable ‘diabetes’ estimated to be binomially distributed

- This is a generic implementation — without assumption on data
```{r warning=FALSE, message=FALSE}
logit <- glm(Outcome~.,, data = diabetes, family = binomial())
summary(logit)
report(logit)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/f9fbf778-9213-4671-bf1e-4a9c3750bd15)
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/cadb3db1-cd4d-4eb2-ad5c-64fd66b0dab3)
#### Filtering the most important predictors from the GLM model Extracting the N most important GLM coefficients
### Features Selection
- Highest logistic model coefficients
  ```{r}
model_coef <- exp(coef(logit))[2:ncol(diabetes)]
model_coef <- model_coef[c(order(model_coef,decreasing = TRUE)[1:(ncol(diabetes)-1)])]
predictors_names <- c(names(model_coef), names(diabetes)[length(diabetes)])
predictors_names
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/98d630f5-cae8-472b-831e-319e825f3a38)
```{r}
#filter df with most important predictors
diabetes_df <- diabetes[, c(predictors_names)]
head(diabetes_df)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/f4e6dc6f-eb06-44b5-963a-a7fa883b3c24)
### Outlier Detection

```{r warning=FALSE, message=FALSE}
library(DMwR2)
outlier_scores <- diabetes %>% select(-Outcome) %>% 
            lofactor(k=5)
plot(density(outlier_scores))
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/e79d23b0-fbfe-446b-b43f-30a33c0d3163)
```{r warning=FALSE, message=FALSE}
outliers <- order(outlier_scores, 
                  decreasing = TRUE)[1:5]
print(outliers)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/d9cd58e9-e090-4dde-aaf9-3103a3b796c8)

## **The five outliers obtained in the output are the row numbers in the diabetes1 data derived from the diabetes data set.**
```{r warning=FALSE, message=FALSE}
##labels outliers
n <- nrow(diabetes)
labels <- 1:n
labels[-outliers] <- "."
biplot(prcomp(diabetes[,-9], na.rm=TRUE), cex=.8, xlabs=labels)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/c599f783-d5cc-4784-90d0-6208c8081255)
```{r warning=FALSE, message=FALSE}
library(Rlof)
outlier.scores <- lof(diabetes[,-9], k=5)
outlier.scores<-lof(diabetes[,-9],k=c(5:10))
outlier.scores %>% head(4)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/a6508df8-019b-42d3-86ac-6b36b75ba8df)

## Data Modelling
```{r}
#data partitioning/Train and Test
library(rsample)
split <- initial_split(diabetes, prop = 8/10)
train <- training(split)
test <- testing(split)
```
**Basic GLM with all Variables**

    ```{r warning=FALSE, message=FALSE}
    log <- glm(Outcome~., family = binomial(), data = train)
    summary(log)
    ```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/0d4bfa04-d227-4d2c-8846-0e5322e39459)
The result shows that the variables Triceps_Skin, Serum_Insulin, and Age are not statistically significant. p_values is \> 0.01 so we can experiment by removing it.

## **Logistic Model**

input: explanatory variables xk and provides a prediction p with parameters $βk$.

The logit transformation constrains the value of p to the interval [0, 1].

\#

βk represents the log-odds of feature xk says how much the logarithm of the odds of a positive outcome (i.e. the logit transform) increases when predictor xk increases by .

Likelihood of the model as follows:

\#

$Y\^i$ = outcome of subject i.

Maximizing the likelihood = maximizing the log-likelihood(model)

\#

The above equation is non-linear for logistic regression and its minimization is generally done numerically by iteratively re-weighted least-squares

```{r}
smodel <- step(log)
```
### **The final model is chosen with AIC as the selection generated from a logistic regression model with the lowest AIC value of 584.68.**

## **Initial Parameter Tuning**

```{r}
library(rpart)
library(rpart.plot)
tree <- rpart(Outcome~., method = 'class', data = diabetes)
rpart.plot(tree)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/05b9df30-a565-4e97-8e0d-c185d646c7a5)

```{r}
plotcp(tree)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/e480bcc7-d494-4316-b5ad-e20cad37837d)
**Complexity parameter**

The above tree was tuned using a reference of the Relative error VS Complexity parameter. From the above figure the Cp value of 0.016, the decision tree was pruned. The final decision tree

```{r}
tree1 <- rpart(Outcome~., method = 'class', data = diabetes, cp=0.016)
rpart.plot(tree1)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/00ce65c4-3e99-4c67-b114-6f36f0b1deba)
If CP value is lower, tree will grow large. A cp = 1 will provide no tree which helps in pruning the tree. Higher complexity parameters can lead to an over pruned tree.

**2nd Model By removing 3 features-**

```{r}
log1 <- glm(Outcome~., family = binomial(), data=(train %>% select(-c(Age,SkinThickness, BloodPressure ,  Insulin))))

summary(log1)
report(log1)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/ecbb1f9d-0484-4d08-8cb6-66a40b2bae86)
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/461852ae-8602-42bd-8ff2-7517fe5d65cd)
```{r}
par(mfrow=c(2,2))
plot(log1)
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/e1173144-1606-4ff2-a014-65347433eaf0)

1\. **Residuals vs fitted values**; Here dotted line at y=0 indicates fit line. The points on fit line indication of zero residual. Points above are having positive residuals similarly points below have negative residuals. . The red line is indicates smoothed high order polynomial curve which provides the idea behind pattern of residual movements. Here the residuals have logarithmic pattern hence we got a good model.

2\. **Normal Q-Q Plot:** In general Normal Q-Q plot is used to check if our residuals follow Normal distribution or not. The residuals are said to be **normally distributed** if points follow the dotted line closely.

In our case residual points follow the dotted line closely except for observation at 229, 350 and 503 So this model residuals passed the test of Normality.

3\. **Scale — Location Plot:** It indicates spread of points across predicted values range.

Assumption:

\- Variance should be reasonably equal across the predictor range(Homoscedasticity)

So this horizontal red line is set to be ideal and it indicates that residuals have uniform variance across the Predictor range. As residuals spread wider from each other the red spread line goes up. In this case the data is Homoscedastic i.e having **uniform variance**.

4\. **Residuals vs Leverage Plot**:

**Influence**: The Influence of an observation can be defined in terms of how much the predicted scores would change if the observation is excluded. Cook’s Distance

**Leverage**: The leverage of an observation is defined on how much the observation’s value on the predictor variable differs from the mean of the predictor variable. **The more the leverage of an observation , the greater potential that point has in terms of influence**.

In our plot the dotted red lines are the cook’s distance and the areas of interest for us are the ones outside the dotted line on the top right corner or bottom right corner. If any point falls in that region, we say the observation has high leverage or having some potential for influencing our model is higher if we exclude that point.

**3rd Model: Predict Diabetes Risk on new patients using Decision Tree**

```{r}
library(party)
ct <- ctree(Outcome~.,data=train)
# plot(ct)
predict_clas <- predict(ct, test,
                         type=c('response'))
table(predict_clas, test$Outcome)
```
```{r}
library(caret)
con_mat <- confusionMatrix(test$Outcome, predict_clas, positive = NULL,
              dnn = c('Prediction', 'References'))
con_mat
```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/aead0533-2b57-49c1-b330-70723990eb7d)
### 4th Model Naïve Bayes:

```{r warning=FALSE, message=FALSE}
library(e1071)
nb <- naiveBayes(Outcome~., data = train)
pred <- predict(nb, test)
confusionMatrix(test$Outcome, pred)

```
![image](https://github.com/LangatErick/R-Project-Extensive-Guide-for-Health-Care-Data-Analysis-using-R-Machine-Learning-Algorithms-GLM-/assets/124883947/8eb9dd15-b8b2-499a-aa07-c3ce8f825708)

# **Though it’s a basic model still it performed well with 77% accuracy on an average**


