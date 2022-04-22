# radiology_mo
## Problem #1: Significance Testing
I have run some models with svm with different parameters I need to do some significant testing analysis to see if there is some different or not such as calculate p values. I have the prediction results from each model and actual which is one for all of them. i have tried before to calculate the McNemar table then some other results two models they have the same dataset. I got almost the same accuracy I want to test that if there is significant different between them or not.
Because I'm doing Ten cross-validation So there are prediction and actual for each fold before I calculated the MaCnamar table as the average of 10 I'm not sure if that correct or there is another way better for calculation.

## Solution #1: 
p-values won't work for categorical data, we will have to use Chi-square


## Problem #2: Increasing testing precision from 0% to 90% 

I need two things but let’s start with first one :
- I need to change the way TCV svm run
for TCV svm: the input is the training only and each fold should run with grid search parameters to chose the best one So by the end of TCV I need to save the results and the output is what is the best parameters based on F1 value from the all 10 folds

- then I want to use this parameters for the svm only method which will take the training and testing data and calculate the four metrics and AUC values and also save the prediction and actual data as I did this is the first thing that I want to do to make sure that the model run in correct way. I only need you to test one model where the scaling is Standard and oversampling is Smote so both should be true

In summary: 
Doing binary classification with Accuracy and Auc measurement
the results is the measurement of that model with reasonable and better precision value

## Solution: #2
The model has very high training and validation precision, it also uses accuracy for scoring (which is not right for this kind of data)
- Change scoring from accuracy to f1score
- The sample size is too small given the features (even after oversampling) - use dimensionality reduction (Although it's not a best practice)
- Use TSNE
