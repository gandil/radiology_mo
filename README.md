# radiology_mo
## Problem #1: Significance Testing
I have run some models with svm with different parameters I need to do some significant testing analysis to see if there is some different or not such as calculate p values. I have the prediction results from each model and actual which is one for all of them. i have tried before to calculate the McNemar table then some other results two models they have the same dataset. I got almost the same accuracy I want to test that if there is significant different between them or not.
Because I'm doing Ten cross-validation So there are prediction and actual for each fold before I calculated the MaCnamar table as the average of 10 I'm not sure if that correct or there is another way better for calculation.

## Solution #1: 
p-values won't work for categorical data, we will have to use Chi-square

