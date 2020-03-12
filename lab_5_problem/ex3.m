%% Machine Learning Online Class - Exercise 6 |: One-vs-all

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
sel2 = X(rand_indices(1:4000), :);
sel2y = y(rand_indices(1:4000), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2a: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

% Test case for lrCostFunction
fprintf('\nTesting lrCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);
h=sigmoid(X_t*theta_t)
p=sum((h-y_t).*X_t)/m
size(p)


fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%%%% Second example for testing if lrCostFunction is working

theta_t2 = [2; 1; -1; -2;-1];
X_t2 = [ones(5,1) reshape(1:20,5,4)/10];
y_t2 = ([1;0;1;0;1] >= 0.5);
lambda_t2 = 3;
[J2 grad2] = lrCostFunction(theta_t2, X_t2, y_t2, lambda_t2);
h2=sigmoid(X_t2*theta_t2)
p2=sum((h2-y_t2).*X_t2)/m
size(p2)


fprintf('\nCost: %f\n', J2);
fprintf('Expected cost: 3.898091\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad2);
fprintf('Expected gradients:\n');
fprintf(' -0.543821\n 0.433782\n -1.038129\n -1.910040\n -1.581950\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
%%%%
%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.25;
[all_theta] = oneVsAll(sel2, sel2y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X(rand_indices(4001:5000), :));

y2 = y(rand_indices(4001:5000), :);

pred(10:20)
y2(10:20)

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y2)) * 100);

