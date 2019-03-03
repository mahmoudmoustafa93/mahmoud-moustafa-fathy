

clear ;
close all;
clc
InputOfLayerSize  = 400;  
NumOfLabels = 10;          
                          
%%0 art 1: Loading and Visualizing Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); 
m = size(X, 1);

rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;




%%  part 2a: Vectorize Logistic Regression
fprintf('\nTesting lrCostFunction() with regularization');
ThetaOfT = [-2; -1; 1; 2];
XoFt = [ones(5,1) reshape(1:15,5,3)/10];
YoFt = ([1;0;1;0;1] >= 0.5);
LamdaOfT = 3;
[J, grad] = lrCostFunction(ThetaOfT, XoFt, YoFt, LamdaOfT);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
%%  part 2b: One vs All Training 
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[ALLofTheta] = oneVsAll(X, y, NumOfLabels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% part 3: Predict for One Vs All 

pred = predictOneVsAll(ALLofTheta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

