%% Initialization
clear ; close all; clc

%% Visualizing Data 

fprintf('Loading and Visualizing Data ...\n')

% Load from data1: 

load('data1.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Training Linear SVM 

% Load from data1: 
load('data1.mat');

fprintf('\nTraining Linear SVM ...\n')

C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Implementing Gaussian Kernel

fprintf('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :' ...
         '\n\t%f\n\n'], sim);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Visualizing Dataset 2 

fprintf('Loading and Visualizing Data ...\n')

% Load from data2: 
% You will have X, y in your environment
load('data2.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Training SVM with RBF Kernel (Dataset 2)
%  After you have implemented the kernel, we can now use it to train the 
%  SVM classifier.
% 
fprintf('\nTraining SVM with RBF Kernel ...\n');

% Load from data2: 
load('data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Visualizing Dataset 3 

fprintf('Loading and Visualizing Data ...\n')

% Load from data3: 
load('data3.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Training SVM with RBF Kernel (Dataset 3)

% Load from data3: 
% You will have X, y in your environment
load('data3.mat');

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;
