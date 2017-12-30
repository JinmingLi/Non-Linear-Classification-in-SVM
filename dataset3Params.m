function [C, sigma] = dataset3Params(X, y, Xval, yval)

C = 1;
sigma = 0.3;

TD = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; 
pre_err = zeros(length(TD)); 
for i = 1:length(TD) 
	for j = 1:length(TD) 
        C = TD(i); 
        sigma = TD(j); 
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model, Xval); 
        pre_err(i, j) = mean(double(predictions ~= yval)); 
	end 
end 
mm = min(min(pre_err)); 
[ind_C, ind_sigma] = find(pre_err == mm); 
C = TD(ind_C); 
sigma = TD(ind_sigma);

end
