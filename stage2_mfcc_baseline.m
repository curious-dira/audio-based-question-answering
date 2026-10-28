
%% Case 07 - Stage 2: MFCC + SVM baseline
clear; clc; close all;
rng(42);

% Find dataset paths
if isfile(fullfile("meta","esc50.csv"))
    datasetRoot = pwd;
elseif isfile(fullfile("..","meta","esc50.csv"))
    datasetRoot = fileparts(pwd);
else
    error("Run this script from ESC-50-master or its meta folder.");
end

audioFolder = fullfile(datasetRoot,"audio");
T = readtable(fullfile(datasetRoot,"meta","esc50.csv"));
T.category = string(T.category);
T.filename = string(T.filename);

% Five classes for the first prototype
classes = ["dog","rain","car_horn","chainsaw","clock_tick"];
T = T(ismember(T.category,classes),:);

% Use the predefined folds
split = strings(height(T),1);
split(ismember(T.fold,[1 2 3])) = "train";
split(T.fold == 4) = "validation";
split(T.fold == 5) = "test";
T.split = split;

% Use fold 5 only for final testing
trainRows = T.split == "train";
testRows  = T.split == "test";

% MFCC configuration
targetFs = 44100;
numCoeffs = 13;
featureLength = 28;  % 2 x 14 coefficients returned by mfcc
X = zeros(height(T),featureLength);
Y = T.category;

fprintf("Extracting MFCC features from %d clips...\n",height(T));

for i = 1:height(T)
    filePath = fullfile(audioFolder,T.filename(i));
    [x,fs] = audioread(filePath);

    if size(x,2) > 1
        x = mean(x,2);
    end
    if fs ~= targetFs
        x = resample(x,targetFs,fs);
        fs = targetFs;
    end

    % Mean and standard deviation across time for each MFCC
    C = mfcc(x,fs);
    feature = [mean(C,1), std(C,0,1)];
    X(i,:) = feature;
end

% Train on folds 1-3
Xtrain = X(trainRows,:);
Ytrain = categorical(Y(trainRows));
Xtest = X(testRows,:);
Ytest = categorical(Y(testRows));

% Standardise based on training data only
mu = mean(Xtrain,1);
sigma = std(Xtrain,0,1);
sigma(sigma == 0) = 1;
Xtrain = (Xtrain-mu)./sigma;
Xtest = (Xtest-mu)./sigma;

% Train SVM with error-correcting output codes for multiclass classification
model = fitcecoc(Xtrain,Ytrain, ...
    "Learners",templateSVM("KernelFunction","linear","Standardize",false));

pred = predict(model,Xtest);
accuracy = mean(pred == Ytest);
C = confusionmat(Ytest,pred);

fprintf("\nTest accuracy: %.2f%%\n",100*accuracy);
disp("Confusion matrix (rows=true class, columns=predicted class):");
disp(C);

% Per-class recall and macro-F1
classNames = categories(Ytest);
precision = zeros(numel(classNames),1);
recall = zeros(numel(classNames),1);
f1 = zeros(numel(classNames),1);

for k = 1:numel(classNames)
    tp = C(k,k);
    fp = sum(C(:,k))-tp;
    fn = sum(C(k,:))-tp;
    precision(k) = tp/max(tp+fp,1);
    recall(k) = tp/max(tp+fn,1);
    f1(k) = 2*precision(k)*recall(k)/max(precision(k)+recall(k),eps);
end

fprintf("Macro-F1: %.3f\n",mean(f1));

% Plot confusion matrix
figure("Color","w");
confusionchart(Ytest,pred);
title("MFCC + SVM Baseline: ESC-50 Five-Class Test Set");

% Save results
outFolder = fullfile(datasetRoot,"outputs");
if ~isfolder(outFolder)
    mkdir(outFolder);
end
saveas(gcf,fullfile(outFolder,"baseline_confusion_matrix.png"));

results = table(accuracy,mean(f1), ...
    'VariableNames',{'TestAccuracy','MacroF1'});
writetable(results,fullfile(outFolder,"baseline_metrics.csv"));

save(fullfile(outFolder,"baseline_model.mat"), ...
    "model","mu","sigma","classes","numCoeffs","targetFs");

fprintf("\nSaved baseline metrics, confusion matrix, and model under outputs/.\n");
