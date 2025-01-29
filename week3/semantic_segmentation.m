%% MVDIA - Exercise 3
% Eetu Knutars
% January 29th

%% Task 3

%% Following the MATLAB tutorial
clc; close all; clearvars;

dataSetDir = fullfile(toolboxdir("vision"),"visiondata","triangleImages");
imageDir = fullfile(dataSetDir,"trainingImages");
labelDir = fullfile(dataSetDir,"trainingLabels");

imds = imageDatastore(imageDir);

classNames = ["triangle" "background"];
labelIDs   = [255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

I = read(imds);
C = read(pxds);

I = imresize(I,5,"nearest");
L = imresize(uint8(C{1}),5,"nearest");
imshowpair(I,L,"montage")

trainingData = pixelLabelImageDatastore(imds,pxds);

numFilters = 64;
filterSize = 3;
numClasses = 2;
layers = [
    imageInputLayer([32 32 1])
    convolution2dLayer(filterSize,numFilters,Padding=1)
    reluLayer()
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(filterSize,numFilters,Padding=1)
    reluLayer()
    transposedConv2dLayer(4,numFilters,Stride=2,Cropping=1);
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    ];

opts = trainingOptions("sgdm", ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=100, ...
    MiniBatchSize=64);

net = trainnet(trainingData,layers,@modelLoss,opts);
figure
testImage = imread("triangleTest.jpg");
imshow(testImage)
figure
C = semanticseg(testImage,net,Classes=classNames);
B = labeloverlay(testImage,C);
imshow(B)

%% Experimenting with U-Net
clc; close all; clearvars;

dataSetDir = fullfile(toolboxdir("vision"),"visiondata","triangleImages");
imageDir = fullfile(dataSetDir,"trainingImages");
labelDir = fullfile(dataSetDir,"trainingLabels");

imds = imageDatastore(imageDir);

classNames = ["triangle" "background"];
labelIDs   = [255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

imageSize = [32 32];
numClasses = 2;

% Defining the U-Net layers
lgraph = unet(imageSize, numClasses);

ds = combine(imds,pxds);

opts = trainingOptions("sgdm", ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=100, ...
    MiniBatchSize=64);

net = trainnet(ds,lgraph,@modelLoss,opts);

figure
testImage = imread("triangleTest.jpg");
imshow(testImage)
figure
C = semanticseg(testImage,net,Classes=classNames);
B = labeloverlay(testImage,C);
imshow(B)

function loss = modelLoss(Y,T)
    mask = ~isnan(T);
    T(isnan(T)) = 0;
    loss = crossentropy(Y,T,Mask=mask,NormalizationFactor="mask-included");
end