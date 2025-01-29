%% MVDIA - Exercise 3
% Eetu Knutars
% January 29th

%% Task 2
clc; close all; clearvars

window_size = 31;
k = 0.2;

for i = 1:10
img = im2gray(imread(num2str(i) +".bmp"));
img_gt = im2gray(imread(num2str(i)+"_gt.bmp"));
result_niblack = niblack(img, window_size, k);
result_sauvola = sauvola(img, window_size, k);
accuracy_niblack = round(100*mean(result_niblack==img_gt,'all'),1);
psnr_niblack = round(psnr(result_niblack, img_gt),1);
psnr_sauvola = round(psnr(result_sauvola, img_gt),1);
accuracy_sauvola = round(100*mean(result_sauvola==img_gt,'all'),1);
disp("Image "+num2str(i)+": Niblack " + num2str(accuracy_niblack)+ ...
    "% (PSNR "+num2str(psnr_niblack)+"), Sauvola " + ...
    num2str(accuracy_sauvola) + "% (PSNR " + num2str(psnr_sauvola) +")")
figure
subplot(2,2,1); hold on
imshow(img)
title("Original")
subplot(2,2,2); hold on
imshow(img_gt)
title("Ground truth")
subplot(2,2,3); hold on
imshow(result_sauvola)
title("Sauvola")
subplot(2,2,4); hold on
imshow(result_niblack)
title("Niblack")
end


%% FUNCTIONS

function [mean_img, std_img, pad] = get_mean_std(image, window)
    image = im2gray(image);
    
    % Add padding to the image
    pad = floor(window/2);
    padded = padarray(double(image), [pad pad], 'replicate');
    
    % Calculate mean and standard deviation in local windows
    mean_img = imfilter(padded, ones(window)/(window^2), 'replicate');
    mean_sqr = imfilter(padded.^2, ones(window)/(window^2), 'replicate');
    std_img = real(sqrt(mean_sqr - mean_img.^2));
end


function bw = sauvola(image, window, k)
    % Input parameters:
    % image: grayscale input image
    % window: window size (default=15)
    % k: control parameter (default=0.2)
    
    % Calculate mean and std
    [mean_img, std_img, pad] = get_mean_std(image, window);
    % Calculate threshold using Sauvola's formula
    R = max(std_img(:));
    threshold = mean_img .* (1 + k * (std_img/R - 1));
    % Remove padding
    threshold = threshold(pad+1:end-pad, pad+1:end-pad);
    % Apply threshold
    bw = image > threshold;
    bw = uint8(255*bw);
end

function bw = niblack(image, window, k)
    % Input parameters:
    % image: (grayscale) input image
    % window: window size
    % k: control parameter
    
    % Calculate mean and std
    [mean_img, std_img, pad] = get_mean_std(image, window);
    % Calculate threshold 
    threshold = mean_img + k * std_img;
    % Remove padding
    threshold = threshold(pad+1:end-pad, pad+1:end-pad);
    % Apply threshold
    bw = image > threshold;
    bw = uint8(255*bw);
end