%% MVDIA
% Exercise 1
% Eetu Knutars
% January 13th 2025
%% Task 1
clc; close all; clearvars;

J_ref = imread("ex21a.tif");
J = imread("ex21b.tif");
J = J(:,:,1);

% Collecting four control points manually
fixed_points = [600,608;619,75;115,110;74,631];
moving_points = [618,663;601,125;97,59;94,578];

subplot(1,2,1)
imshow(J_ref)
hold on
plot(fixed_points(:,1),fixed_points(:,2),'ro',LineWidth=2)
title("Reference image")
subplot(1,2,2)
imshow(J)
hold on
plot(moving_points(:,1),moving_points(:,2),'ro',LineWidth=2)
title("Input")

T = fitgeotrans(moving_points, fixed_points, "affine");

J_reg = imwarp(J,T,'OutputView',imref2d(size(J)));
figure
imshowpair(J_ref,J_reg)


%% Task 2
clc; close all; clearvars;

% Data onboarding
load data; f1 = data(:,1); f2 = data(:,2); y = data(:,3);

% Visualizing data
figure; grid on; hold on;
plot(f1(y==0),f2(y==0),'b.', MarkerSize=10)
plot(f1(y==1),f2(y==1),'r.', MarkerSize=10)
xlabel("Lightness", "Interpreter","latex")
ylabel("Width", 'Interpreter','latex')
legend("Salmon", "Sea Bass", 'Interpreter', 'latex')
%title("Fish type classification", 'Interpreter','latex')

% Method 1: Bayesian classification
salmon = data(y==0,1:2);
mu_1 = mean(salmon);
Sigma_1 = diag(std(salmon)).^2;

sea_bass = data(y==1,1:2);
mu_2 = mean(sea_bass);
Sigma_2 = diag(std(sea_bass)).^2;

prior = [length(salmon(:,1)), length(sea_bass(:,1))]/length(y);
% Priors are the same, data set contains equal amount of sea basses
% and salmons

% 2-dimensional gaussian likelyhood function
likelyhood = @(x,mu,Sigma) (2*pi)^(-1) * det(Sigma)^(-1/2) *...
    exp((-1/2) * (x-mu)' * (Sigma \ (x-mu)));

% Prior is equal for both classes, so we can just use likelyhood
% for calculating the posterior value

params = {};
params.L = likelyhood;
params.mu1 = mu_1;
params.mu2 = mu_2;
params.Sigma1 = Sigma_1;
params.Sigma2 = Sigma_2;

y_est = bayes_classifier(data(:,1:2), params);
xspan = 0:.1:10;
yspan = 12:.1:22;
[X,Y] = meshgrid(xspan, yspan);
fun = @(x,y) bayes_classifier([x,y], params);
Z = arrayfun(fun, X, Y);
contour(xspan, yspan, Z, 1, LineColor="k")
legend("Salmon", "Sea Bass", "Decision boundary", 'Interpreter', 'latex')
accuracy_bayes = mean(y==y_est)

% Method 2: k-nearest neighbours
k = 3; % example k value
N = length(y);
knn_data = zeros(N,3);
idx = randperm(N);
knn_data(idx,:) = data;
XTrain = knn_data(1:round(3*N/4),1:2);
yTrain = knn_data(1:round(3*N/4),3);
XTest = knn_data(round(3*N/4)+1:end,1:2);
yTest = knn_data(round(3*N/4)+1:end,3);

y_est = zeros(size(yTest));
for i = 1:length(yTest)
    [val,ind] = sort(sum((XTrain-XTest(i,:)).^2,2));
    y_est(i)=mode(yTrain(ind(1:5)));
end

figure; grid on; hold on;
plot(f1(y==0),f2(y==0),'b.', MarkerSize=10)
plot(f1(y==1),f2(y==1),'r.', MarkerSize=10)
xlabel("Lightness", "Interpreter","latex")
ylabel("Width", 'Interpreter','latex')
legend("Salmon", "Sea Bass", 'Interpreter', 'latex')

%y_est = knn_classifier(data(:,1:2), params);
xspan = 0:.1:10;
yspan = 12:.1:22;
[X,Y] = meshgrid(xspan, yspan);
fun = @(x,y) knn_classifier([x,y], XTrain, yTrain, k);
Z = arrayfun(fun, X, Y);
contour(xspan, yspan, Z, 1, LineColor="k")
legend("Salmon", "Sea Bass", "Decision boundary", 'Interpreter', 'latex')

accuracy_knn = mean(yTest==y_est)



% Method 3: MLP
% implemented in Python, see mlp.ipynb

% Method 4: SVM
% implemented in Python, see svm.ipynb


%% Task 3
clc; close all; clearvars;
% Lets create some artificial dataset with three clusters
X1 = 1.1*randn([50,2]);
X2 = 5+.8*randn([50,2]);
X3 = [1,-10]+1.5*randn([50,2]);
X = [X1;X2;X3];

figure; 
subplot(1,2,1)
hold on; grid on;
plot(X(:,1),X(:,2),'b.', MarkerSize=15)

C = 3;
N = max(size(X));
centroids_ind = randi(N,[3,1]);
centroids = X(centroids_ind,:);
plot(centroids(:,1),centroids(:,2), 'ro', MarkerSize=15)
legend("Data", "Initial centroid", Location="nw")

while true
    old_centroids = centroids;
    new_labels = zeros(N,1);
    for i = 1:N % finding the closest centroid for each data point
        [~,idx] = min(sum((X(i,:)-centroids).^2,2));
        new_labels(i) = idx;
    end
    for i = 1:C % updating the centroids
        centroids(i,:) = mean(X(new_labels==i,:));
    end

    % End the algorithm when no more change in clusters occus
    if norm(centroids-old_centroids) < 0.01
        break
    end
end

subplot(1,2,2)
hold on; grid on;
plot(X(new_labels==1,1), X(new_labels==1,2),'b.', MarkerSize=15)
plot(X(new_labels==2,1), X(new_labels==2,2),'r.', MarkerSize=15)
plot(X(new_labels==3,1), X(new_labels==3,2),'g.', MarkerSize=15)
legend("Cluster 1", "Cluster 2", "Cluster 3", Location="nw")



function [est] = bayes_classifier(x, params)
    N = length(x(:,1));
    likelyhood = params.L;
    mu_1 = params.mu1;
    mu_2 = params.mu2;
    Sigma_1 = params.Sigma1;
    Sigma_2 = params.Sigma2;
    est = zeros(N,1);
    for i = 1:N
        L_1 = likelyhood(x(i,:)', mu_1', Sigma_1);
        L_2 = likelyhood(x(i,:)', mu_2', Sigma_2);
        if L_1 > L_2
            est(i) = 0;
        else
            est(i) = 1;
        end
    end
end

function [est] = knn_classifier(x, traindata, trainclass, k)
    s = size(x);
    est = zeros(s(1),1);
    for i = 1:length(est)
        [~,ind] = sort(sum((traindata-x(i,:)).^2,2));
        est(i)=mode(trainclass(ind(1:k)));
    end
end


