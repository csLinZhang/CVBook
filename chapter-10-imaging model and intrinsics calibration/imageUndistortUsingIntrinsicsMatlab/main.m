% 张林，同济大学，2024年4月

% 该程序演示如何利用相机的内参数来进行图像的畸变去除
% 相机的内参数已经通过前期标定步骤获得，
% 并已存储在磁盘中为文件'cameraParams.mat'

% 导入相机参数数据
camParamsFile = load('cameraParams.mat');
camPrams = camParamsFile.cameraParams;
%读入由同款相机拍摄的原始图像，该图像带有明显畸变
oriImg = imread('img.png');
%对原始输入图像进行图像去畸变操作，这里需要利用已经获得的相机内参数
undistortedImage = undistortImage(oriImg, camPrams);

%显示结果
figure; 
imshowpair(oriImg, undistortedImage, 'montage');
title('Original Image (left) vs. Corrected Image (right)');

