%假设已经通过Matlab的Stereo Camera Calibrator
%APP对双目系统进行了标定，得到的双目系统参数文件已经存储为stereoParams.mat。该程序将导入双目参数文件，之后对一对双目图像计算视差图并基于此计算出3D点云
%2024年4月，张林，同济大学

%读入双目系统参数
stereoParams = load('stereoParams.mat');
stereoParams = stereoParams.stereoParams;

% 可视化双目系统外参
showExtrinsics(stereoParams);

%从双目图像对中选择一对，计算它的视差图以及点云
path = '.\stereo-imgs\';
leftImg = imread([path 'test-left\4.jpg']);
rightImg = imread([path 'test-right\4.jpg']);

%基于双目系统参数，完成左右目图像的校正
%reprojectionMatrix，对应于教材中式17-28中的矩阵Q，用于为已知视差的像素点计算其三维空间坐标
[frameLeftRect, frameRightRect, reprojectionMatrix] = rectifyStereoImages(leftImg, rightImg, stereoParams);

frameLeftGray  = im2gray(frameLeftRect);
frameRightGray = im2gray(frameRightRect);

%计算视差图
disparityMap = disparitySGM(frameLeftGray, frameRightGray);

figure;
imshow(disparityMap, [0, 128]);
title('Disparity Map');
colormap jet
colorbar

%基于视差图和投影矩阵reprojectionMatrix，计算出与左目校正化图像对应的3D点云
points3D = reconstructScene(disparityMap, reprojectionMatrix);

%之前系统中的物理单位使用的都是毫米，现在转换成米
points3D = points3D ./ 1000;
ptCloud = pointCloud(points3D, 'Color', frameLeftRect);
%将点云存储至本地磁盘
pcwrite(ptCloud,'result.ply','Encoding','ascii');

% 创建一个点云观察器
player3D = pcplayer([-3, 3], [-3, 3], [0, 3], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');
% 查看所生成的3D点云
view(player3D, ptCloud);