%此程序完整实现了Harris角点检测、block描述子提取、描述子匹配
%Lin Zhang, Tongji Univ., Apr. 2024
imgColor1 = imread('sse1.bmp');
imgColor2 = imread('sse2.bmp');    
% 
I1 = rgb2gray(imgColor1);
I2 = rgb2gray(imgColor2);
    
%计算M矩阵时，高斯窗口的标准差
sigma = 4.0;  
%判断一个点是否是角点的阈值，cornerness>thresh才被认为可能是角点
thresh = 200000;  
nonmaxrad = 5; %做非最大值抑制操作时候，局部窗口的大小，必须设置为整数，最终的局部窗口边长为2*nonmaxrad+1
    
%调用harrisCornerDetector函数，返回角点所在的行与列
%points1，存储图像1的角点坐标
%points2，存储图像2的角点坐标
[rows, cols] = harrisCornerDetector(I1, sigma,thresh, nonmaxrad);
points1 = zeros(size(rows,1),2);
points1(:,1) = cols;
points1(:,2) = rows;
[rows, cols] = harrisCornerDetector(I2, sigma,thresh, nonmaxrad);
points2 = zeros(size(rows,1),2);
points2(:,1) = cols;
points2(:,2) = rows;

%为角点集合中的每个角点提取block描述子，block的边长设为11
[descriptors1,valid_points1] = extractBlockDescriptors(I1,points1,11); 
[descriptors2,valid_points2] = extractBlockDescriptors(I2,points2,11);

maxRatioThreshold = 0.6; %进行匹配无歧义测试时候的阈值
match_thresh = 0.04; %一对描述子是否匹配的上的阈值，大于match_thresh认为不能匹配

descriptors1 = descriptors1';
descriptors2 = descriptors2';
%对描述子集合进行匹配，匹配上的描述子对的索引存在了indexPairs中
%indexPairs是一个matchedNum*2的矩阵，每一行代表了一对特征点匹配对信息，第一个元素是特征点（图I1中）在valid_points1中的位置，
% 第二个元素是特征点（图I2中）在valid_points2中的位置
indexPairs = matchDescriptors(descriptors1, descriptors2, match_thresh, maxRatioThreshold);
    
matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);

%以下代码是为了把特征点以及它们之间的匹配关系可视化出来
pp1 = matchedPoints1;
pp2 = matchedPoints2;

[rows1, cols1] = size(I1);
[rows2, cols2] = size(I2);
 
rows = max([rows1, rows2]);
cols = cols1 + cols2 + 3;
im = zeros(rows, cols);
 
im(1:rows1, 1:cols1) = I1;
im(1:rows2, cols1+4:cols) = I2;
 
pp2(:, 1) = pp2(:, 1) + cols1 + 3;
 
figure;
imshow(im,[]);
hold on
for index = 1:size(pp1, 1)
    x1 = pp1(index, 1);
    y1 = pp1(index, 2);
    plot(x1, y1, 'Marker', 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 5);
    x2 = pp2(index, 1);
    y2 = pp2(index, 2);
    plot(x2, y2, 'Marker', 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 5);
    line([x1, x2], [y1, y2], 'Color', 'y');
end
hold off