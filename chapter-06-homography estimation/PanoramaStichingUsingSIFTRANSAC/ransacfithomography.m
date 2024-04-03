% 该函数从输入点对关系pp1Homo和pp2Homo中估计出点集之间的射影变换
% pp1Homo和pp2Homo都是3xN的数组，对应的一列是一对对应点的齐次坐标
% pp1Homo，pp2Homo必须要有相同的列，即相同的点数，且列数要大于4
% t，RANSAC框架中用于内点判定的阈值，点带入模型之后，误差大于t，则认为是外点
% H，从对应点对关系集合中估计出的射影矩阵，尽可能使得cx2 = H*x1
% inliers，索引，指明pp1Homo和pp2Homo的哪些列是H的一致集
function [H, inliers] = ransacfithomography(pp1Homo, pp2Homo, t)    
  %最少需要4个点对能唯一确定两个平面间的射影变换
  %与算法6-1中的n意义相同
  n = 4; 
  %从点对关系数据进行射影变换估计的函数句柄
  fittingfn = @homography2d; 
  %当有了当前拟合模型之后，如何计算某数据点误差的函数句柄
  %对于我们的问题来说，homogdist2d计算在当前估计出的射影变换下，
  %对应点对的双向重投影误差
  distfn = @homogdist2d; 
  %判断用于模型估计的点集是否是退化点集的函数句柄
  %如果给定的4个点中有三点共线，该点集便为退化点集
  degenfn = @isdegenerate; 
  
  %传给ransac函数的数据是6xN的格式
  inliers = ransac([pp1Homo; pp2Homo], fittingfn, distfn, degenfn, n, t);
   
  %最后，从最大的一致集中再用最小二乘法拟合出H
  H = homography2d([pp1Homo(:,inliers); pp2Homo(:,inliers)]);
end

%----------------------------------------------------------------------
% x是数据集合，每一列为一个点对，每列是6维向量，前3行为第一个点，后三行为第2个点。
% 该函数计算对应点在当前射影矩阵H下的匹配误差，该误差为双向的。
% 比如x1与x1'为一对对应点对，那么需要计算||x1'-Hx1||^2+||x1-H^(-1)x1'||^2，
% 注意：计算该位置误差时，点的坐标都要先转化成规范化齐次坐标
% 返回数据x中与当前射影矩阵H相一致（即距离要小于阈值t）的内点索引
function inliers = homogdist2d(H, x, t)
  x1 = x(1:3,:); % Extract x1 and x2 from x   
  x2 = x(4:6,:);    
  %从两个方向计算经H（或者H^(-1)）变换后的点坐标    
  Hx1    = H*x1;
  invHx2 = H\x2;
    
  %计算距离之前转换成规范化齐次坐标    
  x1     = hnormalise(x1);
  x2     = hnormalise(x2);     
  Hx1    = hnormalise(Hx1);
  invHx2 = hnormalise(invHx2); 
    
  %计算在当前H下，对应点的双向重投影误差
  d2 = sum((x1-invHx2).^2)  + sum((x2-Hx1).^2);
  %误差值小于t的点对被认为是当前模型H的内点
  inliers = find(abs(d2) < t);    
end

%----------------------------------------------------------------------
%x为6*4矩阵，前三行是图像1中四个特征点的齐次坐标x11,x12,x13,x14
%后三行为图像2中对应的四个特征点的齐次坐标x21,x22,x23,x24
%该函数判断x11,x12,x13,x14中或x21,x22,x23,x24中是否存在三点共线的情况
function r = isdegenerate(x)

  x1 = x(1:3,:);    % Extract x1 and x2 from x
  x2 = x(4:6,:);    
    
  r = ...
  iscolinear(x1(:,1),x1(:,2),x1(:,3)) | ...
  iscolinear(x1(:,1),x1(:,2),x1(:,4)) | ...
  iscolinear(x1(:,1),x1(:,3),x1(:,4)) | ...
  iscolinear(x1(:,2),x1(:,3),x1(:,4)) | ...
  iscolinear(x2(:,1),x2(:,2),x2(:,3)) | ...
  iscolinear(x2(:,1),x2(:,2),x2(:,4)) | ...
  iscolinear(x2(:,1),x2(:,3),x2(:,4)) | ...
  iscolinear(x2(:,2),x2(:,3),x2(:,4));
end

%判断平面内三点p1、p2、p3是否共线
function r = iscolinear(p1, p2, p3)
  %若p2-p1与p3-p1的叉乘为零向量，则说明3点共线
  r = norm(cross(p2-p1, p3-p1)) < eps;
end