%此函数完成对cornernessMap的非极大值抑制操作，返回角点所在的行和列
function [rows,cols] = nonmaxsuppts(cornernessMap, radius, thresh)
%sze，要执行非局部极大值抑制的窗口边长
sze = 2*radius+1;       
%通过函数ordfilt2，localMaxMap中某个位置处的值为cornernessMap中
%以该位置为中心、在边长为sze的正方邻域内的最大值
localMaxMap = ordfilt2(cornernessMap,sze^2,ones(sze)); 

%做一个图像边界掩膜，去除掉边界位置处的角点
bordermask = zeros(size(cornernessMap));
bordermask(radius+1:end-radius, radius+1:end-radius) = 1;
    
%得到最终的角点集合，角点得同时满足3个条件：局部最大，大于阈值thresh，不在图像边界上
cornersMap = (cornernessMap==localMaxMap) & (cornernessMap>thresh) & bordermask;
[rows, cols] = find(cornersMap);  %返回角点图中，角点标记所在的行与列       
    
if isempty(rows)     
   fprintf('No maxima above threshold found\n');
end

