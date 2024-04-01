%给Harris角点建立block描述子，descriptors中存放的是角点的描述子
%valid_points，返回有效的角点集合，因为block描述子要用到角点周围邻域的信息，如果某个角点太接近图像边界了，就无法构建描述子了，需要把相应的角点从角点集合中删除
%points, Harris角点坐标
%blockSize，建立block描述子时，block的边长
function [descriptors, valid_points] = extractBlockDescriptors(I, points, blockSize)
    
%每一个描述子的长度
lengthFV = blockSize*blockSize;

nPoints = size(points,1);
%描述子矩阵，每一行对应一个角点的描述子向量
descriptors = zeros(nPoints, lengthFV);
%用来记录最终有效的角点
valid_indices = zeros(nPoints, 1);

[nRows, nCols] = size(I);
halfSize =  (blockSize-mod(blockSize, 2)) / 2; %block长度的一半

nValidPoints = 0;

%遍历角点集合，为每一个角点创建描述子；如靠近图像边界的话，此角点被标记为无效
for pointIndex = 1:nPoints
    c = points(pointIndex,1);
    r = points(pointIndex,2);

    % 检查一下角点是否太靠近图像边界
    if (c > halfSize && c <= (nCols - halfSize) && r > halfSize && r <= (nRows - halfSize))
        % 如果在图像内部，nValidPoints增加一个
        nValidPoints = nValidPoints + 1;
        % 以当前角点为中心，把它周围block拉成一个列向量，作为这个角点的描述子
        descriptors(nValidPoints, :) = reshape(I(r-halfSize:r+halfSize, c-halfSize:c+halfSize), 1, lengthFV);
        
        %把向量进行归一化，使其l2-norm为1
        l2norm = norm(descriptors(nValidPoints, :));
        if l2norm < eps(single(1))
            descriptors(nValidPoints, :) = 0;
        else
            descriptors(nValidPoints, :) = descriptors(nValidPoints, :) / l2norm;
        end
        %这个角点为有效角点，把它的索引存入valid_indices
        valid_indices(nValidPoints) = pointIndex;
    end
end

%只保存有效角点的描述子；对于有效角点，valid_indices记录了它在原始角点集合points中的位置
descriptors = descriptors(1:nValidPoints, :);
valid_indices = valid_indices(1:nValidPoints,:);

valid_points = points(valid_indices,:);
