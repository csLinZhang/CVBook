% 从点对集合data中按照最小二乘法估计射影矩阵，具体方法见教材5.1。
% data的维度为6xN，每一列是一个点对，每个点的表示方式必须是规范化二维齐次坐标
% H，3x3的射影矩阵，cx2 = H*x1

function H = homography2d(data)
    points1 = data(1:3,:); %得到点对关系中的第1组点
    points2 = data(4:6,:); %得到点对关系中的第2组点
    Npts = length(points1);
    A = zeros(2*Npts,9); %初始化系数矩阵
    
    O = [0 0 0];
    %此循环是构造系数矩阵A（见教材式5-7中的矩阵A）
    for i = 1:Npts
	    point1i = points1(:,i)';

	    xiprime = points2(1,i); 
        yiprime = points2(2,i); 

        A(2*i-1,:) = [point1i   O    -point1i*xiprime];
	    A(2*i  ,:) = [O     point1i  -point1i*yiprime];
    end
    
    %计算与矩阵A'*A最小特征值所对对应的特征向量smallestEigVector
    [smallestEigVector, ~] = eigs(A'*A, 1, 'smallestabs');
    H = reshape(smallestEigVector,3,3)';
end