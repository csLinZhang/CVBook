%此函数完成Harris角点检测任务
%rows和cols存储了角点所在的行与列
function [rows, cols] = harrisCornerDetector(im, sigma, thresh, nonmaxrad)
%im: 要在其上检测角点的灰度图像
%sigma: harris角点检测算法中需要用高斯窗口来计算矩阵M，这个sigma用来确定高斯窗口的大小6*sigma
%thresh：每个位置计算出角点程度值之后，大于thresh的点被放进角点候选集合
%nonmaxrad：进行非局部极大值抑制操作时，局部窗口的半径
    if ~isa(im,'single')
    	im = single(im);
    end

    %计算图像I的偏导数图像.
    sobelFilter = fspecial('sobel');
    Iy = imfilter(im, sobelFilter);
    Ix = imfilter(im, sobelFilter');
    
    %计算矩阵M，注意对于每个位置，会有一个矩阵M
    gaussFilter = fspecial('gaussian', ceil(6*sigma), sigma);
    Ix2 = imfilter(Ix.^2,  gaussFilter);
    Iy2 = imfilter(Iy.^2,  gaussFilter);    
    Ixy = imfilter(Ix.*Iy, gaussFilter);    

    %计算Harris角点程度值 cornerness = det(M) - k*trace(M)^2
    k = 0.04;
    cornernessMap = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2; 
    
    %在cornernessMap的基础之上，进行阈值化，并进行非极大值抑制操作，得到最终的角点集合
	[rows, cols] = nonmaxsuppts(cornernessMap, nonmaxrad, thresh);
    end
    
