%本示例程序演示Harris角点检测算法的实现
%张林，同济大学，2022年8月
    im = imread('officegray.bmp');
    
    %计算M矩阵时，高斯窗口的标准差
    sigma = 4.0;  
    %判断一个点是否是角点的阈值，cornerness>thresh才被认为可能是角点
    thresh = 200000;  
    nonmaxrad = 5; %做非最大值抑制操作时候，局部窗口的大小，必须设置为整数，最终的局部窗口边长为2*nonmaxrad+1
    
    %调用harrisCornerDetector函数，返回角点所在的行与列
    [rows, cols] = harrisCornerDetector(im, sigma,thresh, nonmaxrad);
    
    figure;
    imshow(im,[]), hold on, plot(cols, rows,'go','LineWidth',2);
    