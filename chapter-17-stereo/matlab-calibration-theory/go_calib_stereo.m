% 该代码可完成双目相机外参标定，主要用于展示教材中说讲授的外参标定方法在Matlab中应该如何实现，帮助读者更好理解
% 该代码修改自Jean-Yves Bouguet的相机标定程序包，原始程序包可在http://robots.stanford.edu/cs223b04/JeanYvesCalib/下载
% 作者对该程序包进行了大量简化，只保留代码的主干部分，其目的是为了让读者更好地理解和学习教材中所讲授的理论
% 该程序固定了左右相机的内参，左右相机的内参通过单目标定获得，在双目标定过程中作为已知量不参与迭代优化，保持不变
% 同济大学，张林，2024年5月

% 程序中主要变量说明
% om, R, T: 双目系统的外参，R为旋转矩阵，om为与R对应的轴角向量，T为平移向量
% 外参按如下定义：一点在左相机坐标系中的坐标为XL，在右相机坐标系中的坐标为XR，则有，
% XR = R * XL + T

%读入左右两个相机的内参数
%在左右相机进行单目标定时，也保存了它们相对于世界坐标系的外参。
%从一组左右目相机相对于世界坐标系的外参，可以估计出一个双目外参（轴角，平移向量）
%这样，一共会估计出14（14是本例双目图像组数）组双目外参，然后取它们的中值，
%作为双目外参优化过程中双目外参的初始值
%所用标定板上交叉点数为54个，共拍摄14组双目图像
load_stereo_calib_files;
fprintf(1,'Gradient descent iterations: ');
    
MaxIter = 100; %优化时的最大迭代次数
change = 1; %记录两次迭代外参向量差异的模，如果差异很小则迭代停止
iter = 1; %记录已经经过的迭代次数

while (change > 1e-11)&&(iter <= MaxIter)
    fprintf(1,'%d...',iter);
    J = []; %误差函数项所组成的矢量函数的雅可比矩阵
    e = []; %误差项

    %param, 优化变量
    %(om,T)，双目系统外参，其初始值已经在load_stereo_calib_files步骤中初始化好，是由拍摄每组双目图像时
    %左右相机世界外参估算出的双目参数的中值
    param = [om;T];

    for kk = 1:n_ima %n_ima是双目图像对的个数，在本例中为14
        %Xckk，标定板所确定的世界坐标系下，标定板交叉点的世界坐标，dim(Xckk)= 3*54
        %(omckk,Tckk)，拍摄第kk组双目图像时，左相机相对于世界坐标系的外参，即轴角向量和平移向量
        %xlkk，标定板三维交叉点所对应的左图像上的像素点坐标，dim(xlkk)= 2*54
        eval(['Xckk = X_left_' num2str(kk) ';']);
        eval(['omckk = omc_left_' num2str(kk) ';']);
        eval(['Tckk = Tc_left_' num2str(kk) ';']);
        eval(['xlkk = x_left_' num2str(kk) ';']);

        %对于右相机来说，只需要交叉点的二维像素坐标
        %xrkk，标定板交叉点所对应的右图像上的像素点坐标，dim(xrkk)= 2*54
        eval(['xrkk = x_right_' num2str(kk) ';']);
            
        %左相机相对于世界系的外参(omckk,Tckk)，也作为待优化变量
        param = [param;omckk;Tckk];
            
        %Nckk, 标定板上交叉点的个数，54个
        Nckk = size(Xckk,2);       
            
        %Jkk是与第kk个双目图像对关联的Jocobian矩阵块，ekk是相应的误差
        %每个交叉点会产生2个误差项（左右双目），而每个误差项实际上包含了2行（像素是2维的），
        %因此，第kk组双目图像形成的Jkk矩阵的行数是2*2*Nckk=4*Nckk
        %Jkk的列数是待优化变量的维度，优化变量是双目外参(om,T)，这是6个维度；
        %拍摄每一组双目图像时，左相机在世界系下的外参(omckk,Tckk)也是6个维度，共n_ima组；
        %因此，优化变量维度总计是6+n_ima*6=(1+n_ima)*6
        %ekk，是与kk组双目图像所关联的误差函数项所组成的矢量函数的值
        Jkk = sparse(4*Nckk, (1+n_ima)*6);
        ekk = zeros(4*Nckk,1);
            
        %根据第kk个左相机的内参和外参，把世界三维点Xckk投影到左相机成像平面上（结果为xl），
        %计算与左相机观测像素点的坐标误差ekk，并计算雅可比矩阵块Jkk
        %dim(xl)=2*54, dim(dxldomckk)=108*3, dim(dxldTckk)=108*3
        %标定板交叉点数为54，因此与左目图像关联的误差项有108个
        %dxldomckk，误差项对第kk个相机在世界坐标系下的轴角的导数
        %dxldTckk，误差项对第kk个相机在世界坐标系下的平移向量的导数
        [xl,dxldomckk,dxldTckk,~,~,~,~] = ...
            project_points2(Xckk,omckk,Tckk,fc_left,cc_left,kc_left,alpha_c_left);
 
        %计算与左目关联的投影像素点位置与观测像素点位置的误差
        %xlkk(:)与xl(:)是108维向量
        ekk(1:2*Nckk) = xlkk(:) - xl(:); 
            
        %填充Jkk中与左目相机有关的内容，即误差项对左目相机kk的外参导数
        Jkk(1:2*Nckk,6*(kk-1)+7:6*(kk-1)+7+2) = sparse(dxldomckk);
        Jkk(1:2*Nckk,6*(kk-1)+7+3:6*(kk-1)+7+5) = sparse(dxldTckk);

        %根据双目外参(om,T)和左目世界系外参(omckk,Tckk)写出右目世界系外参的表达(omr,Tr)
        %根据公式17-8，Rr=RRl, tr=Rtl+t
        [omr,Tr,domrdomckk,domrdTckk,domrdom,domrdT,dTrdomckk,dTrdTckk,dTrdom,dTrdT] = ...
            compose_motion(omckk,Tckk,om,T);
            
        %根据当前右相机的内参和外参，把世界三维点投影到右相机成像平面上（结果为xr），并计算Jacobian
        [xr,dxrdomr,dxrdTr,dxrdfr,dxrdcr,dxrdkr,dxrdalphar] = ...
            project_points2(Xckk,omr,Tr,fc_right,cc_right,kc_right,alpha_c_right);
        ekk(2*Nckk+1:end) = xrkk(:) - xr(:); %计算投影像素点位置与观测像素点位置的误差，并拉成一列（108维）
        
        %误差项对双目外参(om,T)的导数
        dxrdom = dxrdomr * domrdom + dxrdTr * dTrdom;
        dxrdT = dxrdomr * domrdT + dxrdTr * dTrdT;
        %根据链式求导法则，计算误差项对左目相机世界系外参(omckk,Tckk)的导数
        dxrdomckk = dxrdomr * domrdomckk + dxrdTr * dTrdomckk;
        dxrdTckk = dxrdomr * domrdTckk + dxrdTr * dTrdTckk;
            
        %填充Jkk中与双目外参有关的部分
        Jkk(2*Nckk+1:end,1:3) = sparse(dxrdom);
        Jkk(2*Nckk+1:end,4:6) = sparse(dxrdT);

        %填充Jkk中与左目相机世界系外参有关的部分
        Jkk(2*Nckk+1:end,6*(kk-1)+7:6*(kk-1)+7+2) = sparse(dxrdomckk);
        Jkk(2*Nckk+1:end,6*(kk-1)+7+3:6*(kk-1)+7+5) = sparse(dxrdTckk);

        %将当前第kk个双目图像对所形成的Jacobian项接到整体J后面
        %将当前第kk个双目图像对所形成的误差项接到整体误差项e后面
        J = [J;Jkk];
        e = [e;ekk];           
    end
    
    J2 = J'*J;
    J2_inv = inv(J2);
    
    %牛顿法求解非线性最小二乘问题，教材公式9-15，求得本轮优化变量更新量
    param_update = J2_inv*J'*e; 
    %完成本轮优化变量更新
    param = param + param_update; 

    om_old = om; %om_old, 当前双目轴角
    T_old = T; %T_old, 当前双目平移
    
    om = param(1:3); %完成更新之后的双目轴角
    T = param(4:6); %完成更新之后的双目平移
        
    %完成更新之后的左目世界坐标系下的外参
    for kk = 1:n_ima
        omckk = param(6*(kk-1)+7:6*(kk-1)+7+2);
        Tckk = param(6*(kk-1)+7+3:6*(kk-1)+7+5);
        eval(['omc_left_' num2str(kk) ' = omckk;']);
        eval(['Tc_left_' num2str(kk) ' = Tckk;']);
    end

    %计算与上一次的外参相比，变化量有多大
    change = norm([T;om] - [T_old;om_old])/norm([T;om]); 
    iter = iter + 1;
end

R = rodrigues(om);
fprintf(1,'done\n');

fprintf(1,'\n\n\nStereo calibration parameters after optimization:\n');

fprintf(1,'\n\nIntrinsic parameters of left camera:\n\n');
fprintf(1,'Focal Length:          fc_left = [ %3.5f   %3.5f ] \n',fc_left);
fprintf(1,'Principal point:       cc_left = [ %3.5f   %3.5f ] \n',cc_left);
fprintf(1,'Skew:             alpha_c_left = [ %3.5f ] => angle of pixel axes = %3.5f degrees\n',alpha_c_left,90 - atan(alpha_c_left)*180/pi);
fprintf(1,'Distortion:            kc_left = [ %3.5f   %3.5f   %3.5f   %3.5f  %5.5f ] \n',kc_left);   

fprintf(1,'\n\nIntrinsic parameters of right camera:\n\n');
fprintf(1,'Focal Length:          fc_right = [ %3.5f   %3.5f ] \n',fc_right);
fprintf(1,'Principal point:       cc_right = [ %3.5f   %3.5f ] \n',cc_right);
fprintf(1,'Skew:             alpha_c_right = [ %3.5f ] => angle of pixel axes = %3.5f degrees\n',alpha_c_right,90 - atan(alpha_c_right)*180/pi);
fprintf(1,'Distortion:            kc_right = [ %3.5f   %3.5f   %3.5f   %3.5f  %5.5f ] \n',kc_right);   

fprintf(1,'\n\nExtrinsic parameters (position of right camera wrt left camera):\n\n');
fprintf(1,'Rotation vector:             om = [ %3.5f   %3.5f  %3.5f ] \n',om);
fprintf(1,'Translation vector:           T = [ %3.5f   %3.5f  %3.5f ] \n',T);

ext_calib_stereo
