% 该代码修改自Jean-Yves Bouguet的相机标定程序包，原始程序包可在http://robots.stanford.edu/cs223b04/JeanYvesCalib/下载
% 作者对该程序包进行了大量简化，只保留代码的主干部分，其目的是为了让读者更好地理解和学习教材中所讲授的理论
% 同济大学，张林，2024年5月

% 该函数读入左右目相机的内参数以及它们相对于世界坐标系的外参（拍摄每张图像时，相机有一个外参）
% 由每组左右相机的世界系下的外参估计出双目外参，这样一共会有m组（双目图像对数）双目外参，取它们的中值作为双目外参的初始值

% 存储左右相机内参的本地文件
% 实际上读入的数据不单单只是相机内参，还有标定内参时的相关数据，包括标定板交叉点的世界坐标，交叉点在图像上投影像素点坐标，相机
% 在拍摄每组图像时相对于世界坐标系的外参
dataDir = '.\stereo_example\';
calib_file_name_left = [dataDir 'Calib_Results_left.mat'];
calib_file_name_right = [dataDir 'Calib_Results_right.mat'];

% 读入左相机参数
load(calib_file_name_left);

fc_left = fc; %左目焦距，2*1向量
cc_left = cc; %左目主点，2*1向量
kc_left = kc; %左目镜头畸变系数，5*1向量
alpha_c_left = alpha_c; %左目扭曲系数，scalar
KK_left = KK; %左目内参矩阵，3*3

%存储左目相机在拍摄每组双目图像时标定板交叉点在该左目相机坐标系下的坐标，以及
%该相机在世界坐标系下的外参（旋转向量与平移向量）
X_left = [];
om_left_list = []; 
T_left_list = [];

for kk = 1:n_ima %n_ima，双目图像对数，本例为14
    eval(['Xkk = X_' num2str(kk) ';']);
    eval(['omckk = omc_' num2str(kk) ';']);
    eval(['Rckk = Rc_' num2str(kk) ';']);
    eval(['Tckk = Tc_' num2str(kk) ';']);

    N = size(Xkk,2); %N，标定板上交叉点个数，本例为54
    %Xkk, 3*54，每一列为一个交叉点世界坐标
    %(Rckk,Tckk)，左相机在拍摄第kk组照片时的世界外参，omckk为Rckk对应的轴角
    %Xckk，3*54，标定板上交叉点在第kk次拍摄图像时在左目相机坐标系下的坐标
    Xckk = Rckk * Xkk  + Tckk*ones(1,N);
    %记录拍摄第kk组双目图像时，左目相机坐标系下交叉点的坐标和左目相机世界外参
    X_left = [X_left Xckk];
    om_left_list = [om_left_list omckk];
    T_left_list = [T_left_list Tckk];
end
% 运行到这里时，dim(X_left)=3*756(54*14), dim(om_left_list)=3*14, dim(T_left_list)=3*14

% 调入右目相机参数
load(calib_file_name_right);
fc_right = fc; %右目焦距，2*1向量
cc_right = cc; %右目主点，2*1向量
kc_right = kc; %右目畸变系数，5*1向量
alpha_c_right = alpha_c; %右目扭曲系数，scalar
KK_right = KK; %右目内参矩阵，3*3

%存储右目相机在拍摄每组双目图像时标定板交叉点在该右目相机坐标系下的坐标，以及
%该相机在世界坐标系下的外参（旋转向量与平移向量）
X_right = [];
om_right_list = [];
T_right_list = [];

for kk = 1:n_ima %n_ima，双目图像对数，本例为14
    eval(['Xkk = X_' num2str(kk) ';']);
    eval(['omckk = omc_' num2str(kk) ';']);
    eval(['Rckk = Rc_' num2str(kk) ';']);
    eval(['Tckk = Tc_' num2str(kk) ';']);
      
    N = size(Xkk,2); %N，标定板上交叉点个数，本例为54
    
    %Xkk, 3*54，每一列为一个交叉点世界坐标
    %(Rckk,Tckk)，右相机在拍摄第kk组照片时的世界外参,omckk为Rckk的轴角表示
    %Xckk，3*54，标定板上交叉点在第k次拍摄图像时在右目相机坐标系下的坐标
    Xckk = Rckk * Xkk  + Tckk*ones(1,N);

    %记录拍摄第kk组双目图像时，右目相机坐标系下交叉点的坐标和右目相机世界外参
    X_right = [X_right Xckk];
    om_right_list = [om_right_list omckk];
    T_right_list = [T_right_list Tckk];
end
% 运行到这里时，dim(X_right)=3*756(54*14), dim(om_right_list)=3*14, dim(T_right_list)=3*14

% 每次拍摄一组双目图像时，根据左右相机在世界系下的外参都可以计算出一组双目外参
% 把这些双目外参存储为列表形式
om_ref_list = [];
T_ref_list = [];
for ii = 1:size(om_left_list,2) %size(om_left_list,2)=14
    % (R_ref，T_ref)，双目系统外参，计算方式对应于公式17-8
    R_ref = rodrigues(om_right_list(:,ii)) * rodrigues(om_left_list(:,ii))';
    T_ref = T_right_list(:,ii) - R_ref * T_left_list(:,ii);
    om_ref = rodrigues(R_ref); %将旋转矩阵转换为轴角向量

    %将根据左右目相机在世界系下的外参估算出的双目外参存储为列表形式
    om_ref_list = [om_ref_list om_ref];
    T_ref_list = [T_ref_list T_ref];
end
% 运行到这里时，dim(om_ref_list)=3*14, dim(T_ref_list)=3*14

% 把双目外参最终的初始值估计为所有双目外参估计值的中值
om = median(om_ref_list,2);
T = median(T_ref_list,2);

R = rodrigues(om); %将轴角om转换为对应的旋转矩阵

% 重新读入左右目相机内参数，初始化几个参量值，为接下来的迭代优化做准备
% X_left_i，拍摄第i组双目图像时，标定板上交叉点的世界坐标（实际上都相同）
% x_left_i，拍摄第i组双目图像时，标定板上交叉点在左目图像上的像素投影
% X_right_i，拍摄第i组双目图像时，标定板上交叉点的世界坐标（实际上都相同）
% x_right_i，拍摄第i组双目图像时，标定板上交叉点在右目图像上的像素投影
% omc_left_i，拍摄第i组双目图像时，左目相机在世界系下的外参中的轴角
% Rc_left_i，拍摄第i组双目图像时，左目相机在世界系下的外参中的旋转矩阵
% Tc_left_i，拍摄第i组双目图像时，左目相机在世界系下的外参中的平移向量
% omc_left_i，Tc_left_i，om和T是接下来双目系统标定优化过程中的待优化变量
load(calib_file_name_left); 
for kk = 1:n_ima
      eval(['X_left_'  num2str(kk) ' = X_' num2str(kk) ';']);
      eval(['x_left_'  num2str(kk) ' = x_' num2str(kk) ';']);
      eval(['omc_left_' num2str(kk) ' = omc_' num2str(kk) ';']);
      eval(['Rc_left_' num2str(kk) ' = Rc_' num2str(kk) ';']);
      eval(['Tc_left_' num2str(kk) ' = Tc_' num2str(kk) ';']);
end
load(calib_file_name_right);
for kk = 1:n_ima
      eval(['X_right_'  num2str(kk) ' = X_' num2str(kk) ';']);
      eval(['x_right_'  num2str(kk) ' = x_' num2str(kk) ';']);
end

%打印内参值以及初始化好的双目外参
fprintf(1,'\n\n\nStereo calibration parameters after loading the individual calibration files:\n');

fprintf(1,'\n\nIntrinsic parameters of left camera:\n\n');
fprintf(1,'Focal Length:          fc_left = [ %3.5f   %3.5f ] \n',fc_left);
fprintf(1,'Principal point:       cc_left = [ %3.5f   %3.5f ] \n',cc_left);
fprintf(1,'Skew:             alpha_c_left = [ %3.5f ] => angle of pixel axes = %3.5f degrees\n',alpha_c_left,90 - atan(alpha_c_left)*180/pi);
fprintf(1,'Distortion:            kc_left = [ %3.5f   %3.5f   %3.5f   %3.5f  %5.5f ] \n',kc_left);   

fprintf(1,'\n\nIntrinsic parameters of right camera:\n\n');
fprintf(1,'Focal Length:          fc_right = [ %3.5f   %3.5f ] \n',fc_right);
fprintf(1,'Principal point:       cc_right = [ %3.5f   %3.5f ] \n',cc_right);
fprintf(1,'Skew:             alpha_c_right = [ %3.5f ]    => angle of pixel axes = %3.5f degrees\n',alpha_c_right,90 - atan(alpha_c_right)*180/pi);
fprintf(1,'Distortion:            kc_right = [ %3.5f   %3.5f   %3.5f   %3.5f  %5.5f ] \n',kc_right);   


fprintf(1,'\n\nExtrinsic parameters (position of right camera wrt left camera):\n\n');
fprintf(1,'Rotation vector:             om = [ %3.5f   %3.5f  %3.5f ]\n',om);
fprintf(1,'Translation vector:           T = [ %3.5f   %3.5f  %3.5f ]\n',T);
