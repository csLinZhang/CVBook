% 该代码修改自Jean-Yves Bouguet的相机标定程序包，原始程序包可在http://robots.stanford.edu/cs223b04/JeanYvesCalib/下载
% 同济大学，张林，2024年5月

%该代码可根据双目系统的内外参数来构建校正化双目系统，并将由物理双目系统所拍摄的
%双目图像映射到校正化双目相机的成像平面上，得到逐行对齐的校正化双目图像
%运行本代码之前，必须要成功运行go_calib_stereo.m，来得到本代码所需要的Matlab工作区变量

%om是双目系统外参中的旋转部分，以轴角表达，将其转换成旋转矩阵R
R = rodrigues(om);

%对应于书中构建Calign-l和Calign-r坐标系的部分
r_r = rodrigues(-om/2); %对应于书中rodrigues(-d/2)
r_l = r_r'; %对应于书中rodrigues(d/2)
%t为基线向量，对应于书中t'，T是双目外参中的平移部分
t = -r_r * T;
% uu对应于书中u
uu = [1;0;0]; 
%ww，对应于书中w
ww = cross(t,uu);
ww = ww/norm(ww);
%此时的ww对应于书中w*theta
ww = acos(abs(dot(t,uu))/(norm(t)*norm(uu)))*ww;
R2 = rodrigues(ww); %对应于书中rodrigues(w*theta)

% R_R，对应于书中Rrect-r
% R_L，对应于书中Rrect-l
R_R = R2 * r_r;
R_L = R2 * r_l;

%校正化双目系统的外参，旋转矩阵R_new为单位矩阵，对应的轴角om_new为零向量
%T_new，校正化双目系统外参中的平移部分，对应于教材中trect
R_new = eye(3);
om_new = zeros(3,1);
T_new = R_R*T;

%设置校正化双目相机的焦距以及主点坐标
fc_new = min(min(fc_left),min(fc_right));
fc_left_new = round([fc_new;fc_new]);
fc_right_new = round([fc_new;fc_new]);
cc_left_new = [(cc_left(1) + cc_right(1))/2; (cc_left(2) + cc_right(2))/2];
cc_right_new = cc_left_new;

% 左右校正相机的内参矩阵
KK_left_new = [fc_left_new(1) 0 cc_left_new(1);...
               0 fc_left_new(2) cc_left_new(2);...
               0 0 1];
KK_right_new = [fc_right_new(1) 0 cc_right_new(1);...
                0 fc_right_new(2) cc_right_new(2);...
                0 0 1];

%左右校正化相机畸变系数都为0
kc_left_new = zeros(5,1);
kc_right_new = zeros(5,1);

%ind_new_left，Irect_l中会在Il上有对应像素位置的合法像素位置索引
%ind_1_left，ind_2_left，ind_3_left，ind_4_left：对于Irect_l每一个位置来说，
%它在Il映射位置处的四个整数位置上的邻居的位置索引
%a1_left,a2_left,a3_left,a4_left：4个邻居的对应权重
[ind_new_left,ind_1_left,ind_2_left,ind_3_left,ind_4_left,a1_left,a2_left,a3_left,a4_left] = ...
    rect_index(zeros(ny,nx),R_L,fc_left,cc_left,kc_left,KK_left_new);
[ind_new_right,ind_1_right,ind_2_right,ind_3_right,ind_4_right,a1_right,a2_right,a3_right,a4_right] = ...
    rect_index(zeros(ny,nx),R_R,fc_right,cc_right,kc_right,KK_right_new);

% Loop around all the frames now: (if there are images to rectify)
calib_name_left = 'left';
calib_name_right = 'right';
format_image_left = 'jpg';
format_image_right = 'jpg';
type_numbering_left = '1';
type_numbering_right = '1';
image_numbers_left = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
image_numbers_right = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
N_slots_left = '2';
N_slots_right = '2';

if ~isempty(calib_name_left)&&~isempty(calib_name_right)
    
    fprintf(1,'Rectifying all the images (this should be fast)...\n\n');
    
    % Rectify all the images: (This is fastest way to proceed: precompute the set of image indices, and blending coefficients before actual image warping!)
    
    for kk = find(active_images)
        
        % Left image:
        
        I = load_image(kk,calib_name_left,format_image_left,type_numbering_left,image_numbers_left,N_slots_left);
        
        fprintf(1,'Image warping...\n');
        
        I2 = 255*ones(ny,nx);
        I2(ind_new_left) = uint8(a1_left .* I(ind_1_left) + a2_left .* I(ind_2_left) + a3_left .* I(ind_3_left) + a4_left .* I(ind_4_left));
        
        write_image(I2,kk,[calib_name_left '_rectified'],format_image_left,type_numbering_left,image_numbers_left,N_slots_left ),
        
        fprintf(1,'\n');
        
        % Right image:
        
        I = load_image(kk,calib_name_right,format_image_right,type_numbering_right,image_numbers_right,N_slots_right);
        
        fprintf(1,'Image warping...\n');
        
        I2 = 255*ones(ny,nx);
        I2(ind_new_right) = uint8(a1_right .* I(ind_1_right) + a2_right .* I(ind_2_right) + a3_right .* I(ind_3_right) + a4_right .* I(ind_4_right));
        
        write_image(I2,kk,[calib_name_right '_rectified'],format_image_right,type_numbering_right,image_numbers_right,N_slots_right );
        
        fprintf(1,'\n');
        
    end
    
end

