function [ind_new,ind_1,ind_2,ind_3,ind_4,a1,a2,a3,a4] = rect_index(I,R,f,c,k,KK_new)
%为与书中内容对应，在以下注释中，相机都默认为是左目相机
%对于右目相机来说，过程是完全相同的
%输入：
% I，待填充的校正化左目图像，对应于书中Irect_l
% R，左目校正旋转矩阵，对应于书中Rrect-l
% f,c,k：物理左目相机Caml的焦距、主点坐标以及畸变系数向量
% KK_new：左目校正化相机内参矩阵，对应于书中Krect
%输出：
%ind_new：Irect_l中会在Il上有对应像素位置的合法像素位置索引
%ind_1，ind_2，ind_3，ind_4：对于Irect_l每一个位置来说，
%它在Il映射位置处的四个整数位置上的邻居的位置索引
%a1,a2,a3,a4：4个邻居的对应权重

%nr,nc，校正化左目图像的行数和列数
[nr,nc] = size(I);
[mx,my] = meshgrid(1:nc, 1:nr);
%px, py中存放的是校正化左目图像每个像素点的坐标
%nc*nr，像素点个数，本例中为307200
px = reshape(mx',nc*nr,1);
py = reshape(my',nc*nr,1);
%[(px - 1)';(py - 1)';ones(1,length(px))]，
%是校正化左目图像上像素位置的归一化齐次坐标
%rays，每个像素点所对应的Camrect-l归一化成像平面上点的齐次坐标
%也是这些点在Crect-l坐标系下的三维坐标
%dim(rays)= 3*307200
rays = inv(KK_new)*[(px - 1)';(py - 1)';ones(1,length(px))];

%R'，就是inv(Rrect-l)
%rays2，这是在Cl坐标系下的三维点
%dim(rays2)= 3*307200
rays2 = R'*rays;
%x，这是转换到了Cl下的归一化成像平面
%dim(x)= 2*307200
x = [rays2(1,:)./rays2(3,:);rays2(2,:)./rays2(3,:)];
%xd,对x施加物理左目相机Caml的畸变操作结果
%dim(xd)= 2*307200
xd = apply_distortion(x,k);

%从归一化成像平面坐标转换成像素坐标
px2 = f(1)*xd(1,:)+c(1);
py2 = f(2)*xd(2,:)+c(2);

%判断索引到Il的坐标是否超出了Il的边界，若超出边界，则原始Irect_l上对应
%像素位置就不再填充了
px_0 = floor(px2);
py_0 = floor(py2);
good_points = find((px_0 >= 0) & (px_0 <= (nc-2)) & (py_0 >= 0) & (py_0 <= (nr-2)));

%接下来为双线性插值准备，对于(px2,py2)定义的每个点，要确定
%出它的4个邻居，并计算出4个邻居的相应权重
px2 = px2(good_points);
py2 = py2(good_points);
px_0 = px_0(good_points);
py_0 = py_0(good_points);

alpha_x = px2 - px_0;
alpha_y = py2 - py_0;

a1 = (1 - alpha_y).*(1 - alpha_x);
a2 = (1 - alpha_y).*alpha_x;
a3 = alpha_y .* (1 - alpha_x);
a4 = alpha_y .* alpha_x;

ind_1 = px_0 * nr + py_0 + 1;
ind_2 = (px_0 + 1) * nr + py_0 + 1;
ind_3 = px_0 * nr + (py_0 + 1) + 1;
ind_4 = (px_0 + 1) * nr + (py_0 + 1) + 1;

ind_new = (px(good_points)-1)*nr + py(good_points);
return
