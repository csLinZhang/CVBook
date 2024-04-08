%对于非线性分类问题，可通过非线性SVM来进行分类
%示范RBF核函数的使用
rng(1); % For reproducibility
r = sqrt(0.5*rand(100,1)); % Radius
t = 2*pi*rand(100,1);  % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points
r2 = sqrt(3*rand(100,1)+1); % Radius
t2 = 2*pi*rand(100,1);      % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

figure;
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
axis equal
hold off
data3 = [data1;data2]; %形成样本集，一共200个样本
%theclass, 类标集合，一共200个类标
theclass = ones(200,1);
theclass(1:100) = -1;

%训练SVM分类器，用RBF核函数
cl = fitcsvm(data3,theclass,'KernelFunction','rbf', 'BoxConstraint',...
    Inf,'ClassNames',[-1,1]);
%以0.02为分辨率，用训练好的SVM分类模型cl对每个平面点进行分类预测
%要得到平面上每个预测点处的分类响应值scores，主要目的是为了可视化出分隔曲线
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)), ...
    min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(cl,xGrid);

%画出数据分布散点图，并画出决策分类面
figure;
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
h(3) = plot(data3(cl.IsSupportVector,1),data3(cl.IsSupportVector,2),...
    'linestyle','none','markersize', 8, 'marker', 'o', 'MarkerEdgeColor','k' );
%scores的等高线绘制，参数[0 0]表示只绘制scores为0处的等高线
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal
hold off