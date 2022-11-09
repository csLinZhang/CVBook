%软间隔支持向量机，基于凸二次规划求解的实现
%同济大学，张林，2022年10月
%随机模拟一些点作为训练数据
random = unifrnd(-1,1,50,2);
%随机生成两组2维数据
group1 = ones(50,2) + random; 
group1(50,:) = [4,5]; 
group2 = 3.5*ones(50,2) + random;
group2(50,:) = [1,0.1]; 

C = 3;

X=[group1;group2]; %X存储样本的特征，每行一个样本，前50个是group1，后50个是group2
data=[group1,-1*ones(50,1);group2,1*ones(50,1)];%形成带label的数据集

y=data(:,end);%训练数据的label

Q=zeros(length(X),length(X));%目标函数中的H。见教材式17-20。
for i=1:length(X)%对于所有样本都要遍历
    for j=1:length(X)
        Q(i,j)=X(i,:)*(X(j,:))'*y(i)*y(j);
    end
end
q=-1*ones(length(X),1);%二次函数中的q

%quadprog调用接口中需要的A和b。A和b的构造方法见教材式17-20。
%quadprog这个程序中，优化问题的形式是，1/2x^(T)Hx+f^(T)x, subject to Ax<=b
A = [];
b= [];
Aeq = y'; %quadprog这个函数中要求的A
beq = zeros(1,1); %quadprog这个函数中要求的b。
lb = zeros(length(X),1);
ub = ones(length(X),1)*C;

[alpha,fval]=quadprog(Q, q, A, b, Aeq,beq,lb,ub);%二次规划问题

%把太小的alpha分量直接置为0
tooSmallIndex = alpha<1e-04;
alpha(tooSmallIndex) = 0;

%下面计算最优分类超平面参数w和b
w=0;
sumPartInb=0; %教材式17-22中，计算b时，求和的部分
svIndices = find(alpha~=0); %找到支持向量样本，alpha为0的分量所对应的样本就是支持向量，这是根据KKT条件得到的，见教材式17-24
j = svIndices(1);%找到一个不为0的alpha分量的下标
for i=1:length(svIndices) %
    w = w+alpha(svIndices(i))*y(svIndices(i))*X(svIndices(i),:)';
    sumPartInb = sumPartInb + alpha(svIndices(i))*y(svIndices(i))*(X(svIndices(i),:)*X(j,:)');
end
b = y(j)-sumPartInb;

%画出点以及对应的超平面
% figure
% gscatter(X(:,1),X(:,2),y); %绘制散点图
% xlabel('dimension1')
% ylabel('dimension2')
% legend('group1','group2')


%画出点以及对应的超平面
figure
gscatter(X(:,1),X(:,2),y); %绘制散点图



supportVecs = X(svIndices,:);

hold on
plot(supportVecs(:,1),supportVecs(:,2),'ko','MarkerSize',10) %圈出支持向量

hold on
k=-w(1)./w(2); %将直线改写成斜截式便于作图
bb=-b./w(2);
xx=0:4;
yy=k.*xx+bb;
plot(xx,yy,'-')
hold on
yy=k.*xx+bb+1./w(2);
plot(xx,yy,'--')
hold on
yy=k.*xx+bb-1./w(2);
plot(xx,yy,'--')
title('support vector machine')
xlabel('dimension1')
ylabel('dimension2')
legend('group1','group2','support vector','separating hyperplane')