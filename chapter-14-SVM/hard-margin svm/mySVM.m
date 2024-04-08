%硬间隔支持向量机，基于凸二次规划求解的实现
%同济大学，张林，2024年4月

%随机模拟一些点作为训练数据
random = unifrnd(-1,1,50,2);
%随机生成两组2维数据
group1 = ones(50,2) + random; 
group2 = 3*ones(50,2) + random;

X=[group1;group2]; %X存储样本的特征，每行一个样本，前50个是group1，后50个是group2
data=[group1,-1*ones(50,1);group2,1*ones(50,1)];%形成带label的数据集

y=data(:,end);%训练数据的label

%matlab中quadprog函数能求解的二次规划问题的形式如下，请对照教材式14-20
%min 0.5*x'*H*x + f'*x   subject to:  A*x <= b 

H=zeros(length(X),length(X));%目标函数中的H。教材中的Q，见教材式14-20。
for i=1:length(X)%对于所有样本都要遍历
    for j=1:length(X)
        H(i,j)=X(i,:)*(X(j,:))'*y(i)*y(j);
    end
end
f=-1*ones(length(X),1);%目标函数中的f

%quadprog调用接口中需要的A和b。A和b的构造方法见教材式14-20。
A = [y';-y';-eye(length(X),length(X))]; %quadprog这个函数中要求的A
b = zeros(length(X)+2,1); %quadprog这个函数中要求的b。
[alpha,fval]=quadprog(H,f,A,b);%二次规划问题

%把太小的alpha分量直接置为0
tooSmallIndex = alpha<1e-04;
alpha(tooSmallIndex) = 0;

%下面计算最优分类超平面参数w和b
w=0;
sumPartInb=0; %教材式14-22中，计算b时，求和的部分
svIndices = find(alpha~=0); %找到支持向量样本，alpha为0的分量所对应的样本就是支持向量，这是根据KKT条件得到的，见教材式14-24
j = svIndices(1);%找到一个不为0的alpha分量的下标
for i=1:length(svIndices) %
    w = w+alpha(svIndices(i))*y(svIndices(i))*X(svIndices(i),:)';
    sumPartInb = sumPartInb + alpha(svIndices(i))*y(svIndices(i))*(X(svIndices(i),:)*X(j,:)');
end
b = y(j)-sumPartInb;

%分类决策函数作预测
% predict=[];
% for i=1:length(X)%预测第i个样本
%     uu=0;%过度变量
%     for j=1:length(X)%利用训练集的所有样本构建预测函数
%         uu=uu+alpha(j)*y(j)*(X(j,:)*X(i,:)');
%     end
%     result=sign(uu+b);
%     predict(i,1)=result;
% end
% judge=(predict==y);
% score=sum(judge)./length(data);

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