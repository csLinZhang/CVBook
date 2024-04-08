%软间隔支持向量机，基于凸二次规划求解的实现
%同济大学，张林，2024年4月

%随机模拟一些点作为训练数据
random = unifrnd(-1,1,50,2);
%随机生成两组2维数据
group1 = ones(50,2) + random; 
group1(50,:) = [4,5]; 
group2 = 3.5*ones(50,2) + random;
group2(50,:) = [1,0.1]; 

C = 3; %式14-25中的C
%X存储样本的特征，每行一个样本，前50个是group1，后50个是group2
X=[group1;group2]; 
%形成带label的数据集
data=[group1,-1*ones(50,1);group2,1*ones(50,1)];
y=data(:,end);%训练数据的label

%用quadprog函数求解式14-35，该目标函数的形式写成标准二次规划问题的形式就是
%式14-20
%matlab中quadprog函数能求解的二次规划问题的形式如下（请对照教材式14-20和14-35）
%min 0.5*x'*Q*x + q'*x, subject to:  A*x <= b, Aeq*x=beq, lb<=x<=ub
%需要根据我们的问题（式14-35）构造Q，q，A，b，Aeq，beq，lb和ub
Q=zeros(length(X),length(X));%构造Q，见教材式14-20。
for i=1:length(X)
    for j=1:length(X)
        Q(i,j)=X(i,:)*(X(j,:))'*y(i)*y(j);
    end
end
q=-1*ones(length(X),1);%二次函数中的q
%由于式14-35中没有形如A*x<=b的不等式约束，所以将A和b都置为空集
A = [];
b= [];
%基于式14-35中的等式约束来构造Aeq和beq
Aeq = y'; 
beq = zeros(1,1); 
%基于式14-35中的box约束来构造lb和ub
lb = zeros(length(X),1);
ub = ones(length(X),1)*C;
%求解二次规划问题，得到解alpha
[alpha,fval]=quadprog(Q, q, A, b, Aeq,beq,lb,ub);

%把太小的alpha分量直接置为0
tooSmallIndex = alpha<1e-04;
alpha(tooSmallIndex) = 0;

%下面计算最优分类超平面参数w和b
w=0;
sumPartInb=0; %教材式14-37中，计算b时，求和的部分
%找到支持向量，alpha为0的分量所对应的样本就是支持向量，
%这是根据KKT条件得到的，见教材式14-40
svIndices = find(alpha~=0); 
%找到一个不为0的alpha分量的下标，见命题14.2
j = svIndices(1); 
%根据式14-36和式14-37计算w*和b*
for i=1:length(svIndices) 
    w = w+alpha(svIndices(i))*y(svIndices(i))*X(svIndices(i),:)';
    sumPartInb = sumPartInb + ...
        alpha(svIndices(i))*y(svIndices(i))*(X(svIndices(i),:)*X(j,:)');
end
b = y(j)-sumPartInb;

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