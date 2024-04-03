%该函数把非归一化齐次坐标化成归一化齐次坐标，就是最后一个维度化为1

function nx = hnormalise(x)
    
    [rows,~] = size(x);
    nx = x;
    nx(1,:) = x(1,:)./x(rows,:);
    nx(2,:) = x(2,:)./x(rows,:);
    nx(rows,:) = 1;

    
