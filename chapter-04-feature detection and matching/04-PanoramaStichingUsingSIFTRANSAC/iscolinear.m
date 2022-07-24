%判断3点是否共线
function r = iscolinear(p1, p2, p3)

%若p2-p1与p3-p1的叉乘为零向量，则说明3点共线
r =  norm(cross(p2-p1, p3-p1)) < eps;

    
