% RANSAC算法框架
% x，数据集，对于我们的问题来说x为6*npts的矩阵，每一列为两个对应点的坐标，npts为点对个数
% fittingfn，模型拟合函数句柄
% distfn，计算数据点在当前拟合模型下"距离"的函数句柄
% n，模型拟合所需最小数据点个数
% t，内点判断阈值
% inliers，最大一致集中的内点索引，根据这个索引，在外层调用程序再最后做一次模型估计
function inliers = ransac(x, fittingfn, distfn, degenfn, n, t)
  %最多尝试maxTrials次模型初始化，从中选出具有最大一致集的模型，并将它的一致集数据索引返回 
  maxTrials = 1000;   
  %从数据集中挑选s个数据点来初始化模型，但如果从挑选的s个数据点中无法初始化模型，
  %就记录失败一次，最多允许失败maxDataTrials次
  maxDataTrials = 100; 
  [~, npts] = size(x);                 
  %迭代完成后，要求至少有一次所有选取的4个随机点均为内点的概率要保证不低于p
  p = 0.99; %书中式6-1中的p 

  bestM = NaN; %当前最好的初始化模型，在该实现中是当前具有最大一致集的模型
  trialcount = 0;  %记录一共已经迭代了多少次
  bestscore = 0;  %用于记录当前找到的最好的随机初始模型的一致集中元素的个数  
  N = 1; % 需要迭代的次数，这是从内点比例计算出来的，相当于书中式6-2中的k
    
  while N > trialcount
    %随机选取n个点来拟合模型，需要检查该随机数据集合是否为不能拟合出模型的退化集合
    degenerate = 1;
    count = 1;
    while degenerate
        %从npts点中，随机选取n个，对于射影矩阵估计问题来说，n=4
	    ind = randsample(npts, n); 
        % 判断ind所索引的这4个点是否是退化的，即是否存在三点共线的情况
        degenerate = feval(degenfn, x(:,ind));
            
        if ~degenerate %当前的4个点是非退化的，即可以拟合出一个模型来
           % 从这4个随机选择的点拟合出模型M
           M = feval(fittingfn, x(:,ind));
           % 如果拟合模型失败，也将degenerate置为1
           if isempty(M)
              degenerate = 1;
           end
        end
            
        %如果尝试了maxDataTrials次依然没有从数据集中选取出能拟合模型的4个点，失败
        count = count + 1;
        if count > maxDataTrials
           warning('Unable to select a nondegenerate data set');
           break
        end
     end
        
     % 计算出当前拟合模型M的一致集inliers，当然inliers只是一些索引，
     % 用于标记出数据集x中与当前模型M一致的内点的索引
     inliers = feval(distfn, M, x, t);
     ninliers = length(inliers); %当前模型一致集中元素的个数
      
     %当前一致集元素个数大于已知最好的一致集元素个数，
     %则需要更新当前最优初始化模型、其一致集以及最大迭代次数
     if ninliers > bestscore 
        bestscore = ninliers; %更新最好一致集元素个数  
        bestinliers = inliers; %更新最好一致集元素索引
        bestM = M; %更新最优初始化模型
            
        %根据当前最优初始化模型的一致集的元素个数，可以动态调整所需最大迭代次数。
        %需要注意到：当前一致集越大，实际上所需的外层while迭代会越少
        fracinliers = ninliers/npts; %内点比例，书中式6-1中的omega
        pNoOutliers = 1 - fracinliers^n; %式6-1中，1-omega^n
        pNoOutliers = max(eps, pNoOutliers);  % Avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers);% Avoid division by 0.
        N = log(1-p)/log(pNoOutliers); %书中式6-2，N相当于书中的k
     end
        
     trialcount = trialcount+1;

     % 迭代次数超够了最大限度，停止
     if trialcount > maxTrials
         break
     end     
  end
    
  if ~isnan(bestM)
     inliers = bestinliers;
  else           
     inliers = [];
     error('ransac was unable to find a useful solution');
  end
end