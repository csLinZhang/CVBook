% RANSAC算法框架
% inliers，最大一致集中的内点索引，根据这个索引，在外层调用程序再最后做一次模型估计

function inliers = ransac(x, fittingfn, distfn, degenfn, s, t)

    maxTrials = 1000;    
    maxDataTrials = 100; 
    
    [~, npts] = size(x);                 
    
    p = 0.99; %迭代完成后，至少有一次所有选取的4个随机点均为内点的概率要保证不低于p

    bestM = NaN;      % Sentinel value allowing detection of solution failure.
    trialcount = 0;  %记录一共已经迭代了多少次
    bestscore =  0;  %用于记录当前找到的最好的随机初始模型的一致集中元素的个数  
    N = 1;            % 需要迭代的次数，这是从内点比例计算出来的，见教材6.1
    
    while N > trialcount
        %随机选取s个点来拟合模型，需要检查该随机数据集合是否为不能拟合出模型的退化集合
        degenerate = 1;
        count = 1;
        while degenerate
		    ind = randsample(npts, s); %从npts点中，随机选取s个，对于射影矩阵估计问题来说，s=4
            % 判断ind所索引的这4个点，是否是退化的，是否退化的判断函数为degenfn
            degenerate = feval(degenfn, x(:,ind));
            
            if ~degenerate %当前的4个点是非退化的，即可以拟合出一个模型来
                % 从这4个随机选择的点拟合出模型M
                M = feval(fittingfn, x(:,ind));
                % 如果拟合模型失败，也说明当前随机选取的点集为退化点集
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
        
        % 计算出当前拟合模型M的一致集inliers，当然inliers只是一些索引，用于表明数据x中符合当前M的内点的索引
        inliers = feval(distfn, M, x, t);
        ninliers = length(inliers); %当前模型一致集中元素的个数
        
        if ninliers > bestscore    % 该一致集元素个数大于已知最好的一致集元素个数，需要更新
            bestscore = ninliers;  
            bestinliers = inliers;
            bestM = M;
            
            % 根据当前模型的一致集的元素个数，可以动态调整所需最大迭代次数。需要注意到：当前一致集越大，实际上所需的外层迭代会越少
            fracinliers =  ninliers/npts;
            pNoOutliers = 1 -  fracinliers^s;
            pNoOutliers = max(eps, pNoOutliers);  % Avoid division by -Inf
            pNoOutliers = min(1-eps, pNoOutliers);% Avoid division by 0.
            N = log(1-p)/log(pNoOutliers);
        end
        
        trialcount = trialcount+1;

        % 迭代次数超够了最大限度，停止
        if trialcount > maxTrials
            break
        end     
    end
    
    if ~isnan(bestM)   % We got a solution 
        inliers = bestinliers;
    else           
        inliers = [];
        error('ransac was unable to find a useful solution');
    end
    