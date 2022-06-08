%执行描述子的匹配
%descriptors1In，descriptors2In，是描述子集合
%matchThreshold, 描述子匹配距离阈值，如果两个描述子之间的距离大于这个阈值，则认为它们不匹配
%maxRatioThreshold,匹配无歧义确认阈值，d1与d2最匹配的,d1与d3是次匹配的，只有当dist(d1,d2)/dist(d1,d3)小于maxRatioThreshold时，
%才认为d1与d2匹配描述子对
function indexPairs = matchDescriptors(descriptors1In, descriptors2In, matchThreshold, maxRatioThreshold)

descriptorNums1 = size(descriptors1In,2);
descriptorNums2 = size(descriptors2In,2);

%初始化好描述子集合比对距离矩阵
scores = zeros(descriptorNums1, descriptorNums2);
%在这个循环中，每次从第一个描述子集合中取出一个描述子currentDescriptor，计算它与descriptors2In中每个描述子的SSD距离
for descriptorIndex = 1: descriptorNums1
    %获取到features1中当前一个描述子currentDescriptor
    currentDescriptor = descriptors1In(:, descriptorIndex);
    tmpDescriptorMat = repmat(currentDescriptor,[1,descriptorNums2]);
    distsCurrentDescriptor2features2 = sum((tmpDescriptorMat - descriptors2In).^2);
    scores(descriptorIndex,:) = distsCurrentDescriptor2features2;
end
%对于每一个特征点，都留下来两个与它最接近的，目的是为了测试歧义性，只有当最匹配的距离与次匹配距离的比例小很多的时候，才接受
%partialSort这个函数，返回scores这个矩阵中，每一行的最大两个值以及它们所在的位置，并且进行了转置
[matchMetric, topTwoIndices] = vision.internal.partialSort(scores, 2, 'ascend');

%indexPairs是一个2*descriptorNums1，第一行是1~descriptorNums1数字，第二行表示与第一行的描述子对应的descriptors2In描述子集合中的索引
indexPairs = vertcat(uint32(1:size(scores,1)), topTwoIndices(1,:));

inds = matchMetric(1,:) <= matchThreshold; %如果描述子之间的距离大于matchThreshold，则不保留
indexPairs = indexPairs(:, inds);
matchMetric = matchMetric(:, inds);

%%%%%%%%%%%%%%%%%
%这一段对应到教材中的匹配无歧义准则，也就是最匹配的距离与次匹配距离之比一定要小于某个阈值才可以
topTwoScores = matchMetric;
zeroInds = topTwoScores(2, :) < cast(1e-6, 'like', topTwoScores);
topTwoScores(:, zeroInds) = 1;
ratios = topTwoScores(1, :) ./ topTwoScores(2, :);
unambiguousIndices = ratios <= maxRatioThreshold;
indexPairs  = indexPairs(:, unambiguousIndices);
%%%%%%%%%%%%%%%%%%

%下面这部分是双向确认原则，要看看descriptors2In中的某个描述子，它"最喜欢"的descriptors1In的描述子是不是也是最喜欢它的
[~, idx] = min(scores(:,indexPairs(2,:)));
uniqueIndices = idx == indexPairs(1,:);

indexPairs  = indexPairs(:, uniqueIndices);
indexPairs = indexPairs';





