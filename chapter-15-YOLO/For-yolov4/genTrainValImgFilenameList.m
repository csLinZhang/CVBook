%该程序可从所有标注图像文件中，按照8：2的比例生成训练文件列表和验证文件列表

%mydata\img目录之下存放了标注好的图像文件，与每个图像文件对应的同名文本文件，是该图像文件的标注文件
%张林，同济大学
allFiles = dir('mydata\img\*.jpg');

trainImgpathlistfile = fopen('mydata\train.txt','wt');
valImgpathlistfile = fopen('mydata\val.txt','wt');

for index = 1:length(allFiles)
    currentFileName = allFiles(index).name;
    fullImageFileDir = ['mydata/img/' currentFileName];

    if mod(index, 10) ==9 || mod(index, 10) ==0
        fprintf(valImgpathlistfile,'%s\n', fullImageFileDir);
    else
        fprintf(trainImgpathlistfile,'%s\n', fullImageFileDir);
    end
end

fclose(trainImgpathlistfile);
fclose(valImgpathlistfile);