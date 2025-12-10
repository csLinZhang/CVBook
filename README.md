# 勘误表
本书在排版校对过程中引入了一些错误，在此表示歉意。请读者阅读[本书勘误表](./errata.pdf)修正。

# 教材信息
张林，赵生捷，《计算机视觉：原理算法与实践》，清华大学出版社，ISBN：978-7-302-68676-7，2025年5月
<img src="imgs\book_cover.jpg" alt="示例图片" width="40%">

# 本书特点
近年来，随着我国对人工智能领域人才培养支持力度的持续加大，越来越多的高校开设了计算机视觉课程。计算机视觉是一门综合性学科，其知识体系非常庞杂；同时，相较于计算机体系结构、数据结构、操作系统等计算机其他分支方向而言，它还是一门非常年轻的学科，其自身的学科内涵和基本研究方法目前仍处于快速完善和迭代阶段。这些现实情况给在大学讲授这门课的老师们提出了一个值得探讨的开放性问题：计算机视觉这门课程应该教什么？怎么教？

<img style="float: left;" src="imgs\problem_solver.png" alt="示例图片" width="60%">

作者认为，回答上述问题的关键在于要先理清我们要培养什么样的人才以及计算机视觉这门学科方向自身的特点。我们希望培养的毕业生不但要掌握前人已经积累好的知识技能，更要具有前瞻意识和创新思维，具备解决未来可能出现的新问题的能力。这就要求我们的教学工作不能只限于传授既有知识，而更要锻炼学生分析问题、逻辑推理、形成方案、迭代优化的综合能力，也就是常说的那句老话“既要授人以鱼，又要授人以渔”。另一方面，就计算机视觉这个学科方向而言，它的特点就是“很难学”。难就难在它对学习者在理论知识和实践技能两方面都有很高的要求。要想在这个领域入门，学习者既要具备综合应用微积分、线性代数、矩阵论、解析几何、射影几何、概率论等数学知识的能力，又要能较为熟练地掌握和运用各种编程工具、算法库和可视化库，比如C++、Python、Matlab、OpenCV、Eigen、Sophus、g2o、PyTorch、PCL、Pangolin等等。综合考虑这些因素，作者认为该课程的教学材料要尽可能地做到“问题与案例驱动，理论与实践并重”。

然而作者发现，目前很难找到满足上述需求的、适合于作为大学授课教材的计算机视觉书籍。根据作者的调研来看，此领域现有的书籍要么只讲算法与原理，但缺少实践指导，导致学生难以找到与书中理论和算法能够一一对应的程序实现以及如何应用这些算法的实操指导，不适合于引领学生入门；要么着重于介绍某些特定程序库的使用，讲解这些程序库接口的调用方式，但在讲解这些库中算法背后的数学理论与设计原理方面却浅尝辄止，容易使学习者成为“调包侠”。
作者在组织本书的材料时，力图能有效弥补现有计算机视觉书籍在作为教材方面的不足。该书在内容组织上遵循了“问题与案例驱动”的宏观原则。除第1章绪论外，全书内容按照图像的全景拼接、单目测量、目标检测和三维立体视觉四条技术主线来组织。根据作者的经验，这四条技术主线可以较为全面地覆盖计算机视觉领域比较成熟的知识点。对于每一条技术主线来说，其最终目标都是要解决一个明确的具体的问题。作者围绕如何解决这个具体问题，把相关的重要知识点循序渐进地、有机地组织在一起，并有意识地为读者建立起 "**数学→算法→技术→应用**" 这样一个支撑体系。作者多年的教学经验表明，这种形式的内容组织方式很容易为学习者所接受，使得初学者更容易从宏观上掌握该学科脉络并深刻理解每一个知识点的内涵和作用。

<img style="float: left;" src="imgs\logic_mansion.png" alt="示例图片" width="40%">

“理论与实践并重”是本书的一个显著特点。对每一个具体的模型或者算法，本书都尽可能详细地阐述清楚它的来龙去脉，给出必要的数学预备知识以及推导，帮助学习者构建起知识的“逻辑大厦”，努力让学习者知其然更知其所以然。另一方面，从很大程度上来说，计算机视觉是一门应用科学，学习者只有通过编程实现（以及必要的实际动手操作）才能深刻理解所学技术的本质。为此，配合理论教学内容，本书提供了丰富的示例程序和实践操作指导，帮助学习者消化理解相关模型、算法以及技术。拿单目测量这条技术主线来说，单目测量是一项技术，它能支撑的应用包括车载环视图中的平面目标检测、传送带上的扁平物品尺寸测量、路面目标测距等等；它所用到的算法包括相机内参平面标定算法、图像镜头畸变去除算法、平面间单应变换估计算法、鸟瞰视图生成算法等；为了掌握这些算法，读者需要了解的数学知识包括线性几何变换、平面射影几何、线性最小二乘问题、拉格朗日乘子法、旋转的轴角表达与罗德里格斯公式、非线性最小二乘问题、高斯牛顿法、列文伯格-马夸尔特法等等。在行文时，作者采用“倒叙”手法来讲述理论内容，先铺垫会用到的数学知识，再讲解相关算法，最后延伸到技术以及技术所支撑的应用。配合理论内容，作者在这一部分提供了Matlab版相机标定与图像去畸变示例代码、OpenCV版相机标定与图像去畸变代码、OpenCV相机标定核心源代码注释、鸟瞰视图视频合成代码。本书的Github代码仓库网址为https://github.com/csLinZhang/CVBook。

从2011年秋季开始，作者即在同济大学讲授计算机视觉课程。本书是作者在总结十余年教学实践经验的基础上形成的。作者于2022年2月便着手开始本书的撰写工作，直到2024年4月才完成了初稿，深感教材写作工作之不易！本书初稿部分内容在同济大学软件学院2022年春季学期、2023年春季学期、2023年秋季学期的计算机视觉课程中进行了试用。在此，对给本书初稿提出反馈意见的同学们表示感谢。

本书可作为人工智能、计算机和软件工程等专业高年级本科生或研究生计算机视觉课程的教材，也可供相关领域的工程技术人员参考。本书内容力求做到“自封闭”，读者只需具有高等数学、线性代数、概率论、解析几何和数字图像处理方面的基本知识即可，涉及到的稍复杂的数学预备知识、程序库的编译安装说明以及核心代码片段等都可在本书的附录中找到。

# 本书按四条技术主线组织材料

全景拼接技术主线知识点支撑体系

<img style="float: left;" src="imgs\theme1.png" alt="示例图片" width="80%">

相机标定与单目测量技术主线知识点支撑体系

<img style="float: left;" src="imgs\theme2.png" alt="示例图片" width="80%">

目标检测技术主线知识点支撑体系

<img style="float: left;" src="imgs\theme3-1.png" alt="示例图片" width="80%">
<img style="float: left;" src="imgs\theme3-2.png" alt="示例图片" width="80%">

三维立体视觉技术主线知识点支撑体系

<img style="float: left;" src="imgs\theme4.png" alt="示例图片" width="80%">

# 实践效果举例

全景拼接

<img style="float: left;" src="imgs\panorama.png" alt="示例图片" width="80%">

单目测量

<img style="float: left;" src="imgs\measurement-1.gif" alt="示例图片" width="60%">

<img style="float: left;" src="imgs\measurement-2.gif" alt="示例图片" width="60%">

目标检测

<img style="float: left;" src="imgs\obj-detect.gif" alt="示例图片" width="60%">

三维立体视觉

<img style="float: left;" src="imgs\3d-1.png" alt="示例图片" width="60%">

<img style="float: left;" src="imgs\3d-2.gif" alt="示例图片" width="60%">

<img style="float: left;" src="imgs\3d-3.gif" alt="示例图片" width="60%">

<img style="float: left;" src="imgs\3d-4.gif" alt="示例图片" width="60%">

<img style="float: left;" src="imgs\ilovetjcs.gif" alt="示例图片" width="60%">

# 授课视频
[00-本书编写指导思想和内容组织思路介绍](https://www.bilibili.com/video/BV12aVqzAELD/?spm_id_from=333.337.search-card.all.click&vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[01-计算机视觉介绍-（什么是CV，发展简史）](https://www.bilibili.com/video/BV18u55zCEof?spm_id_from=333.788.videopod.sections&vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[02-计算机视觉介绍-（发展简史，我们的工作）](https://www.bilibili.com/video/BV1Dt5VzrEto?spm_id_from=333.788.videopod.sections&vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[03-计算机视觉介绍-（我们的工作，课程主线安排）](https://www.bilibili.com/video/BV192EEzBEgU?spm_id_from=333.788.videopod.sections&vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[04-全景拼接主线介绍](https://www.bilibili.com/video/BV1bF7fzuEGD/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[05-线性几何变换-（旋转，欧氏，齐次坐标）](https://www.bilibili.com/video/BV1SbEBzQEhx/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[06-线性几何变换-（相似，仿射，射影）](https://www.bilibili.com/video/BV1W8ECz4Epw/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[07-线性几何变换-（变换群与几何学，基本几何不变量）](https://www.bilibili.com/video/BV1c7EyzsEm9?spm_id_from=333.788.recommend_more_video.0&vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[08-线性几何变换-（李群与李代数）](https://www.bilibili.com/video/BV1RZEfzpEHw/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[09-Harris特征-（角点检测基本思想）](https://www.bilibili.com/video/BV1uqJqzHEDm/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[10-Harris特征-（角点检测数学建模）](https://www.bilibili.com/video/BV1uqJqzHEt3/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[11-Harris特征-（角点检测程序实现流程）](https://www.bilibili.com/video/BV1pGJzzjEWK/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[12-Harris特征-（块描述子及其距离，特征点匹配）](https://www.bilibili.com/video/BV113j1z8ERG/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[13-Harris特征-（Harris角点匹配程序演示），SIFT特征](https://www.bilibili.com/video/BV1M6jXz9EiG/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[14-SIFT特征-（尺度不变点检测，方向归一化）](https://www.bilibili.com/video/BV1kBjYzWERZ/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[15-SIFT特征-（SIFT描述子，SIFT开源实现，基于LLM做个调包侠）](https://www.bilibili.com/video/BV1HzjyzpE8g/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[16-ORB特征-（FAST特征点，BRIEF描述子，海明距离）](https://www.bilibili.com/video/BV1t1T7zcEPJ/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[17-ORB特征-（ORB中的多尺度处理），矩阵微分](https://www.bilibili.com/video/BV1NwTszyEBg/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[18-线性最小二乘问题-（矩阵微分，拉格朗日乘子法，奇异值分解）](https://www.bilibili.com/video/BV1WwTszCEdo/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[19-线性最小二乘问题-（齐次线性最小二乘问题建模）](https://www.bilibili.com/video/BV1gWM4zcEYh/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[20-线性最小二乘问题-（齐次线性最小二乘问题求解）](https://www.bilibili.com/video/BV15LM4ziEu4/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[21-线性最小二乘问题-（非齐次线性最小二乘问题，凸函数二阶判定，穆尔-彭罗斯广义逆）](https://www.bilibili.com/video/BV1HgMYzPERi/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[22-单应矩阵的鲁棒估计-（随机采样一致集框架）](https://www.bilibili.com/video/BV183KbzFEDK/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[23-全景拼接实现-（图像插值，Matlab实现，Python实现）](https://www.bilibili.com/video/BV1zRK8zUE1C/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[24-单目测量主线介绍](https://www.bilibili.com/video/BV1jzgHzaEZq/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[25-射影几何初步-（矢量运算，射影平面）](https://www.bilibili.com/video/BV18m8uzTEUQ/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[26-射影几何初步-（射影平面，齐次坐标）](https://www.bilibili.com/video/BV1i98MzVEqU/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[27-射影几何初步-（射影平面上的点与线）](https://www.bilibili.com/video/BV1Ta8MzbEig/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[28-非线性最小二乘问题-（前置知识，非线性优化问题定义）](https://www.bilibili.com/video/BV1ia8tzFEwz/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[29-非线性最小二乘问题-（非线性优化问题，阻尼法）](https://www.bilibili.com/video/BV16n8tzqEMz/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[30-非线性最小二乘问题-（增益比，阻尼系数，雅可比矩阵）](https://www.bilibili.com/video/BV1uy8DznEQX/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[31-非线性最小二乘问题-（高斯牛顿法，列文伯格-马夸特法）](https://www.bilibili.com/video/BV1i789ziEGe/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[32-非线性最小二乘问题-（列文伯格马夸特法及其实现实例）](https://www.bilibili.com/video/BV1Xc8XztE4o/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[33-相机成像流程建模-（针孔相机模型）](https://www.bilibili.com/video/BV1a1YDz8Etd/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[34-相机成像流程建模-（针孔相机模型2）](https://www.bilibili.com/video/BV1iiYDziEV8/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[35-相机成像流程建模-（镜头畸变建模）](https://www.bilibili.com/video/BV172eJzpEYH/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[36-相机内参平面标定法-（问题的数学建模，旋转的轴角表达）](https://www.bilibili.com/video/BV1Ctvcz1EXt/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[37-相机内参平面标定法-（罗德里格斯公式，内参初始化）](https://www.bilibili.com/video/BV1CevAzJEAz/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[38-相机内参平面标定法-（消失点）](https://www.bilibili.com/video/BV1wse1zYEF8/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[39-相机内参平面标定法-（焦距初始化）](https://www.bilibili.com/video/BV1o3eyzpEXX/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[40-相机内参平面标定法-（外参初始化）](https://www.bilibili.com/video/BV1NwhXzDEXx/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[41-相机内参平面标定法-（雅可比矩阵元素表达式推导）](https://www.bilibili.com/video/BV1HYh9zEEBk/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[42-相机内参应用-（镜头畸变去除，鸟瞰视图）](https://www.bilibili.com/video/BV1AXhCzREMV/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[43-相机标定与单目测量实践](https://www.bilibili.com/video/BV168aGzAE45/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[44-目标检测主线介绍，凸集合](https://www.bilibili.com/video/BV1ftYNz5EL5/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[45-凸优化基础-（仿射集，仿射包，内部，闭包，边界，开集，相对内部，相对边界）](https://www.bilibili.com/video/BV1CxHUzLEip/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[46-凸优化基础-（仿射函数，凸函数，凹函数，凸函数的一阶判定条件）](https://www.bilibili.com/video/BV1s1HSzHE3d/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[47-凸优化基础-（凸函数的二阶判定，凸函数逐点求最大及非负组合）](https://www.bilibili.com/video/BV1oPH8zmEQb/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[48-凸优化基础-（优化问题，凸优化问题，凸二次规划问题）](https://www.bilibili.com/video/BV184HezEEY7/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[49-凸优化基础-（拉格朗日函数，拉格朗日对偶函数，对偶问题）](https://www.bilibili.com/video/BV1BjHBzrEdu/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[50-凸优化基础-（弱对偶性，强对偶性，斯莱特条件，对偶间隔）](https://www.bilibili.com/video/BV19vpNzVEmg/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[51-凸优化基础-（KKT条件）](https://www.bilibili.com/video/BV1YhnFz9ELS/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[52-凸优化基础-（问题举例，本章概念总结）](https://www.bilibili.com/video/BV1WwnTzoEgg/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[53-支持向量机-（线性可分，超平面，感知机）](https://www.bilibili.com/video/BV1J1nxzgE5W/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[54-支持向量机-（感知机学习，分隔超平面，几何间隔）](https://www.bilibili.com/video/BV1CdnszCE7T/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[55-支持向量机-（硬间隔SVM及其凸二次规划问题形式）](https://www.bilibili.com/video/BV1dYn4zTE4M/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[56-支持向量机-（硬间隔SVM的对偶求解）](https://www.bilibili.com/video/BV12BnXzfEmK/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[57-支持向量机-（软间隔SVM的问题建模）](https://www.bilibili.com/video/BV1ohnBzMEth/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[58-支持向量机-（软间隔SVM求解，非线性SVM问题引入）](https://www.bilibili.com/video/BV1hE4QzjE7w/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[59-支持向量机-（核函数，核化SVM，多类分类SVM）](https://www.bilibili.com/video/BV1SKWiztE2w/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[60-支持向量机-（SVM+HOG目标检测，YOLO概况）](https://www.bilibili.com/video/BV1DusczaEPe/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[61-YOLO-（目标检测标注真值理解，YOLOv1详解）](https://www.bilibili.com/video/BV11wWDz4E3M/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[62-YOLO-（YOLOv3详解）](https://www.bilibili.com/video/BV12ZWSzFEzE/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[63-YOLO-（YOLOv8，Darknet+YOLOV4实践，Ultralytics+YOLOV11实践）](https://www.bilibili.com/video/BV1UxsEzyEXi/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[64-三维视觉主线介绍 双目视觉系统构建流程](https://www.bilibili.com/video/BV1PgS6BkEPy/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[65-双目视觉系统-（校正化双目系统，双目标定问题建模）](https://www.bilibili.com/video/BV1rfSkB2ERg/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[66-双目视觉系统-（双目系统标定）](https://www.bilibili.com/video/BV1CJSYBpEaM/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[67-双目视觉系统-（校正化双目系统构造）](https://www.bilibili.com/video/BV1pJ2vBJExB/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[68-双目视觉系统-（运行时校正化双目图像对的获取，立体匹配，基于视差的空间点三维坐标计算）](https://www.bilibili.com/video/BV1Ni2fBdEps/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)

[69-双目视觉系统-（三维点云重建，双目视觉系统构造实践）](https://www.bilibili.com/video/BV16RmxBtEf6/?vd_source=5cecc28f8471406b6fec5f23dc6619fd)
