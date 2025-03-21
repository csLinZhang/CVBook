张林，赵生捷，《计算机视觉：原理算法与实践》

计算机视觉是一门研究如何构建具有“视觉”功能的计算机系统的学科，是人工智能研究领域的一个重要分支。从刷脸支付到太空探索，从智能监控到视觉导航，计算机视觉技术正在越来越多的应用领域中影响和改变着人类的生产和生活方式。
近年来，随着我国对人工智能领域人才培养支持力度的持续加大，越来越多的高校开设了计算机视觉课程。计算机视觉是一门综合性学科，其知识体系非常庞杂；同时，相较于计算机体系结构、数据结构、操作系统等计算机其他分支方向而言，它还是一门非常年轻的学科，其自身的学科内涵和基本研究方法目前仍处于快速完善和迭代阶段。这些现实情况给在大学讲授这门课的老师们提出了一个值得探讨的开放性问题：计算机视觉这门课程应该教什么？怎么教？

作者认为，回答上述问题的关键在于要先理清我们要培养什么样的人才以及计算机视觉这门学科方向自身的特点。我们希望培养的毕业生不但要掌握前人已经积累好的知识技能，更要具有前瞻意识和创新思维，具备解决未来可能出现的新问题的能力。这就要求我们的教学工作不能只限于传授既有知识，而更要锻炼学生分析问题、逻辑推理、形成方案、迭代优化的综合能力，也就是常说的那句老话“既要授人以鱼，又要授人以渔”。另一方面，就计算机视觉这个学科方向而言，它的特点就是“很难学”。难就难在它对学习者在理论知识和实践技能两方面都有很高的要求。要想在这个领域入门，学习者既要具备综合应用微积分、线性代数、矩阵论、解析几何、射影几何、概率论等数学知识的能力，又要能较为熟练地掌握和运用各种编程工具、算法库和可视化库，比如C++、Python、Matlab、OpenCV、Eigen、Sophus、g2o、PyTorch、PCL、Pangolin等等。综合考虑这些因素，作者认为该课程的教学材料要尽可能地做到“问题与案例驱动，理论与实践并重”。

然而作者发现，目前很难找到满足上述需求的、适合于作为大学授课教材的计算机视觉书籍。根据作者的调研来看，此领域现有的书籍要么只讲算法与原理，但缺少实践指导，导致学生难以找到与书中理论和算法能够一一对应的程序实现以及如何应用这些算法的实操指导，不适合于引领学生入门；要么着重于介绍某些特定程序库的使用，讲解这些程序库接口的调用方式，但在讲解这些库中算法背后的数学理论与设计原理方面却浅尝辄止，容易使学习者成为“调包侠”。
作者在组织本书的材料时，力图能有效弥补现有计算机视觉书籍在作为教材方面的不足。该书在内容组织上遵循了“问题与案例驱动”的宏观原则。除第1章绪论外，全书内容按照图像的全景拼接、单目测量、目标检测和三维立体视觉四条技术主线来组织。根据作者的经验，这四条技术主线可以较为全面地覆盖计算机视觉领域比较成熟的知识点。对于每一条技术主线来说，其最终目标都是要解决一个明确的具体的问题。作者围绕如何解决这个具体问题，把相关的重要知识点循序渐进地、有机地组织在一起，并有意识地为读者建立起“数学→算法→技术→应用”这样一个支撑体系。作者多年的教学经验表明，这种形式的内容组织方式很容易为学习者所接受，使得初学者更容易从宏观上掌握该学科脉络并深刻理解每一个知识点的内涵和作用。

“理论与实践并重”是本书的一个显著特点。对每一个具体的模型或者算法，本书都尽可能详细地阐述清楚它的来龙去脉，给出必要的数学预备知识以及推导，帮助学习者构建起知识的“逻辑大厦”，努力让学习者知其然更知其所以然。另一方面，从很大程度上来说，计算机视觉是一门应用科学，学习者只有通过编程实现（以及必要的实际动手操作）才能深刻理解所学技术的本质。为此，配合理论教学内容，本书提供了丰富的示例程序和实践操作指导，帮助学习者消化理解相关模型、算法以及技术。拿单目测量这条技术主线来说，单目测量是一项技术，它能支撑的应用包括车载环视图中的平面目标检测、传送带上的扁平物品尺寸测量、路面目标测距等等；它所用到的算法包括相机内参平面标定算法、图像镜头畸变去除算法、平面间单应变换估计算法、鸟瞰视图生成算法等；为了掌握这些算法，读者需要了解的数学知识包括线性几何变换、平面射影几何、线性最小二乘问题、拉格朗日乘子法、旋转的轴角表达与罗德里格斯公式、非线性最小二乘问题、高斯牛顿法、列文伯格-马夸尔特法等等。在行文时，作者采用“倒叙”手法来讲述理论内容，先铺垫会用到的数学知识，再讲解相关算法，最后延伸到技术以及技术所支撑的应用。配合理论内容，作者在这一部分提供了Matlab版相机标定与图像去畸变示例代码、OpenCV版相机标定与图像去畸变代码、OpenCV相机标定核心源代码注释、鸟瞰视图视频合成代码。本书的Github代码仓库网址为https://github.com/csLinZhang/CVBook。

从2011年秋季开始，作者即在同济大学讲授计算机视觉课程。本书是作者在总结十余年教学实践经验的基础上形成的。作者于2022年2月便着手开始本书的撰写工作，直到2024年4月才完成了初稿，深感教材写作工作之不易！本书的第13、14章由赵生捷撰写，其余部分均由张林撰写，全书的统稿工作也由张林完成。本书初稿部分内容在同济大学软件学院2022年春季学期、2023年春季学期、2023年秋季学期的计算机视觉课程中进行了试用。在此，对给本书初稿提出反馈意见的同学们表示感谢。

本书可作为人工智能、计算机和软件工程等专业高年级本科生或研究生计算机视觉课程的教材，也可供相关领域的工程技术人员参考。本书内容力求做到“自封闭”，读者只需具有高等数学、线性代数、概率论、解析几何和数字图像处理方面的基本知识即可，涉及到的稍复杂的数学预备知识、程序库的编译安装说明以及核心代码片段等都可在本书的附录中找到。

计算机视觉学科仍处于蓬勃发展阶段，新理论、新算法、新技术层出不穷，加之作者水平有限，书中难免存在缺陷和不足，殷切希望广大读者批评指正。

作者

2024年4月
