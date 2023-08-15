![](images/1.jpg)

**Lianfa Li, 2022, Deep Learning: Principles and Geoscience Analysis of Remote Sensing (in Chinese), Science Press(https://zhuanlan.zhihu.com/p/591913706)**

<img src="https://render.githubusercontent.com/render/math?math=x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2b}">
<img src="https://render.githubusercontent.com/render/math?math=x = -1 - &mu; "> 
<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1"> 
<img src="https://render.githubusercontent.com/render/math?math={\sum_{d=0}^{d_{max}}}">
<img src="https://render.githubusercontent.com/render/math?math=x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2b}"> 
<img src="https://render.githubusercontent.com/render/math?math=\begin{equation}\sum_{n=0}^\infty\frac{1}{2^n}\end{equation}">

$$
E[x]=1\cdot \mu +0\cdot (1-\mu )=\mu 
<img/>
$$

$$ 1-\mu <img/> $$
you are $$ 
Var[x]=(1 - \mu ) 
<img/> 
$$


### **问题及答案（Problems and answers）**
**问题1.1** 如何证明伯努利分布的均值及变差分别为式（1.11）与式（1.12）？<br>**答**：伯努利试验有两个可能的结果，即1和0，前者发生的概率为$μ$，后者的概率为$1-\mu$。则试验的期望值为$E[x]=1\cdot \mu +0\cdot (1-\mu )=\mu$ ，该试验的方差则为$Var[x]=(1-\mu )^{2}\cdot \mu + (0-\mu )^{2}\cdot (1-\mu )=\mu (1-\mu )$。<br>

**问题1.2**  确定一个$d $ 维高斯分布的参数有多少个？是否根据一些特征减少其参数个数？<br>**答**：（1）确定一个$d$维高斯分布的参数有2个：均值$μ$和多维随机变量的协方差矩阵$Σ$。<br>（2）若各维度独立同分布，则协方差矩阵是单位向量，即$Σ=I$。<br>

**问题1.3**   试证明Box-Muller算法生成的输出符合标准高斯分布。<br>**答**：证明（1）假设变量x和y是[0,1]均匀分布的随机数，且$x$和$y$彼此相互独立，则构造随机数$g=\sqrt[2]{-2\log{x}}\cos(2y\pi)$和$h=\sqrt[2]{-2\log{x}}\sin(2y\pi)$；<br>（2）易知，$g^2+h^2=-2\log{x}$为符合指数分布的随机数；<br>（3）由定理“两个相互独立的高斯分布随机数的平方和服从指数分布”可知，$g$和$h$为两个相互独立且正交的、均值为0、方差为1的高斯分布随机数，即符合标准高斯分布。<br>

**问题3.1**   令$X_i （i=1,2,…,n）$独立同分布（均值：$μ$；变差：$σ^2$），证明$Y_{n}=\frac{\sum_{i=1}^{n} X_{i}}{n} \sim N(\mu ,\frac{\sigma ^{2}}{n})$。<br>**答**（1）由独立同分布的中心极限定理得到，$Z_{n}=\frac{\sum_{i=1}^{n} -n\mu }{\sigma \sqrt{n} }f\sim N(0,1)$。<br>（2）则$\sum_{i=1}^{n} X_{i}=\sigma \sqrt{n}Z_{n}+n\mu$近似地服从正态分布$N(n\mu ,n\sigma ^{2})$。<br>（3）由此得出，$E(\frac{\sum_{i=1}^{n} X_{i}}{n})=\frac{E(\sum_{i=1}^{n} X_{i})}{n}=\mu$，$D（\frac{\sum_{i=1}^{n} X_{i}}{n}）=D(E(\sum_{i=1}^{n} X_{i})/n^2 =σ^2/n$，即$Y_{n}=\frac{\sum_{i=1}^{n} X_{i}}{n} \sim N(\mu ,\frac{\sigma ^{2}}{n})$。<br>

**问题3.2**  在重要性抽样中，为什么采用估计变差最小作为选择代理概率函数$q(x)$的标准？<br>**答**：重要性采样改变原来的旧分布，用新的分布去采集样本然后求出目标期望，但前提条件是两个分布不能相差太大，因此采用估计变差最小作为选择$q(x)$的标准。<br>

**问题3.3**  设置不同的初始参数，验证一下上面实例中最后的结果是否稳定？<br>**答**：假设初始状态为（0.25，0.25，0.25，0.25）：<br>（1）第一次状态转移结果为$P(1)=\begin{bmatrix}0.25&0.25&0.25&0.25\end{bmatrix}\begin{bmatrix}0.2&0.8&0&0\\0&0&1&0\\0.6&0&0&0.4\\0.9&0.1&0&0\end{bmatrix}=\begin{bmatrix}0.43&0.23&0.25&0.10\end{bmatrix}$；<br>（2）第二次状态转移结果为$P(2)=P(1)\begin{bmatrix}0.2&0.8&0&0\\0&0&1&0\\0.6&0&0&0.4\\0.9&0.1&0&0\end{bmatrix}=\begin{bmatrix}0.43&0.23&0.25&0.10\end{bmatrix}\begin{bmatrix}0.2&0.8&0&0\\0&0&1&0\\0.6&0&0&0.4\\0.9&0.1&0&0\end{bmatrix}=\begin{bmatrix}0.33&0.35&0.23&0.10\end{bmatrix}$；<br>（3）多次迭代之后，即$P(n)=P(2)\begin{bmatrix}0.2&0.8&0&0\\0&0&1&0\\0.6&0&0&0.4\\0.9&0.1&0&0\end{bmatrix} \begin{bmatrix}0.2&0.8&0&0\\0&0&1&0\\0.6&0&0&0.4\\0.9&0.1&0&0\end{bmatrix}\dots \begin{bmatrix}0.2&0.8&0&0\\0&0&1&0\\0.6&0&0&0.4\\0.9&0.1&0&0\end{bmatrix}$，结果收敛到（0.33，0.28，0.28，0.11）；<br>（4）综上所述，上面实例中最后的结果是稳定的。<br>

**问题3.4**  采用特征值及特征矢量的方法求上面实例中的稳定解。<br>**答**：（1）平稳分布满足$Pπ=π$，与特征方程$Ax=λx$联系可知，平稳分布$π$就是转移矩阵$P$的特征值为1对应的特征向量（归一化），因此直接对转移矩阵进行特征分解来求得平稳分布。<br>（2）对矩阵$P$进行特征值分解：$P=Q\Lambda Q^{-1}$，对角矩阵$Λ$的对角线上的元素即为$P$的特征值结果：$\Lambda =\begin{pmatrix}\lambda _{1} &0&\dots &0\\0&\lambda _{2} &\dots &0\\\vdots \\0&0&\dots &\lambda _{k} \end{pmatrix}$，$\lambda _{i} \le 1$;$i=1,2,\dots ,k$。<br>（3）将$P$的分解结果代入$\pi _{t+1}= \pi _{t}P^{t} =\pi _{t}(Q\Lambda Q^{-1})^{t}=\pi _{t}(Q\Lambda Q^{-1}Q\Lambda Q^{-1}\dots Q\Lambda Q^{-1})$，由于$Q^{-1}Q=E$，即单位矩阵，因此$\pi _{t+1}=\pi _{t}Q\begin{pmatrix}\lambda _{1} &0&\dots &0\\0&\lambda _{2} &\dots &0\\\vdots \\0&0&\dots &\lambda _{k} \end{pmatrix}Q^{-1}$。<br>（4）转移次数足够多的时候，即$t$足够大的时候：$\lim_{t \to \infty}\lambda _{i}^{t}=0(i=1,2,\dots ,k)$。<br>（5）假设转移次数分别执行了$m$次和$m+1$次，它们对应随机变量的概率分布表示如下：$\pi _{m+1}=\pi _{m}Q\Lambda ^{m}Q^{-1}$，$\pi _{m+2}=\pi _{m+1}Q\Lambda ^{m+1}Q^{-1}$，而此时的对角阵$\Lambda ^{m}$，$\Lambda ^{m+1}$中的各元素均无限趋近于0（特征值为1的结果除外）。<br>（6）因此，在$m$足够大时：$\Lambda ^{m}=\Lambda ^{m+1}$，从而$\pi _{m+1}=\pi _{m+2}$。<br>（7）基于以上推导过程，转移次数$t>m$时：$\pi _{m+1}=\pi _{m+2}=⋯=\pi _{end}$，$\pi _{end}$即为最终的平稳状态。<br>

**问题3.5**  推导证明上式成立。<br>**答**：（1）由细致平稳条件$\pi (i)P_{ij}=\pi (j)P_{ji}$可得，$\sum_{i=1}^{\infty }\pi (i)P_{ij}=\sum_{i=1}^{\infty }\pi (j)P_{ji}$。<br>（2）取$j\to \infty$，此时$π(j)$为稳定值，则$\sum_{i=1}^{\infty }\pi (j)P_{ji}=\pi (j)\sum_{i=1}^{\infty }P_{ji}$。<br>（3）多次迭代后处于平稳状态，此时$\sum_{i=1}^{\infty }\pi (i)P_{ij}=1$，则$\pi (j)\sum_{i=1}^{\infty }P_{ij}=\pi (j)$，上式成立。<br>

**问题3.6**  对于连续数据而言，在确定平稳分布恰好是$p(x)$，没有先验知识的情况下如何设定状态转移概率？解释一下$α(x_t,z)$引入的物理意义。<br>**答**：接收率$α(x_t,z)$，其物理意义可以理解为在原来的马尔可夫链基础上，将状态$i$以$q(i,j)$概率转移到状态$j$时，通过以$α(i,j)$的概率接受这个转移，从而得到新的马尔可夫链的转移概率为$q(i,j)α(i,j)$，使得其符合细致平稳条件。<br>

**问题3.7**  证明式(3.11)满足细致平稳条件，从而抽取的样本最终符合目标概率分布p(x)。<br>**答**：$p(i)q(i,j)\alpha (i,j)=p(i)Q(i,j)\cdot \min \left \{ \frac{p(j)q(j,i)}{p(i)q(i,j)} ,1 \right \}=\min \left \{ p(i)q(i,j),p(j)q(j,i) \right \}=p(j)q(j,i)\cdot \min \left \{ \frac{p(i),q(i,j)}{p(j)q(j,i)},1  \right \}=p(j)q(j,i)\alpha (j,i))$。<br>

**问题3.8**  证明上式的概率为1。<br>**答**：$\alpha (x^{(i)},x^{\ast })=\min \left \{ \frac{p(x^{\ast })q(x^{(i)}\mid x^{\ast })}{p(x^{(i)})q(x^{\ast }\mid x^{(i)})} ,1 \right \}=\min \left \{ \frac{p(x^{\ast })q(x_{j}^{(i)} \mid x_{-j}^{i})}{p(x^{(i)})q(x_{j}^{\ast }\mid x_{-j}^{(i)})} ,1 \right \}=\min \left \{ 1,\frac{p(x_{-j}^{\ast })}{p(x_{-j}^{x} )}  \right \}=1$。<br>

**问题3.9**  如何抽样$x_{1}^{(t+1)}\sim p(x_{1}\mid x_{2}^{(t)},x_{3}^{(t)},\dots ,x_{n}^{(t)})$？<br>**答**：高维Gibbs采样的过程为依次轮换坐标轴进行采样，$x_{2}^{(t)},x_{3}^{(t)},\dots ,x_{n}^{(t)})$即为更新样本点$x_{1}$维度的数据：固定$x_{1}$后进行采样，得到$x_{1}$维度新的值x_{1}^{(t+1)}。<br>

**问题3.10**  令A，B，C代表3枚硬币，投掷这些硬币正面出现的概率分别是$π$，$p$和$q$。反复进行以下操作：掷A，根据其结果选出硬币B或C，正面选B，反面选C；然后投掷选中硬币，正面记作1，反面记作0。独立地重复n次（n=10)，结果为1111110000。只能观察投掷硬币的结果，而不知其过程，估计这三个参数$π$，$p$和$q$ (Vapnik，1998)。尝试一下如何采用Gibbs抽样获得估计结果。<br>**答**：（1）记最终的观测结果为$\overrightarrow{x}=\left \{ x_1,x_2,\dots ,x_n \right \}$，每次观测结果使用的硬币为$\overrightarrow{z}=\left \{ z_1,z_2,\dots ,z_n \right \}$，对于硬币$k$出现正反面符合伯努利分布，记作$P(x\mid p_{k})=p_{k}^{x}(1-p_{k})^{1-x}$，其中随机变量的$x$的取值为0、1。<br>![](images/2.jpg#pic_center)<br>（2）这个试验主要分为两个过程。一是$\overrightarrow{\alpha }\longrightarrow \theta \longrightarrow z_m$：投掷硬币A，生成观测结果中第$m$次观测结果所使用的硬币编号$k$；二是$\overrightarrow{\beta }\longrightarrow \phi _{k}  \longrightarrow x_m\mid z_m=k$：生成第$m$次观测结果时，选择编号为$k$的硬币，投掷这枚硬币，生成观测结果$x_m$。<br>（3）第一个过程是二项分布，有$P(\overrightarrow{z}\mid \theta )=\theta _{k}(1-\theta )^{n-k}$，$k$为硬币A正面朝上的次数。<br>（4）因为$P(\overrightarrow{z}\mid \theta )\sim B(n,p_A)$，可以取参数$\theta \sim Beta(\theta ,\overrightarrow{\alpha })$，组成伯努利-Beta共轭分布，则后验分布为：$P(\theta \mid \overrightarrow{z})\sim Beta(\theta \mid \alpha _1+k,\alpha _2+n-k)$，$P(\overrightarrow{z}\mid \overrightarrow{\alpha })=\int P(\overrightarrow{z}\mid \theta )P(\theta \mid \overrightarrow{\alpha })d\theta =\frac{B(\alpha _1+k,\alpha _2+n-k)}{B(\alpha _1,\alpha _2)}$。<br>（5）我们取参数θ的值为在后验分布下的期望，则$\theta =E(Beta(\theta \mid \alpha _1+k,\alpha _2+n-l))=\frac{\alpha _1+k}{(\alpha _1+k)(\alpha _2+n-k)}$。<br>（6）$n_1$和$n_2$分别表示使用硬币B和硬币C的次数，如果已知每次观测结果来自哪枚硬币，任何两次观测结果都是可交换的，将来自同一枚硬币的观测结果放在一起：$\overrightarrow{x}=(\overrightarrow{x}_B,\overrightarrow{x}_C)$，$\overrightarrow{z}=(\overrightarrow{z}_B,\overrightarrow{z}_C)$。<br>（7）同上可知，对于来自硬币$k$的观测结果，$P(\overrightarrow{x}_k\mid \phi _{k})\sim B(n_k,\phi _{k})$，参数$\phi _{k}\sim Beta(\phi _{k}\mid \overrightarrow{\beta }_k)$，组成二项-Beta共轭分布，则后验分布：$P(\phi _{k}\mid \overrightarrow{x}_k)\sim Beta(\phi _{k}\mid \beta _{k,1}+n_{k,1},\phi _{k}\mid \beta _{k,2}+n_{k,2})$，$P(\overrightarrow{x}_k\mid \overrightarrow{z}_k,\overrightarrow{\beta }_k)=\frac{B(\beta _{k,1}+n_{k,1},\beta _{k,2}+n_{k,2})}{B(\beta _{k,1},\beta _{k,2})} k\in \left \{ B,C \right \}$，$n_{k,1}$和$n_{k,2}$分别是$k$硬币出现正反面的次数。<br>（8）参数的值为$\phi _{k}=\frac{\beta _{k,1}+n_{k,1}}{(\beta _{k,1}+n_{k,1})+(\beta _{k,2}+n_{k,2})}$，因此，$P(\overrightarrow{x}\mid \overrightarrow{z},\overrightarrow{\beta })=P(\overrightarrow{x}_B\mid \overrightarrow{z}_B,\overrightarrow{\beta }_B)=P(\overrightarrow{x}_C\mid \overrightarrow{z}_C,\overrightarrow{\beta }_C)=\frac{B(\beta _{B,1}+k_{B},\beta _{B,2}+n_B-k_{B})}{B(\beta _{B,1},\beta _{B,2})}\frac{B(\beta _{C,1}+k_{C},\beta _{C,2}+n_C-k_{C})}{B(\beta _{C,1},\beta _{C,2})}$，综上，可以得到联合分布$p(\overrightarrow{x},\overrightarrow{z}\mid \overrightarrow{\alpha },\overrightarrow{\beta })=p(\overrightarrow{z}\mid \overrightarrow{\alpha })p(\overrightarrow{x}\mid \overrightarrow{z},\overrightarrow{\beta })=\frac{B(\alpha _1+k,\alpha _2+n-k)}{B(\alpha _1,\alpha _2)}\frac{B(\beta _{B,1}+k_{B},\beta _{B,2}+n_B-k_{B})}{B(\beta _{B,1},\beta _{B,2})}\frac{B(\beta _{C,1}+k_{C},\beta _{C,2}+n_C-k_{C})}{B(\beta _{C,1}, \beta _{C,2})}$。<br>（9）由于$\overrightarrow{x}$是观测到的已知变量，只有$\overrightarrow{z}$是隐含的变量，所以真正需要采样的是条件分布$p(\overrightarrow{z}\mid \overrightarrow{x})$。根据Gibbs采样算法的要求，需求得任意一个坐标轴$i$对应的条件分布$p(z_i=k\mid \overrightarrow{z_{\neg i}},\overrightarrow{x})$。<br>（10）假设已经观测到$x_i=t$，根据贝叶斯公式可以得到$p(z_i=k\mid \overrightarrow{z_{\neg i}},\overrightarrow{x})=p(z_i=k\mid x_i=t,\overrightarrow{z_{\neg i}},\overrightarrow{x_{\neg i}})=\frac{p(z_i=k,x_i=t\mid \overrightarrow{z_{\neg i}},\overrightarrow{x_{\neg i}})}{p(x_i=t\mid \overrightarrow{z_{\neg i}},\overrightarrow{x_{\neg i}})}\propto p(z_i=k,x_i=t\mid \overrightarrow{z_{\neg i}},\overrightarrow{x_{\neg i}})$。去掉第$i$次观测值并不影响其他共轭结构，其他共轭结构与$z_i=k$，$x_i=t$是相互独立的，因此$p(z_i=k\mid \overrightarrow{z_{\neg i}},\overrightarrow{x})\propto p(z_i=k,x_i=t\mid \overrightarrow{z_{\neg i}},\overrightarrow{x_{\neg i}})=p(z_i=k,x_i=t\mid \overrightarrow{z_{k,\neg i}},\overrightarrow{x_{k,\neg i}},\overrightarrow{z_{\neg k}},\overrightarrow{x_{\neg k}})=p(z_i=k,x_i=t\mid \overrightarrow{z_{k,\neg i}},\overrightarrow{x_{k,\neg i}})$，$\overrightarrow{x_{k,\neg i}}$表示去除第$i$次观测所属k硬币的观测值。<br>（11）因此，得到Gibbs采样公式：$P(z_i=k\mid \overrightarrow{z_{\neg i}},\overrightarrow{x})\propto p(z_i=k,x_i=t\mid \overrightarrow{z_{k,\neg i}},\overrightarrow{x_{k,\neg i}})=\int P(z_i=k,x_i=t,\theta ,\phi _{k}\mid \overrightarrow{z_{k,\neg i}},\overrightarrow{x_{k,\neg i}})d\theta d\phi _{k}=E(\theta )E(\phi _{k})=\frac{\alpha _{1}+n_{(k,\neg i),1}}{(\alpha _{1}+n_{(k,\neg i),1})+(\alpha _{2}+n_{(k,\neg i),2})}\frac{\beta _{k,1}+n_{(k,\neg i),1}}{(\beta _{k,1}+n_{(k,\neg i),1})+(\beta _{k,2}+n_{(k,\neg i),2})}$。<br>（12）利用python实现Gibbs采样过程，得到最终的参数估计结果：$π$=0.75，$p$=0.7，$q$=0.75。<br>

**问题3.11**  参考破解凯撒密码的论文，体会MCMC的应用(Diaconis，2008)。<br>**答**：（1）化学和物理方面：从点阵规范理论到硬盘，MCMC计算是化学和物理的支柱；可以利用MCMC算法计算普通液体的性质，计算结果几乎和计算稀气体和谐波固体的性质一样确定。<br>（2）生物学方面：基于MCMC算法研究某地区赤杨、铁山纯林和混交林密度对生长和优势度的影响等。<br>（3）统计学方面：基于MCMC算法，针对多种超市商品日销数额的多步预测，预测个体顾客交易，并预测每笔交易的商品数量等；将MCMC算法应用于粒子滤波器领域及工程应用中。<br>（4）理论研究方面：许多问题，如计算一个矩阵的恒量或一个凸多面体的体积的精确答案可能需要指数级的时间，但是只要能找到一个快速混合马尔可夫链来随机生成问题实例，就可以在多项式次数的操作中找到可证明的精确近似。<br>

**问题4.1**  根据推导的欧拉方程，采用变分法求目标函数方程式4.1、4.2及4.3的最优解。<br>**答**：（1）求解式4.1的最优解：<br>
最速降线问题满足边界条件$y(0)=0$，$y(x)=y$。$F(x,y,y^{,})=F(y,y^{,})=\sqrt{\frac{1+y^{,2} }{2gy}}$，其欧拉方程为$\frac{\partial F}{\partial Y}-\frac{d}{dx}(\frac{\partial F}{\partial y^{,}} )=0$。<br>
由于$\frac{d}{dx}[F-y^{,}\frac{\partial F}{\partial y^{,}}]=y^{,}\frac{\partial F}{\partial y}+y^{,,}\frac{\partial F}{\partial y^{,}}-y^{,,}\frac{\partial F}{\partial y^{,}}-y^{,}\frac{d}{dx}(\frac{\partial F}{\partial y^{,}} )=0$，所以$F-y^{,}\frac{\partial F}{\partial y^{,}}=C$，可得出$y=2r\sin^{2} \frac{\theta }{2} =r(1-\cos \theta )$。上式对$θ$求导，得到$x=r(\theta -\sin \theta )+x_{0}$。<br>
根据曲线过原点（0，0）及（x，y）可求出$x_0=0$及$r$，由此得出所求曲线为$\begin{cases}x=r(\theta -\sin \theta )\\y=r(1-\cos \theta )\end{cases}$。<br>
	（2）求解式4.2的最优解：<br>
	根据欧拉-卜阿松方程可得到最小旋转曲面问题的欧拉方程：$(1+y^{,2} )^{\frac{1}{2} }-\frac{y^{,2}}{(1+y^{,2} )^{\frac{1}{2} }}-\frac{y\cdot y^{,,}}{(1+y^{,2} )^{\frac{1}{2} }}=0$，化简后得到：$1+y^{,2}=y\cdot y^{,,}$。<br>
	令$y^{,}=\sin ht$，则有$y^{,,}=\cos ht$做代换，将$y^,$和$y^{,,}$代入到化简后的等式中，得到$y=cos⁡ht$。<br>
同时，$dx=\frac{dy}{y^{,}}=\frac{\sin htdt}{\sin ht} =dtdt$，两边积分得：$x=t+E$。<br>
我们所求的曲线就是由$y=\cos ht$和$x=t+E$所形成的曲线，即为余弦正切曲线，一般形式为：$y=A\cos h(\frac{x-B}{A} )$。<br>
	（3）求解式4.3的最优解：<br>
	取$F(x,y,y^, )=F(y,y^, )=y\sqrt{1+y^{,2} }$，其欧拉方程为$\sqrt{1+y^{,2} }-\frac{d}{dx}(\frac{yy^, }{\sqrt{1+y^{,2}}})=0$，$\sqrt{1+y^{,2}}-\frac{y^{,2}}{\sqrt{1+y^{,2}}}-\frac{y^, y^{,,}}{\sqrt{1+y^{,2}}}+\frac{yy^{,2}y^{,,}}{(1+y^{,2} )^{\frac{3}{2} }}=0$。<br
	等式两边同时乘以$\frac{y^{,}}{(1+y^{,2} )^{\frac{3}{2} }}$，化简可得$\frac{d}{dx} (\frac{y}{\sqrt{1+y^{,2}}} )=0$，则$\frac{y}{\sqrt{1+y^{,2}}}=C$，$x=\int \frac{dy}{\sqrt{\frac{y^2}{C^2}-1 } }$。<br>
令$y=C cos⁡ht$，则$dy= C sin⁡ht$，带入等式$x=\int \frac{dy}{\sqrt{\frac{y^2}{C^2}-1 } }$可得到$x=Ct+D$，反解得到$y=C cos⁡h  \frac {x-D}{C}$。<br>

**问题4.2**  采用Jensen不等式(Chandler 1987; Jensen 1906)证明K-L散度≥0。<br>**答**：K-L散度表达式为：$KL(q||p)=-\sum _{Z}q(Z)\ln {\frac{p(Z|X,\theta )}{q(Z)} }= \sum _{Z}q(Z)\ln {\frac{q(Z)}{p(Z|X,\theta )} }=E(\ln {\frac{q(Z)}{p(Z|X,\theta )}})=E(-\ln {\frac{p(Z|X,\theta )}{q(Z)} })$，由于该对数函数是凸函数，根据凸函数的性质可得，$E(-\ln {\frac{p(Z|X,\theta )}{q(Z)} })\ge -\ln {[\sum q(Z)\frac{p(Z|X,\theta )}{q(Z)}]}=-\ln {[\sum p(Z|X,\theta )]}=0$，因此$KL(q||p)\ge 0$。<br>

**问题4.3**  EM算法也用于处理连续数据的缺值插补。令k维数据集$D$包括了一些缺失值，其可靠的观察值为$Y$，缺失值为$X$，令数据集$D~N(μ,Σ)$，试设计EM算法插补缺失值，证明其类似图4.3所示的收敛性。<br>**答**：（1）EM算法通过迭代的方式估计缺失值，首先使用插补法填充缺失值，然后使用期望最大化算法迭代更新缺失值的估计值，直到收敛为止。<br>
（2）证明其收敛性：<br>
EM算法的目标是寻找一个合适的模型参数$θ$使得$P(X|θ)$尽可能大，由于EM是迭代式算法，所以要证明它是收敛的，只需要有$P(X|θ^{(t+1)})≥P(X|θ^{(t)})$成立即可。<br>
我们知道，$log⁡P(X|θ)=log⁡P(X,Z|θ)-log⁡P(Z|X,θ)$，两边同时求关$Z|X,\theta ^{(t)}$的期望，有：$E_{Z|X,\theta ^{(t)}}\left [ \lg {P(X|\theta )}  \right ]=E_{Z|X,\theta ^{(t)}}[\lg {P(X,Z|\theta )}]-E_{Z|X,\theta ^{(t)}}[\lg {P(Z|X,\theta )}]$。<br>对于等式左边，因为$log⁡P(X|θ)$与$Z$无关，所以：$E_{Z|X,\theta ^{(t)}}\left [ \lg {P(X|\theta )}  \right ]=\int P(Z|X,\theta ^{(t)})\lg {P(X|\theta )}dZ=\log P(X,\theta )$。<br>对于等式右边，我们首先看第一项，它其实就是E-step的那个$Q(\theta ,\theta ^{(t)})$，因为$\theta ^{(t+1)}=\arg \max_{\theta }Q(\theta ,\theta ^{(t)})$ ，所以必然有$Q(\theta ^{(t+1)},\theta ^{(t)})\ge Q(\theta ,\theta ^{(t)})$。<br>因为$θ$是一个变量，所以我们可以令$\theta =\theta ^{(t)}$，那$Q(\theta ^{(t+1)},\theta ^{(t)})\ge Q(\theta ^{(t)},\theta ^{(t)})$。<br>我们要证明$\log P(X|\theta ^{(t+1)})≥\log P(X|\theta ^{(t)})$，已经证明了$Q(\theta ^{(t+1)},\theta ^{(t)})\ge Q(\theta ^{(t)},\theta ^{(t)})$，下面只需要保证$E_{Z|X,\theta ^{(t)}}[\log P(Z|X,\theta ^{(t+1)})]\le E_{Z|X,\theta ^{(t)}}[\log P(Z|X,\theta ^{(t)})]$，即证明$E_{Z|X,\theta ^{(t)}}\left [\frac{P(Z|Z,\theta ^{t+1} )}{P(Z|X,\theta ^{t} )}\right ]\le 0$。<br>根据相对熵的性质，上式必然成立，那么$log⁡P(X|θ^{(t+1)})≥log⁡P(\theta ^{(t)})$，得证。
<br>

**问题4.4**  推导先验知识只知道均值情况下，最大熵分布式服从指数分布。<br>**答**：（1）假设随机数X有n可能的值，分别为$x_1,x_2,x_3,⋯,x_n$，其均值为$μ$，对应的概率分别为$p_1,p_2,p_3,⋯,p_n$。<br>
（2）熵为$ent=-∑_{i=1}^{n}p_i\ln {(p_i)}$，我们要最大化熵，等同于最小化$∑_{i=1}^{n}p_i\ln {(p_i)}$，由于$∑_{i=1}^{n}p_i x_i=u$，$∑_{i=1}^{n}p_i=1$由此构造拉格朗日函数：$L=∑_{i=1}^{n}p_i\ln {(p_i)}+a(∑_{i=1}^{n}p_i x_i-μ)+b(∑_{i=1}^{n}p_i-1)$，其中$a$和$b$是拉格朗日乘子。<br>
（3）关于$p_i$对$L$求导：$\frac {∂L}{∂p_i}=\ln {⁡(p_i)}+1+ax_i+b$，根据导数为0可得到：$p_i=\exp ⁡(-1+ax_i-b)$，简化$为p_i=βe-^{ax_i }$。<br>
（4）由此证明，先验知识只知道均值情况下，最大熵分布式服从指数分布。
<br>

**问题4.5**  解释一下为什么$\sum _{i}(\int Q_i (Z_i )\ln Q_i (Z_i )dZ_i)_{Q_{-i}(Z_{-i} ) }=\sum _{i}\int Q_i (Z_i)\ln Q_i (Z_i)dZ_i$从而$L(Q(Z))=\sum _{i}\int Q_{i}(Z_i)\ln Q_i ^{\ast } (Z_i)dZ_i+\sum _{i}\sum _{i}\int Q_i (Z_i)\ln Q_i (Z_i)dZ_i+\ln C=-D_{KL}(Q_i (Z_i)||Q_i ^{\ast }(Z_i)+H(Q_{i-1}(Z_{i-1})) +C$。<br>**答**：。<br>

**问题5.1**  令正态分布$N(μ,σ^2)$的样本${x_1,x_2,⋯,x_N }$，证明其均值$\hat{\mu }_{N} =\frac{1}{N} {\textstyle \sum_{n=1}^{N}x_i}$ 是高斯均值参数$μ$的无偏估计，而样本的方差$\hat{\sigma }_{N}^{2}=\frac{1}{N} {\textstyle \sum_{n=1}^{N}(x_i-\hat{\mu }_{N})^{2}}2$ 是方差参数$\sigma ^{2}$的有偏估计。<br>**答**（1）均值估计：$E(\mu )=E(\frac{1}{N}\sum_{j=1}^{N}x_j )=\frac{1}{N}E(\sum_{j=1}^{N}x_j )=\frac{1}{N}\sum_{j=1}^{N}E(x_j)=\frac{1}{N}\cdot N\cdot \hat{\mu }_{N}=\hat{\mu }_{N}$，样本均值是高斯均值参数的无偏估计。<br>
（2）方差估计：$E(\sigma ^{2})=E(\frac{1}{N}\sum_{j=1}^{N}(x_j -\mu )^{2} )=E(\frac{1}{N}\sum_{j=1}^{N}(x_j^{2} -2x_j \mu +\mu ^{2} ))=E(\frac{1}{N}\sum_{j=1}^{N}x_j^{2}-2\mu \frac{1}{N}\sum_{j=1}^{N}x_j +\frac{1}{N}\sum_{j=1}^{N}\mu ^{2} )=E(\frac{1}{N}\sum_{j=1}^{N}x_j^{2}-2\mu ^{2}+\mu ^{2})=\frac {1}{N}\sum_{j=1}^{N}E(x_j^{2})-E(\mu ^{2} )$。由于$\hat{\sigma }_{N}^{2}=E(x_j -\hat{\mu }_{N})^2=E(x_j^2 -2\hat{\mu }_{N}x_j+\hat{\mu }_{N}^2)=E(x_j^2)-\mu _{N}^2$，所以$E(x_j^2)=\hat{\sigma }_{N}^{2}+\hat{\mu }_{N}^{2}$。<br>又有$E(\mu ^{2} )=D(\mu )+[E(\mu )]^2=D(\frac{1}{N}\sum_{j=1}^{N}x_j)+[E(\mu )]^2=\frac{1}{N^2}\sum_{j=1}^{N}D(x_j)+\hat{\mu }_{N}^{2}=\frac{\hat{\sigma }_{N}^{2}}{N}+ \hat{\mu }_{N}^{2}$。<br>因此，$E(\sigma ^{2})=\hat{\sigma }_{N}^{2}+\hat{\mu }_{N}^{2}-\hat{\mu }_{N}^{2}-\frac{\hat{\sigma }_{N}^{2}}{N}=\frac{N-1}{N}\hat{\sigma }_{N}^{2}$，样本方差是高斯方差参数的有偏估计。<br>

**问题6.1**  在梯度迭代计算中，节点拓扑排序可由深度优先（Depth First Search，DFS）及宽度优先方法（Breadth First Search，BFS）实现，二种方法分别如何实现？<br>**答**：（1）深度优先搜索会沿着一条路径一直搜索下去，在无法继续搜索时回退到刚刚访问过的结点，深度优先搜索的本质就是持续搜索，遍历所有可能的情况，每次一条路走到底。<br>（2）宽度优先搜索是从初始节点开始，应用产生式规则和控制策略生成第一层结点，同时检查目标结点是否在这些生成的结点中。若没有，再用产生式规则将所有第一层结点逐一拓展，得到第二层结点，并逐一检查第二层结点是否包含目标结点。若没有，再用产生式规则拓展第二层结点。如此依次拓展检查下去，直至发现目标结点为止。如果拓展完所有结点，都没有发现目标结点，则问题无解。<br>

**问题7.1**  为什么在深度学习中进行参数正则化时，偏差（bias）不参与正则化？偏差参与正则化会有什么样后果？<br>**答**：（1）在机器学习中，偏差是指模型预测值与真实值之间的平均误差，这个误差通常是由于模型过于简单或欠拟合导致的；方差是指模型对训练数据中噪声的敏感程度。而正则化是一种技术，通过限制参数大小或将某些参数设置为零来控制模型的复杂度，可以帮助平衡偏差和方差，提高模型的泛化能力。当我们使用正则化时，会强制学习一些关键特征，从而减少偏差也就是说偏差会随着参数正则化而发生变化。因此，在深度学习中进行参数正则化时，偏差不参与正则化。<br>（2）若偏差参与正则化，则会导致模型偏差混乱、参数难以控制，将会导致模型难以进行优化。<br>

**问题7.2**  根据式（7.15）是否可以得到以下结论，正则化L1所得的解均为正值或0，没有负值？<br>**答**：可以。式（7.15）$\omega _{i}=\max {\left \{\omega _{i}^{\ast }-\frac{\alpha }{H_{ii} },0  \right \} }$表明，当$\omega _{i}^{\ast }\le \frac{\alpha }{H_{ii} }$ 时，$\omega _{i}=0$；当$\omega _{i}^{\ast }>\frac{\alpha }{H_{ii} }$时，$\omega _{i}=\omega _{i}^{\ast }-\frac{\alpha }{H_{ii} }$ 。此时的$\frac{\alpha }{H_{ii} }$相当于一个界限，小于此界线的全部衰减为0，而高于此界线的部分成为新的权重值。所以，L1正则化所得到的解均为正值或0，没有负值。<br>

**问题7.3**  以上关于L1及L2的推导是针对线性回归的，对于多层深度学习网络，是否有类似的推论？说明二者之间有联系？<br>**答**：（1）对于多层深度学习网络，具有类似关于L1和L2针对线性回归的推论。L1正则化导出的稀疏性质已经被广泛地用于特征选择，使部分子集的权重为0，表明相应的特征可以被安全地忽略。L2正则化使只有在显著减小目标函数方向上的参数会保留得相对完好，无助于目标函数减小方向上的参数会因为正则化而被逐步地衰减掉。<br>（2）多层深度学习网络是线性回归的多次叠加，二者相互关联。<br>

**问题7.4**  无论L1还是L2，应用时需要设定一个超参数$α$，即正则化项目所占的比重，如何确定$α$的值？可否采用Grid Search来设定，以便优化该超参数的值？<br>**答**：（1）首先设定超参数的范围，然后从设定的范围中进行随机取样并使用采样得到的超参数值进行学习、评估识别精度，重复以上步骤，最后根据识别精度结果缩小超参数的范围。<br>（2）Grid Search网格搜索是一种调参手段，即穷举搜索法：在所有候选的参数中，通过循环遍历尝试每一种可能性，表现最好的参数就是最终的结果。该方法存在的弊端就是：比较耗时；参数越多，候选值就越多，耗费时间就越长。在进行神经网络的超参数优化时，与网格搜索等有规律的搜索相比，随机采样的搜索方式效果更好，这是因为：在多个超参数中，各个超参数对最终的识别精度的影响程度不同，有可能某些参数对最终结果影响并不大，但 网格搜索却会固定其他重要参数的值并重复搜索这些不重要参数的取值，这些搜索都是不必要的。<br>

**问题7.5**   说明满足KKT条件可以保证取得局部的极值。<br>**答**：以含有一个不等式约束的KKT条件为例：$\begin{cases}\min f(X) \\ s.t.g(X)\le 0\end{cases}$，其KKT条件为$\begin{cases}∇f(X^* )+λ∇g(X^* )=0\\λg(X^* )=0\\λ≥0\\g(X^* )≤0\end{cases}$。<br>（1）式$∇f(X^* )+λ∇g(X^* )=0$为对拉格朗日函数求梯度（若X为一维则为求导），梯度（导数）为0，则保证所求$X^*$是f(X)的极值。<br>（2）将求梯度（导数）所得的值$X^*$代入至约束条件：若$g(X^* )<0$，刚好满足约束条件，则$X^*$就是我们要找的最优解，此时该约束为不起作用约束，则$λ=0$；若$g(X^*)=0$，也满足约束条件，可转化为等式约束的优化问题；若$g(X^*)>0$，此时的$X^*$不满足约束条件，应舍弃。综上，得到$λg(X^* )=0$这一约束条件及原问题自己的约束条件$g(X^* )≤0$。<br>（3）梯度方向垂直于函数等值线，指向函数值增长的方向；针对此最小化问题，$∇f(X^* )$方向和$g(X^* )$方向共线且相反，则若满足$∇f(X^* )+λ∇g(X^* )=0$，应当$λ≥0$。<br>（4）综上所述，满足KKT条件可以保证取得局部的极值。其他约束的KKT条件道理相同。<br>

**问题7.6**   对于线性的浅层模型，可以采用几何平均值的方式说明权重尺度化有效地恢复了输入数据的期望值。对于深层网络模型，情况如何？<br>**答**：对于深层网络模型，几何平均值不能够说明权重尺度化能够有效地恢复输入数据的期望值，可通过内积的方法将响应逐层“还原”成相应的输入。<br>

**问题7.7**  结合第6章介绍的自微分方法，如何实现算法7.3所显示的误差反向传播更新参数梯度？<br>**答**：在批标准化的误差反向传播过程中根据自动微分方法求解各参数的一阶导数，从而完成误差信息的反馈及参数的学习。此过程中，批量样本的各个实例参与一阶导数的计算，所以反向传播针对该批数据及批正则化层的参数$γ$及$β$，从损失函数后向传播计算风险函数，通过$\frac {∂J}{∂x_i}$根据导数链式法则用于更新产生该隐藏层输入的上层的各参数。<br>

**问题10.1** 假定可以生成[0,1]间的均匀随机数，如何由它来生成服从正态分布的伪随机数？即怎么将均匀分布$U[0,1]$映射成标准正态分布$N(0,1)$？<br>**答**：建立均匀分布的随机变量$x$同符合标准正态分布的随机变量$y$之间的转换函数$y=f(x)$，即可通过均匀分布得到标准正态分布。<br>

### **软件及共享代码（Software and shared codes）**
**1.	Library of Physics-Aware Geograph Hybrid Modeling (phygeograph)**
<br>![](images/3.png)<br>
The python library of physics-inspired geograph hybrid modeling (phygeograph). Current version just supports the PyTorch package of deep geometric learning and will extend to the others in the future. This package is for the paper: " Improving air quality assessment using physics-inspired deep graph learning" (Li, L., Wang, J., Franklin, M., et al., preprint: https://assets.researchsquare.com/files/rs-2303533/v1/53848e3c-b7b3-4051-8782-6137790913db.pdf?c=1669922832)<br>
Online address: https://pypi.org/project/phygeograph<br>
Online sample data: https://github.com/phygeograph/phygeographdata

**2.	Library of Geographic Graph Hybrid Network (geographnet)**
<br>![](images/4.png)<br>
The python library of geographic graph hybrid network with attention layers (geographnet). Current version just supports the PyTorch package of deep geometric learning and will extend to the others in the future. This package is for the paper: "Geographic Graph Hybrid Network for Robust Inversion of Particular Matters" (Li, L., 2021, Remote Sens., 13,21, 4341; https://www.mdpi.com/2072-4292/13/21/4341)<br>
Online address: https://pypi.org/project/geographnet  

**3.	Library of Autoencoder-based Residual Deep Network (resautonet)**
<br>![](images/5.png)<br>
The python library of autoencoder based residual deep network (resautonet). Current version (2.0) just supports the KERAS package of deep learning and will extend to the others in the future. This package is for the papers: "Encoder-decoder full residual deep networks for robust regression and spatiotemporal estimation" (Li, L., Fang, Y., and Wu, J. etc., 2021, IEEE Trans Neural Netw Learn Syst., 32(9): 4217–4230); https://ieeexplore.ieee.org/document/9186306); "Spatiotemporal imputation of MAIAC AOD using deep learning with downscaling" (Li, L., Franklin, M., and Girguisa, M. et al., 2020, Remote Sens. Environ., 32(9): 4217–4230; https://www.sciencedirect.com/science/article/abs/pii/S0034425719306042)<br>
Online address: https://pypi.org/project/resautonet 

**4.	Library for Bagging of Deep Residual Neural Networks (baggingrnet)**
<br>![](images/6.png)<br>
This package provides The python Library for Bagging of Deep Residual Neural Networks (baggingrnet). Current version just supports the KERAS package of deep learning and will extend to the others in the future. The following functionaity is provoded in this package: * model multBagging: Major class to parallel bagging of autoencoder-based deep residual networks. You can setup its aruments for optimal effects. See the class and its member functions' help for details. resAutoencoder: Major class of the base model of autoencoder-based deep residual network. See the specifics for its details. ensPrediction: Major class to ensemble predictions and optional evaluation for independent test. * util pmetrics: main metrics including rsquare and rmse etc. This package is for the paper: "Ensemble-based deep learning for estimating PM2.5 over California with
multisource big data including wildfire smoke" (Li, L., Girguis, M, Lurmann, F., et al. 2020, 145, 106143, https://www.sciencedirect.com/science/article/pii/S0160412020320985).
<br>Online address: https://pypi.org/project/baggingrnet 

**5.	Library of Full Residual Deep Network with Attention Layers (fullresattn)**
<br>![](images/7.png)<br>
The python library of full residual deep network with attention layers (fullresattn). Current version just supports the KERAS package of deep learning and will extend to the others in the future. <br>
Online address: https://pypi.org/project/fullresattn 

**6.	Library for Deep Residual Multiscale Segmenter (resmcseg)**
<br>![](images/8.png)<br>
The python library of Deep Residual Multiscale Segmenter (autonet). Current version just supports the KERAS package of deep learning and will extend to the others in the future. This package is for the paper: Deep Residual Autoencoder with Multiscaling for Semantic Segmentation of Land-Use Images (2019, Remote Sens., 11(18), 2142; https://www.mdpi.com/2072-4292/11/18/2142)<br>
Online address: https://pypi.org/project/resmcseg

### **专利（Patents）**
**1.	一种基于深度双模态的气象参数精细尺度转化方法 （发明专利）**<br>A fine-scale transformation method of meteorological parameters based on deep dual-mode (invention patent)<br>

**2.	基于非监督限制性优化的空气污染时空趋势预测方法  （发明专利）**<br>An air pollution spatiotemporal trend prediction method based on unsupervised restrictive optimization (invention patent)<br>

**3.	一种多源时空大数据深度融合的空气污染预测方法  （发明专利）**<br>An air pollution prediction method based on deep fusion of multi-source spatiotemporal big data (invention patent)<br>

**4.	 一种半监督深度图卷积的遥感土地利用语义分割方法 （发明专利）**<br>A semi-supervised depth image convolution method for semantic segmentation of remote sensing land use (invention patent)<br>

**5.	一种基于无人机的地物高光谱仪遥感土地利用样本采集仪（实用新型专利）**<br>A remote sensing land use sample collection instrument based on a UAV-based ground object hyperspectrometer (utility model patent)<br>

**6.	一种提高海量空间数据处理效率的方法 （发明专利）**<br>A method to improve the processing efficiency of massive spatial data (invention patent)<br>
![](images/9.jpg#pic_center)<br>

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']]}
});
</script>

<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.2/MathJax.js?config=default"></script>
