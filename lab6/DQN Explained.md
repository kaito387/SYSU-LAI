# DQN Explained

本文章不包含的前置知识：深度学习与神经网络。

首先是需要理解强化学习里的 Q-Learning。

那么首先需要理解强化学习。

## RL Overview

强化学习大概可以看作是玩游戏，我们当前有一个状态（state）$s$，每次行动可以选择一个动作（action）$a$，那么环境（environment）（可以看作游戏引擎）在你做完这个动作后，会返回一个新的状态 $s'$ 和本次动作的奖励（reward）$r$。我们还有一个策略（policy）$\pi$，表示在状态 $s$ 下，应该选择动作 $\pi(s)$。

> 在下图中，policy 对应的就是 agent。

![The RL process](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/RL_process.jpg)

这里 Agent 的目标就是，最大化累计奖励（cumulative reward or **expected return**）。

---

## Q-Learning Algorithm

Q-Learning 的大概想法，是我们存储一张表，记录 $Q(s, a)$ 表示在状态 $s$ 下执行动作 $a$，未来奖励的期望值。

> Q 可以理解为 Quality

算法初始化的时候，什么都不知道，可以把所有的 $Q(s, a)$ 置 $0$ 或者随机小数。

假设现在计算 $Q(s, a)$，我们先根据上文定义，得到四元组 $(s, a, r, s')$，则 Q-Learning 的更新公式为
$$
Q(s, a)\gets Q(s, a) + \alpha\left(r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right)
$$
其中两个参数：

- $\alpha$：学习率（learning rate），表示我们有多么接受新的估计值。$\alpha = 0$ 表示完全不学习，$\alpha = 1$ 表示完全接受新的估计。
- $\gamma$：折扣系数（discount factor），表示我们多么看好未来价值。$\gamma = 0$ 表示完全不关心未来奖励，$\gamma = 1$ 表示完全累计未来奖励（注意到从 $s'$ 开始，未来的第 $i$ 步的 reward 会被乘以 $\gamma^i$ 的系数），一般不收敛，不会这样设计。

有一个小问题是，如果每次在状态 $s$ 的时候，都选择最优策略 ${\arg\max}_a Q(s, a)$，会导致不去尝试很多可能成为更优解的动作。所以一般采取 $\varepsilon$-greedy，设置一个参数 $\varepsilon$，以 $\varepsilon$ 的概率随机选择一个动作，以 $1-\varepsilon$ 的概率选择最优动作。

训练完之后，可以定义我们的 policy 为 $\pi(s) = {\arg\max}_a Q(s, a)$。

---

## Deep Q-Network(DQN) Explained

Q-Learning 的缺点在于，无法处理一些状态空间特别大的情况（表都开不下），比如游戏 [Space Invaders](https://en.wikipedia.org/wiki/Space_Invaders)。

![A vertical rectangular video game screenshot that is a digital representation of a battle between aliens and a laser cannon. The white aliens hover above four green, inverted U-shaped blocks. Below the blocks is a smaller horizontal block with a triangle on its top.](https://upload.wikimedia.org/wikipedia/en/2/20/SpaceInvaders-Gameplay.gif)

我们想要训练一个能玩这个游戏的 agent。如果使用 Q-Learning 算法，需要设计状态。会发现不管怎么设计都很复杂，如果简单粗暴地直接将游戏画面作为状态，更是有 $256^{3\times210\times160}$ 种状态（考虑到游戏画面是 $210\times 160$ 像素的，每个像素有 R/G/B 三个通道）。

DQN 的想法就是利用深度学习处理这种状态空间特别大的情况。特别的，还能有效处理连续状态空间。

具体地，我们建立一个网络，以「游戏状态（可以是画面像素值）」为输入，以「每个 action（如，上下移动或开火）的估值 $Q_a$」为输出。然后让模型在这个网络里学习。

![Sampling Training](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/sampling-training.jpg)

想法是很好的，不过有一些问题。下面我们来解决这三个问题。

### State Compression

这个游戏，只有静态的一帧画面会缺失很多信息，例如敌人的运动方向或子弹的运动方向。所以我们需要将连续的画面（如取连续的 4 帧画面）作为输入。其次，我们应该尽量简化状态空间，比如可以把画面先缩小为 $84\times 84$ 的尺寸，然后再灰度化（颜色并没有提供特别有价值的信息）。甚至可以裁剪掉不影响你判断游戏局面的地方。

然后经典地，我们将画面接入若干卷积层提取信息，最后通过全连接层接到对动作的估值输出。

### Experience Replay

但是我们不能直接一边玩游戏一边训练，因为机器学习里有个很重要的假设，就是训练的样本应该是大致独立同分布（i.i.d.）。否则，想象一下如果连续采样一段时间的样本，可能在这段时间所有敌人都在向右移动，这个 mini-batch 体现的梯度（即给出的更新方向）无法代表整个样本空间（敌人不一定只会往右移动），训练容易震荡。

DQN 引入了 Experience Replay。即把这局游戏的经验存入缓存区，然后训练时从中随机采样，能近似恢复一些 i.i.d. 的性质，使训练更加稳定。

![Experience Replay](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/experience-replay.jpg)

### Fixed Q-Target to stabilize the training

注意我们如果直接使用当前网络提供的估计值 $Q_\theta(s', a')$ 来我们对更新 $Q_\theta(s, a)$ 的估计，是非常不稳定的。情景有点类似下面这种情况：

![Q-target](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/qtarget-1.jpg)

我们的牛仔（Q estimation），尝试通过反向传播更新网络，靠近奶牛（Q target）。但每次反向传播后，奶牛也会因为网络的变化而移动，相当于你在追逐一个移动的靶子。本质上，你不能直接利用证明 Q-Learning 收敛性的方法证明 DQN 也会收敛。所以网络很有可能会震荡，影响训练。

DQN 则引入 $\theta$ 表示训练网络和 $\theta^-$ 表示目标网络，训练时先令 $\theta^-\gets \theta$，之后固定目标网络不动，类似监督学习，而只修改训练网络 $\theta$ 的权值：
$$
Q(s, a)_\theta\gets Q_\theta(s, a) + \alpha\left(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\right)
$$

---

## References

- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction)