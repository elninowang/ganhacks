# How to Train a GAN? Tips and tricks to make GANs work

# 如何训练GAN？ 提示和技巧使GAN工作

While research in Generative Adversarial Networks (GANs) continues to improve the
fundamental stability of these models, we use a bunch of tricks to train them and make them stable day to day.

虽然生成对抗网络（GAN）的研究继续改善
这些模型的基本稳定性，我们用一大堆技巧训练他们，使其稳定日复一日。

Here are a summary of some of the tricks.

以下是一些技巧的总结。

[Here's a link to the authors of this document](#authors)

If you find a trick that is particularly useful in practice, please open a Pull Request to add it to the document.
If we find it to be reasonable and verified, we will merge it in.

如果您发现在实践中特别有用的技巧，请打开一个Pull请求将其添加到文档中。
如果我们发现它是合理和验证的，我们将它合并。

## 1. Normalize the inputs 归一化输入

- normalize the images between -1 and 1
- Tanh as the last layer of the generator output

- 在-1和1之间归一化
- Tanh作为生成器输出的最后一层

## 2: A modified loss function 修改的损失函数

In GAN papers, the loss function to optimize G is `min (log 1-D)`, but in practice folks practically use `max log D`
  - because the first formulation has vanishing gradients early on
  - Goodfellow et. al (2014)
  
在GAN论文中，优化G的损失函数是 `min (log 1-D)`，但实际上人们实际上使用 `max log D`
  - 因为第一个配方早期有消失的梯度
 - Goodfellow 等 （2014）

In practice, works well:
  - Flip labels when training generator: real = fake, fake = real
  
在实践中，效果很好：
  - 训练发生器时翻转标签：real = fake，fake = real

## 3: Use a spherical Z  使用球形Z
- Dont sample from a Uniform distribution
- 不要从统一分配中抽取

![cube](images/cube.png "Cube")

- Sample from a gaussian distribution
- 高斯分布的样本

![sphere](images/sphere.png "Sphere")

- When doing interpolations, do the interpolation via a great circle, rather than a straight line from point A to point B
- Tom White's [Sampling Generative Networks](https://arxiv.org/abs/1609.04468) ref code https://github.com/dribnet/plat has more details

- 进行插值时，通过一个大圆做插补，而不是从点A到点B的直线
- Tom White的 [采样生成网络](https://arxiv.org/abs/1609.04468) [ref代码](https://github.com/dribnet/plat) 有更多的细节

## 4: BatchNorm 批归一化

- Construct different mini-batches for real and fake, i.e. each mini-batch needs to contain only all real images or all generated images.
- when batchnorm is not an option use instance normalization (for each sample, subtract mean and divide by standard deviation).]

- 构建实际和假的不同的迷你批次，即每个迷你批次只需要包含所有的真实图像或所有生成的图像。
- 当批归一化不是选项时，使用实例归一化（对于每个样本，减去平均值并除以标准偏差）。

![batchmix](images/batchmix.png "BatchMix")

## 5: Avoid Sparse Gradients: ReLU, MaxPool  避免稀疏梯度：ReLU，MaxPool
- the stability of the GAN game suffers if you have sparse gradients
- LeakyReLU = good (in both G and D)
- For Downsampling, use: Average Pooling, Conv2d + stride
- For Upsampling, use: PixelShuffle, ConvTranspose2d + stride
  - PixelShuffle: https://arxiv.org/abs/1609.05158

- 如果您有稀疏梯度，GAN游戏的稳定性将受到损害
- LeakyReLU = 好（在G和D中）
- 对于下采样，使用：平均池，Conv2d + stride
- 对于上采样，请使用：PixelShuffle，ConvTranspose2d + stride
  - PixelShuffle：https://arxiv.org/abs/1609.05158

## 6: Use Soft and Noisy Labels 使用软和嘈杂的标签

- Label Smoothing, i.e. if you have two target labels: Real=1 and Fake=0, then for each incoming sample, if it is real, then replace the label with a random number between 0.7 and 1.2, and if it is a fake sample, replace it with 0.0 and 0.3 (for example).
  - Salimans et. al. 2016
- make the labels the noisy for the discriminator: occasionally flip the labels when training the discriminator

- 标签平滑，即如果您有两个目标标签：Real = 1和Fake = 0，则对于每个传入的样本，如果它是真实的，则用0.7和1.2之间的随机数替换标签，如果它是假的 样品，替换为0.0和0.3（例如）。
  - Salimans等。人。2016
- 使标签对于鉴别器有噪音：训练鉴别器时偶尔翻转标签

## 7: DCGAN / Hybrid Models DCGAN/混合模型

- Use DCGAN when you can. It works!
- if you cant use DCGANs and no model is stable, use a hybrid model :  KL + GAN or VAE + GAN

- 可以使用DCGAN。 有用！
- 如果您不能使用DCGAN，且型号不稳定，请使用混合型号：KL+GAN 或 VAE+GAN

## 8: Use stability tricks from RL  使用RL的稳定性技巧

- Experience Replay
  - Keep a replay buffer of past generations and occassionally show them
  - Keep checkpoints from the past of G and D and occassionaly swap them out for a few iterations
- All stability tricks that work for deep deterministic policy gradients
- See Pfau & Vinyals (2016)

- 体验回放
  - 保留过去几代的重放缓冲区，并偶尔显示它们
  - 保留G和D过去的检查点，然后将它们交换出来几次
- 所有稳定性技巧都适用于深层确定性政策梯度
- 见Pfau＆Vinyals（2016）

## 9: Use the ADAM Optimizer 使用ADAM优化器

- optim.Adam rules!
  - See Radford et. al. 2015
- Use SGD for discriminator and ADAM for generator

- best.Adam规则！
  - 见Radford et。人。2015年
- 将SGD用于鉴别器和ADAM用于发生器

## 10: Track failures early 及早跟踪故障

- D loss goes to 0: failure mode
- check norms of gradients: if they are over 100 things are screwing up
- when things are working, D loss has low variance and goes down over time vs having huge variance and spiking
- if loss of generator steadily decreases, then it's fooling D with garbage (says martin

- D丢失到0：故障模式
- 检查梯度的规范：如果他们超过100件事情正在拧紧
- 当事情发生时，D损失的方差偏低，随着时间的推移而下降，造成巨大变化
- 如果生成器的损耗稳定下降，那么他就能用垃圾数据愚弄D（马丁说）

## 11: Dont balance loss via statistics (unless you have a good reason to)

## 11: 不通过统计平衡损失（除非你有一个很好的理由）

- Dont try to find a (number of G / number of D) schedule to uncollapse training
- It's hard and we've all tried it.
- If you do try it, have a principled approach to it, rather than intuition

- 不要试图找到一个（G的数量/D的数量）时间表进行不合格的训练
- 很难，我们都试过了。
- 如果你尝试它，有一个原则性的方法，而不是直觉

For example 例如
```
while lossD > A:
  train D
while lossG > B:
  train G
```

## 12: If you have labels, use them  如果您有标签，请使用它们

- if you have labels available, training the discriminator to also classify the samples: auxillary GANs

- 如果您有标签可用，训练鉴别器还对样品进行分类：auxillary GANs

## 13: Add noise to inputs, decay over time  增加输入噪声，随着时间的推移衰减

- Add some artificial noise to inputs to D (Arjovsky et. al., Huszar, 2016)
  - http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
  - https://openreview.net/forum?id=Hk4_qw5xe
- adding gaussian noise to every layer of generator (Zhao et. al. EBGAN)
  - Improved GANs: OpenAI code also has it (commented out)
  
- 向D的输入添加一些人为噪声（Arjovsky et al。，Huszar，2016）
  - http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
  - https://openreview.net/forum?id=Hk4_qw5xe
- 向每层生成器增加高斯噪声（Zhao et al。EBGAN）
  - 改进的GAN：OpenAI代码也有它（注释掉）

## 14: [notsure] Train discriminator more (sometimes) 
[不确定]，(有时)训练辨别器更多

- especially when you have noise
- hard to find a schedule of number of D iterations vs G iterations

- 特别是当你有噪音
- 很难找到D次迭代次数与G次迭代的时间表

## 15: [notsure] Batch Discrimination 
[不确定]批量辨识

- Mixed results 混合结果

## 16: Discrete variables in Conditional GANs 条件GAN中的离散变量

- Use an Embedding layer
- Add as additional channels to images
- Keep embedding dimensionality low and upsample to match image channel size

- 使用嵌入层
- 作为附加频道添加到图像
- 保持嵌入维度低和上采样以匹配图像通道大小

## 17: Use Dropouts in G in both train and test phase
在测试和测试阶段，在G中使用Dropouts

- Provide noise in the form of dropout (50%).
- Apply on several layers of our generator at both training and test time
- https://arxiv.org/pdf/1611.07004v1.pdf

- 以dropout的形式提供噪音 (50％)。
- 在训练和测试的时候在我们的生成器的几层上应用
- https://arxiv.org/pdf/1611.07004v1.pdf

## Authors 作者
- Soumith Chintala
- Emily Denton
- Martin Arjovsky
- Michael Mathieu
