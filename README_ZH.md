# Stable Diffusion web UI Advance Prompt Tuning

## 特性：

+ 添加负面词条的prompt embedding学习，大幅提高生成图像质量。可以仅从单张图像学习(one-shot learning)
+ 添加重建损失，提高生成图像的细节质量和丰富程度。
+ 增加基于[ConvNeXt](https://github.com/facebookresearch/ConvNeXt)模型的prompt tuning。使用[7eu7d7](https://github.com/7eu7d7)之前开发的[AI p站爬虫](https://github.com/7eu7d7/pixiv_AI_crawler)中 
训练的convnext模型，用另一个AI评估生成图像质量，作为判别器辅助模型训练。可以让prompt学会高质量这一概念，或是学会你的xp。

## 使用方法

在创建embedding时，会自动分别创建正面和负面的```{name}```和```{name}-uc```，建议大小分别为3和10。
训练prompt tuning时，选择正面prompt ```{name}```，会自动同时训练负面embedding。

其他使用方法与原版一致，prompt tuning训练过程可以选择是否加载判别器模型。

模型下载：

[checkpoint-best_t5](https://github.com/7eu7d7/pixiv_AI_crawler/releases/download/v2/checkpoint-best_t5.pth) 

[ImageNet](https://github.com/facebookresearch/ConvNeXt#results-and-pre-trained-models)

下载模型后将路径填写在`ConvNext model path`中

训练判别器或使用现有判别器见[AI p站爬虫](https://github.com/7eu7d7/pixiv_AI_crawler)。
