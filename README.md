# DocBank

# 好像没有训练模型的结构

**\*\*\*\*\* DocBank is a natural extension of the TableBank ([repo](https://github.com/doc-analysis/TableBank), [paper](https://arxiv.org/abs/1903.01949)) dataset \*\*\*\*\***

**\*\*\*\*\* LayoutLM ([repo](https://github.com/microsoft/unilm/tree/master/layoutlm), [paper](https://arxiv.org/abs/1912.13318)) is an effective pre-training method of text and layout and archives the SOTA result on DocBank \*\*\*\*\***

DocBank是使用弱监督方法构建的新的大规模数据集。 它使模型能够集成文本和布局信息以用于下游任务。 
当前的DocBank数据集总共包括500K文档页面，其中400K用于训练，50K用于验证，50K用于测试。



## Introduction
对于文档布局分析任务，已经有一些基于图像的文档布局数据集，而其中大多数是为计算机视觉方法构建的，
很难应用于NLP方法。另外，基于图像的数据集主要包括页面图像和大型语义结构的边界框，
它们不是细粒度的token级注释。此外，产生人工标记和细粒度的令牌级文本块排列也是费时且费力的。
因此，至关重要的是要利用薄弱的监督来以最小的努力获得带标签的细粒度文档，同时使数据易于应用于任何NLP和计算机视觉方法。

为此，我们构建了DocBank数据集，这是一个文档级基准，具有用于布局分析的细粒度token级注释。
与常规的人类标记数据集不同，我们的方法以简单而有效的方式在弱监督下获得了高质量注释。

## Statistics of DocBank
DocBank数据集由具有12种类型的语义单元的500K文档页面组成。

### Semantic Structure Statistics of DocBank
| Split | Abstract | Author | Caption |  Date | Equation | Figure | Footer |  List  | Paragraph | Reference | Section | Table | Title |  Total  |
|:-----:|:--------:|:------:|:-------:|:-----:|:--------:|:------:|:------:|:------:|:---------:|:---------:|:-------:|:-----:|:-----:|:-------:|
| Train |   25,387  |  25,909 |  106,723 |  6,391 |  161,140  |  90,429 |  38,482 |  44,927 |   398,086  |   44,813   |  180,774 | 19,638 | 21,688 |  400,000 |
|       |   6.35%  |  6.48% |  26.68% | 1.60% |  40.29%  | 22.61% |  9.62% | 11.23% |   99.52%  |   11.20%  |  45.19% | 4.91% | 5.42% | 100.00% |
|  Dev  |   3,164   |  3,286  |  13,443  |  797  |   20,154  |  11,463 |  4,804  |  5,609  |   49,759   |    5,549   |  22,666  |  2,374 |  2,708 |  50,000  |
|       |   6.33%  |  6.57% |  26.89% | 1.59% |  40.31%  | 22.93% |  9.61% | 11.22% |   99.52%  |   11.10%  |  45.33% | 4.75% | 5.42% | 100.00% |
|  Test |   3,176   |  3,277  |  13,476  |  832  |   20,244  |  11,378 |  4,876  |  5,553  |   49,762   |    5,641   |  22,384  |  2,505 |  2,729 |  50,000  |
|       |   6.35%  |  6.55% |  26.95% | 1.66% |  40.49%  | 22.76% |  9.75% | 11.11% |   99.52%  |   11.28%  |  44.77% | 5.01% | 5.46% | 100.00% |
| Total |   31,727  |  32,472 |  133,642 |  8,020 |  201,538  | 113,270 |  48,162 |  56,089 |   497,607  |   56,003   |  225,824 | 24,517 | 27,125 |  500,000 |
|       |   6.35%  |  6.49% |  26.73% | 1.60% |  40.31%  | 22.65% |  9.63% | 11.22% |   99.52%  |   11.20%  |  45.16% | 4.90% | 5.43% | 100.00% |

### Year Statistics of DocBank

|  Year |  Train |         |  Dev  |         |  Test |         |   ALL  |         |
|:-----:|:------:|:-------:|:-----:|:-------:|:-----:|:-------:|:------:|:-------:|
|  2014 |  65,976 |  16.49% |  8,270 |  16.54% |  8,112 |  16.22% |  82,358 |  16.47% |
|  2015 |  77,879 |  19.47% |  9,617 |  19.23% |  9,700 |  19.40% |  97,196 |  19.44% |
|  2016 |  87,006 |  21.75% | 10,970 |  21.94% | 10,990 |  21.98% | 108,966 |  21.79% |
|  2017 |  91,583 |  22.90% | 11,623 |  23.25% | 11,464 |  22.93% | 114,670 |  22.93% |
|  2018 |  77,556 |  19.39% |  9,520 |  19.04% |  9,734 |  19.47% |  96,810 |  19.36% |
| Total | 400,000 | 100.00% | 50,000 | 100.00% | 50,000 | 100.00% | 500,000 | 100.00% |

### Comparison of DocBank with existing document layout analysis datasets
|     Dataset     |  #Pages | #Units | Image-based? | Text-based? | Fine-grained? | Extendable? |
|:---------------:|:-------:|:------:|:------------:|:-----------:|:-------------:|:-----------:|
| Article Regions |   100   |    9   |       ✔      |      ✘      |       ✔       |      ✘      |
|     GROTOAP2    | 119,334 |   22   |       ✔      |      ✘      |       ✘       |      ✘      |
|    PubLayNet    | 364,232 |    5   |       ✔      |      ✘      |       ✔       |      ✘      |
|    TableBank    | 417,234 |    1   |       ✔      |      ✘      |       ✔       |      ✔      |
|     DocBank     | 500,000 |   12   |       ✔      |      ✔      |       ✔       |      ✔      |

## Baseline
由于数据集在标记级别得到了完全注释，因此我们将文档布局分析任务视为基于文本的序列标记任务

在此设置下，我们评估了数据集上的三个代表性的预训练语言模型，包括BERT，RoBERTa和LayoutLM，以验证DocBank的有效性。

### Metrics
于我们模型的输入是序列化的2-D文档，因此典型的BIO标签评估不适合我们的任务。每个语义单元的令牌可以在输入序列中不连续地分布。

在这种情况下，我们提出了一个新的指标，特别是针对基于文本的文档布局分析方法。对于每种文档语义结构，我们分别计算其度量。定义如下：

<img src='Metrics.png' width=500>

### Settings
我们的BERT和RoBERTa基线是基于HuggingFace的Transformers，
而LayoutLM基线是通过[LayoutLM的官方存储库]（https://aka.ms/layoutlm）中的代码库实现的。 
我们使用了8个V100 GPU，每个GPU的批量大小为10。 
训练400K文档页面上的1个epoch需要5个小时。 
我们使用BERT和RoBERTa标记器对训练样本进行标记，并使用AdamW优化模型。 
优化器的初始学习率为5e-5。 我们将数据分成最大块大小N = 512。

### Results

#### The evaluation results of BERT, RoBERTa and LayoutLM
|      Unit     | bert-base | roberta-base | layoutlm-base | bert-large | roberta-large | layoutlm-large |
|:-------------:|:---------:|:------------:|:-------------:|:----------:|:-------------:|:--------------:|
|    Abstract   |  0.9294   |    0.9288    |    **0.9816**     |   0.9286   |    0.9479     |     0.9784     |
|     Author    |  0.8484   |    0.8618    |    0.8595     |   0.8577   |    0.8724     |     **0.8783**     |
|    Caption    |  0.8629   |    0.8944    |    **0.9597**     |   0.8650   |    0.9081     |     0.9556     |
|    Equation   |  0.8152   |    0.8248    |    0.8947     |   0.8177   |    0.8370     |     **0.8974**     |
|     Figure    |  1.0000   |    1.0000    |    1.0000     |   1.0000   |    1.0000     |     **1.0000**     |
|     Footer    |  0.7805   |    0.8014    |    0.8957     |   0.7814   |    0.8392     |     **0.9146**     |
|      List     |  0.7133   |    0.7353    |    0.8948     |   0.6960   |    0.7451     |     **0.9004**     |
|   Paragraph   |  0.9619   |    0.9646    |    0.9788     |   0.9619   |    0.9665     |     **0.9790**     |
|   Reference   |  0.9310   |    **0.9341**    |    0.9338     |   0.9284   |    0.9334     |     0.9332     |
|    Section    |  0.9081   |    0.9337    |    **0.9598**     |   0.9065   |    0.9407     |     0.9596     |
|     Table     |  0.8296   |    0.8389    |    0.8633     |   0.8320   |    0.8494     |     **0.8679**     |
|     Title     |  0.9442   |    0.9511    |    **0.9579**     |   0.9430   |    0.9461     |     0.9552     |
| Macro average |  0.8705   |    0.8793    |    0.9214     |   0.8707   |    0.8933     |     **0.9243**     |



我们在DocBank的测试集上评估了六个模型。我们注意到LayoutLM在{摘要，作者，标题，方程式，图形，页脚，列表，段落，节，表，标题}标签上得分最高。 
RoBERTa模型在“reference”标签上获得最佳性能，但与LayoutLM的差距很小。
这表明在文档布局分析任务中，LayoutLM体系结构明显优于BERT和RoBERTa体系结构。

## License
DocBank is released under the [Attribution-NonCommercial-NoDerivs License](https://creativecommons.org/licenses/by-nc-nd/4.0/). You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may not use the material for commercial purposes. If you remix, transform, or build upon the material, you may not distribute the modified material.

## **Model Zoo and Scripts**

训练好的模型下载连接 [DocBank Model Zoo](MODEL_ZOO.md).

我们提供了一个脚本，可将PDF文件转换为DocBank格式的数据。
您可以在脚本目录中运行PDF处理脚本pdf_process.py。您可能需要通过pip软件包安装程序安装此脚本的某些依赖项。

~~~bash
cd scripts
python pdf_process.py   --data_dir /path/to/pdf/directory \
                        --output_dir /path/to/data/output/directory
~~~

## **Get Data**

**\*\*Please DO NOT re-distribute our data.\*\***

If you use the corpus in published work, please cite it referring to the "Paper and Citation" Section.

我们提供了[100 samples](DocBank_samples/README.md)进行预览，以及在indexed_files目录中提供了训练，验证和测试集的索引文件。

<!-- Please fill this [form](https://forms.office.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbRw1hSTX2waZIoerSk1J6CyNUQjA3QlBVUDYxMTY4NFpVR1UxNVRRU0ZIUS4u). If the review is approved, the download link will be sent to your email address. 

The link will be reviewed and sent **the next Monday after the application** -->
The annotations and original document pictures of the DocBank dataset **can be download from the [DocBank dataset homepage](https://doc-analysis.github.io/docbank-page/index.html)**.


## **Paper and Citation**
### DocBank: A Benchmark Dataset for Document Layout Analysis

Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, Ming Zhou

https://arxiv.org/abs/2006.01038
```
@misc{li2020docbank,
    title={DocBank: A Benchmark Dataset for Document Layout Analysis},
    author={Minghao Li and Yiheng Xu and Lei Cui and Shaohan Huang and Furu Wei and Zhoujun Li and Ming Zhou},
    year={2020},
    eprint={2006.01038},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
