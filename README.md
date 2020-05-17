# 🗺️ Natural Language Processing Roadmap

一个自然语言处理的学习路线图。

> ⚠️注意:
> 
> 1. 这个项目包含一个名为 `PCB` 的小实验，这个的 PCB 不是印刷电路板 `Printed Circuit Board` 的意思, 而是 `Paper Code Blog` 的缩写。我认为 `论文`、`代码` 和 `博客` 这三个东西可以让我们兼顾理论和实践同时，快速地掌握知识点！
> 
> 2. 每篇论文后面的星星个数代表论文的重要性（主观意见，仅供参考）。
>     1. 🌟: 一般；
>     2. 🌟🌟: 重要；
>     3. 🌟🌟🌟: 非常重要。

## 1 分词 `Word Segmentation`

**词是能够独立活动的最小语言单位。** 在自然语言处理中，通常都是以词作为基本单位进行处理的。由于英文本身具有天生的优势，以空格划分所有词。而中文的词与词之间没有明显的分割标记，所以在做中文语言处理前的首要任务，就是把连续中文句子分割成「词序列」。这个分割的过程就叫**分词**。[更多](https://www.v2ai.cn/nlp/2018/04/26/NLP-4.html)

### 综述

- paper:
  - [汉语分词技术综述](http://www.lis.ac.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=9402)🌟
  - [国内中文自动分词技术研究综述](http://www.lis.ac.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=11361)🌟
  - [汉语自动分词的研究现状与困难](http://sourcedb.ict.cas.cn/cn/ictthesis/200907/P020090722605434114544.pdf)🌟🌟
  - [汉语自动分词研究评述](http://59.108.48.5/course/mining/12-13spring/%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE/02-01%E6%B1%89%E8%AF%AD%E8%87%AA%E5%8A%A8%E5%88%86%E8%AF%8D%E7%A0%94%E7%A9%B6%E8%AF%84%E8%BF%B0.pdf)🌟🌟
  - [中文分词十年又回顾: 2007-2017](https://arxiv.org/pdf/1901.06079.pdf)🌟🌟🌟
- code: [chinese-word-segmentation](https://github.com/Ailln/chinese-word-segmentation) ![](https://img.shields.io/github/stars/Ailln/chinese-word-segmentation.svg)
- blog: [深度学习中文分词调研](http://www.hankcs.com/nlp/segment/depth-learning-chinese-word-segmentation-survey.html)

## 2 词嵌入 `Word Embedding`

**词嵌入**就是找到一个映射或者函数，生成在一个新的空间上的表示，该表示被称为「单词表示」。[更多](https://www.v2ai.cn/2018/04/26/nature-language-processing/2-word-segmentation/)

### 综述

- paper:
  - [Word Embeddings: A Survey](https://arxiv.org/pdf/1901.09069.pdf) 🌟🌟🌟
  - [Visualizing Attention in Transformer-Based Language Representation Models](https://arxiv.org/pdf/1904.02679.pdf) 🌟🌟
  - [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/pdf/2003.08271.pdf) 🌟🌟🌟

### 核心

- **NNLM**: A Neural Probabilistic Language Model [\[Paper\]](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) [\[Code\]](https://github.com/FuYanzhe2/NNLM) [\[Blog\]](https://zhuanlan.zhihu.com/p/21240807) 🌟
- **W2V**: Efficient Estiation of Word Representations in Vector Space [\[Paper\]](https://arxiv.org/abs/1301.3781) 🌟🌟
- **Glove**: Global Vectors for Word Representation [\[Paper\]](https://nlp.stanford.edu/pubs/glove.pdf) 🌟🌟
- **FastText**: Bag of Tricks for Efficient Text Classification [\[Paper\]](https://arxiv.org/pdf/1607.01759.pdf) 🌟🌟
- **ELMo**: Deep contextualized word representations [\[Paper\]](https://arxiv.org/pdf/1802.05365.pdf) 🌟🌟
- **Transformer**: Attention is All you Need [\[Paper\]](https://arxiv.org/pdf/1706.03762.pdf) [\[Code\]](https://github.com/tensorflow/tensor2tensor) [\[Blog\]](http://jalammar.github.io/illustrated-transformer/) 🌟🌟🌟
- **GPT**: Improving Language Understanding by Generative Pre-Training [\[Paper\]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 🌟🌟
- **BERT**: Pre-training of Deep Bidirectional Transformers for Language Understanding [\[Paper\]](https://arxiv.org/pdf/1810.04805.pdf) [\[Code\]](https://github.com/google-research/bert) [\[Blog\]](https://zhuanlan.zhihu.com/p/49271699) 🌟🌟🌟

### 其他

- BERT Rediscovers the Classical NLP Pipeline [\[Paper\]](https://arxiv.org/pdf/1905.05950.pdf) 🌟🌟

## 3 序列标注 `Sequence Labeling`

### 综述

- blog: [Sequence Labeling的发展史（DNNs+CRF）](https://zhuanlan.zhihu.com/p/34828874)

### Bi-LSTM + CRF

- paper: [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://www.aclweb.org/anthology/P16-1101)🌟🌟
- code:
  - [pytorch_NER_BiLSTM_CNN_CRF](https://github.com/bamtercelboo/pytorch_NER_BiLSTM_CNN_CRF) ![](https://img.shields.io/github/stars/bamtercelboo/pytorch_NER_BiLSTM_CNN_CRF.svg)
  - [NN_NER_tensorFlow](https://github.com/LopezGG/NN_NER_tensorFlow) ![](https://img.shields.io/github/stars/LopezGG/NN_NER_tensorFlow.svg)
  - [End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial](https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial) ![](https://img.shields.io/github/stars/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial.svg)
- blog: [Bi-directional LSTM-CNNs-CRF](https://zhuanlan.zhihu.com/p/30791481)
