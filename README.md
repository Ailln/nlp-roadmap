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

**词是能够独立活动的最小语言单位。** 在自然语言处理中，通常都是以词作为基本单位进行处理的。由于英文本身具有天生的优势，以空格划分所有词。而中文的词与词之间没有明显的分割标记，所以在做中文语言处理前的首要任务，就是把连续中文句子分割成「词序列」。这个分割的过程就叫**分词**。[了解更多](https://www.v2ai.cn/2018/04/26/nature-language-processing/2-word-segmentation/)

### 综述

- 汉语分词技术综述 [\[Paper\]](http://www.lis.ac.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=9402) 🌟
- 国内中文自动分词技术研究综述 [\[Paper\]](http://www.lis.ac.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=11361) 🌟
- 汉语自动分词的研究现状与困难 [\[Paper\]](http://sourcedb.ict.cas.cn/cn/ictthesis/200907/P020090722605434114544.pdf) 🌟🌟
- 汉语自动分词研究评述 [\[Paper\]](http://59.108.48.5/course/mining/12-13spring/%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE/02-01%E6%B1%89%E8%AF%AD%E8%87%AA%E5%8A%A8%E5%88%86%E8%AF%8D%E7%A0%94%E7%A9%B6%E8%AF%84%E8%BF%B0.pdf) 🌟🌟
- 中文分词十年又回顾: 2007-2017 [\[Paper\]](https://arxiv.org/pdf/1901.06079.pdf) 🌟🌟🌟
- chinese-word-segmentation [\[Code\]](https://github.com/Ailln/chinese-word-segmentation)
- 深度学习中文分词调研 [\[Blog\]](http://www.hankcs.com/nlp/segment/depth-learning-chinese-word-segmentation-survey.html)

## 2 词嵌入 `Word Embedding`

**词嵌入**就是找到一个映射或者函数，生成在一个新的空间上的表示，该表示被称为「单词表示」。[了解更多](https://www.v2ai.cn/2018/08/27/nature-language-processing/6-word-embedding/)

### 综述

- Word Embeddings: A Survey [\[Paper\]](https://arxiv.org/pdf/1901.09069.pdf) 🌟🌟🌟
- Visualizing Attention in Transformer-Based Language Representation Models [\[Paper\]](https://arxiv.org/pdf/1904.02679.pdf) 🌟🌟
- Pre-trained Models for Natural Language Processing: A Survey [\[Paper\]](https://arxiv.org/pdf/2003.08271.pdf) 🌟🌟🌟

### 核心

- **NNLM**: A Neural Probabilistic Language Model [\[Paper\]](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) [\[Code\]](https://github.com/FuYanzhe2/NNLM) [\[Blog\]](https://zhuanlan.zhihu.com/p/21240807) 🌟
- **W2V**: Efficient Estiation of Word Representations in Vector Space [\[Paper\]](https://arxiv.org/abs/1301.3781) 🌟🌟
- **Glove**: Global Vectors for Word Representation [\[Paper\]](https://nlp.stanford.edu/pubs/glove.pdf) 🌟🌟
- **FastText**: Bag of Tricks for Efficient Text Classification [\[Paper\]](https://arxiv.org/pdf/1607.01759.pdf) 🌟🌟
- **ELMo**: Deep contextualized word representations [\[Paper\]](https://arxiv.org/pdf/1802.05365.pdf) 🌟🌟
- **Transformer**: Attention is All you Need [\[Paper\]](https://arxiv.org/pdf/1706.03762.pdf) [\[Code\]](https://github.com/tensorflow/tensor2tensor) [\[Blog\]](http://jalammar.github.io/illustrated-transformer/) 🌟🌟🌟
- **GPT**: Improving Language Understanding by Generative Pre-Training [\[Paper\]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 🌟
- **GPT2**: Language Models are Unsupervised Multitask Learners [\[Paper\]](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) [\[Code\]](https://github.com/openai/gpt-2) [\[Blog\]](https://openai.com/blog/better-language-models/) 🌟🌟
- **GPT3**: Language Models are Few-Shot Learners [\[Paper\]](https://arxiv.org/pdf/2005.14165.pdf) [\[Code\]](https://github.com/openai/gpt-3) 🌟🌟
- **BERT**: Pre-training of Deep Bidirectional Transformers for Language Understanding [\[Paper\]](https://arxiv.org/pdf/1810.04805.pdf) [\[Code\]](https://github.com/google-research/bert) [\[Blog\]](https://zhuanlan.zhihu.com/p/49271699) 🌟🌟🌟
- **T5**: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer [\[Paper\]](https://arxiv.org/pdf/1910.10683.pdf) [\[Code\]](https://github.com/google-research/text-to-text-transfer-transformer) [\[Blog\]](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) 🌟
- **ERNIE**: ERNIE: Enhanced Language Representation with Informative Entities [\[Paper\]](https://arxiv.org/pdf/1905.07129.pdf) [\[Code\]](https://github.com/thunlp/ERNIE) 🌟

### 其他

- Semi-supervised Sequence Learning [\[Paper\]](https://arxiv.org/pdf/1511.01432.pdf) 🌟🌟
- BERT Rediscovers the Classical NLP Pipeline [\[Paper\]](https://arxiv.org/pdf/1905.05950.pdf) 🌟

## 3 序列标注 `Sequence Labeling`

### 综述

- Sequence Labeling的发展史（DNNs+CRF）[\[Blog\]](https://zhuanlan.zhihu.com/p/34828874)

### Bi-LSTM + CRF

- End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF [\[Paper\]](https://www.aclweb.org/anthology/P16-1101) 🌟🌟

- pytorch_NER_BiLSTM_CNN_CRF [\[Code\]](https://github.com/bamtercelboo/pytorch_NER_BiLSTM_CNN_CRF)
- NN_NER_tensorFlow [\[Code\]](https://github.com/LopezGG/NN_NER_tensorFlow)
- End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial [\[Code\]](https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial)
- Bi-directional LSTM-CNNs-CRF [\[Code\]](https://zhuanlan.zhihu.com/p/30791481)

## 4 知识图谱 `Knowledge Graph`

### 综述

- Towards a Definition of Knowledge Graphs [\[Paper\]](http://ceur-ws.org/Vol-1695/paper4.pdf) 🌟🌟🌟

## 参考

- [thunlp/NLP-THU](https://github.com/thunlp/NLP-THU)
- [iwangjian/Paper-Reading](https://github.com/iwangjian/Paper-Reading)
