<div align="center">
    <img src="lm.jpg" alt="Logo" width="" height="200">

<h1 align="center">Language Modeling</h1>
</div>

## 1. Problem Statement
Language modeling is a fundamental task in natural language processing (NLP) that involves predicting the next word in a sequence given the preceding words. The purpose of language modeling is to create and train a model that generates a meaningful expression. In such a way that it takes an incomplete sentence and by adding words to the end of it, presents a complete and coherent sentence with a concept that can be understood.

This task is crucial for various applications, including text generation, speech recognition, machine translation, and more. Recent advances in language modeling have been driven by large-scale pre-trained models like BERT, GPT-3, and their successors. These models are trained on vast datasets and can be fine-tuned for various specific tasks, demonstrating remarkable capabilities in understanding and generating human-like text. The shift from traditional statistical models to sophisticated neural architectures has led to significant improvements in the ability of machines to understand and generate human language.

## 2. Related Works
This table summarizing the current methods in language modeling, including the deep learning models used and relevant links to papers or GitHub repositories.

| Date       | Title                                  | Description                                                                                                      | Links                                                                                                                                                          |
|------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2017       | Attention Is All You Need              | Introduction of the Transformer model, which uses self-attention mechanisms for language modeling.                | [Paper](https://arxiv.org/abs/1706.03762), [GitHub](https://github.com/tensorflow/tensor2tensor)  
| 2018       | ULMFiT: Universal Language Model Fine-tuning for Text Classification |  Applies transfer learning with LSTM for various NLP tasks, including language modeling and text completion.  | [Paper](https://arxiv.org/abs/1801.06146), [GitHub](https://github.com/fastai/fastai)
| 2018       | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | BERT (Bidirectional Encoder Representations from Transformers) pre-trains deep bidirectional representations.     | [Paper](https://arxiv.org/abs/1810.04805), [GitHub](https://github.com/google-research/bert)                                                                  |
| 2019       | XLNet: Generalized Autoregressive Pretraining for Language Understanding | Combines the strengths of autoregressive and autoencoding models for better performance in language tasks.        | [Paper](https://arxiv.org/abs/1906.08237), [GitHub](https://github.com/zihangdai/xlnet)                                                                       |
| 2019       | RoBERTa: A Robustly Optimized BERT Pretraining Approach | Optimizes BERT by training longer with larger mini-batches and more data.                                         | [Paper](https://arxiv.org/abs/1907.11692), [GitHub](https://github.com/pytorch/fairseq/tree/main/examples/roberta)                                            |
| 2020       | GPT-3: Language Models are Few-Shot Learners | GPT-3 (Generative Pre-trained Transformer 3) with 175 billion parameters, enabling few-shot learning capabilities. | [Paper](https://arxiv.org/abs/2005.14165)                                                                                                                     |
| 2020       | T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | Proposes a unified framework for NLP tasks using the Text-to-Text Transfer Transformer (T5).                      | [Paper](https://arxiv.org/abs/1910.10683), [GitHub](https://github.com/google-research/text-to-text-transfer-transformer)                                      |
| 2020       | Reformer: The Efficient Transformer    | Improves the efficiency of Transformer models using locality-sensitive hashing and reversible layers.              | [Paper](https://arxiv.org/abs/2001.04451), [GitHub](https://github.com/google-research/reformer)                                                              |
| 2021       | GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow | Open-source replication of GPT-3 architecture.                                                                    | [GitHub](https://github.com/EleutherAI/gpt-neo), [Paper](https://arxiv.org/abs/2104.08407)                                                                    |
| 2021       | BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension | BART (Bidirectional and Auto-Regressive Transformers) combines bidirectional and autoregressive approaches.       | [Paper](https://arxiv.org/abs/1910.13461), [GitHub](https://github.com/facebookresearch/fairseq/tree/main/examples/bart)                                       |
| 2021       | Switch Transformers: Scaling to Trillion Parameter Models | Introduces a mixture of experts model to efficiently scale to trillion parameters.                                 | [Paper](https://arxiv.org/abs/2101.03961), [GitHub](https://github.com/tensorflow/lingvo/tree/master/lingvo/core/gshard)                                       |
| 2022       | LaMDA: Language Models for Dialog Applications | Optimizes dialogue applications using open-domain dialogue pre-training.                                           | [Paper](https://arxiv.org/abs/2201.08239)                                                                                                                     |
| 2022       | PaLM: Scaling Language Modeling with Pathways | Utilizes the Pathways system to efficiently train models with up to 540 billion parameters.                        | [Paper](https://arxiv.org/abs/2204.02311)                                                                                                                     |
| 2023       | ChatGPT: Optimizing Language Models for Dialogue | ChatGPT improves interaction in dialogues using reinforcement learning from human feedback (RLHF).                 | [Blog](https://openai.com/blog/chatgpt), [GitHub](https://github.com/openai/gpt-3)                                                                            |


This table and the detailed descriptions provide a comprehensive overview of the current state of language modeling in NLP, highlighting the most significant models and methods along with their corresponding resources.



## 3. The Proposed Method
Long Short-Term Memory (LSTM)
Basic Explanation:
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture designed to effectively capture long-term dependencies in sequential data. Traditional RNNs struggle with long-term dependencies due to the vanishing gradient problem, where gradients become very small and stop the learning process during backpropagation. LSTMs address this issue by using a more complex architecture that includes:

1. **Cell State:** A direct path that allows information to flow unchanged.
2. **Gates:** Mechanisms to control the flow of information:
   - **Forget Gate:** Decides what information to throw away from the cell state.
   - **Input Gate:** Decides which values from the input to update the cell state.
   - **Output Gate:** Decides what part of the cell state to output.

This architecture allows LSTMs to maintain and update information over long sequences, making them suitable for tasks like language modeling and text generation.
 
<div align="center">
   <img  align=center src="LSTM.jpg" alt="Logo"  height="300">
</div>

### Embedding Models

Embedding models are used to convert categorical data, particularly text, into continuous vectors in a high-dimensional space. These vectors capture the semantic meaning of the words, where words with similar meanings are located closer together in the vector space. Word embeddings help transform textual data into a format that can be easily processed by machine learning algorithms.

### Combining LSTM, Embedding, and Fully Connected layers for Language Modeling

When using LSTM for language modeling or text completion, an embedding layer is often used as the first layer to convert input words into dense vectors. These embeddings are then fed into the LSTM, which processes the sequence and generates predictions.

**Used Architecture:**
1. **Embedding Layer:** Converts input words into dense vectors.
2. **LSTM Layer(s):** Processes the sequence of embeddings to capture dependencies.
3. **Fully Connected Layer:** Maps the LSTM outputs to the desired output space (e.g., vocabulary size for next word prediction).

<div align="center">
   <img  align=center src="model architecture.jpg" alt="Logo"  height="300">
</div>
This example demonstrates a basic LSTM model for language modeling, where the embedding layer converts words to dense vectors, and the LSTM processes these vectors to predict the next word in the sequence.

## 4. Implementation
This section delves into the practical aspects of the project's implementation.

### 4.1. Dataset
Under this subsection, you'll find information about the dataset used for Language Modeling task. It includes details about the dataset source, size, composition, preprocessing, and loading applied to it.

The WikiText-2 dataset is a popular benchmark dataset for language modeling tasks. This dataset is more representative of modern English, containing long-term dependencies and requiring a deep understanding of context for accurate predictions. You can load the WikiText-2 dataset using libraries such as [torchtext](https://paperswithcode.com/dataset/wikitext-2) or download from [kaggle](https://www.kaggle.com/datasets/rohitgr/wikitext?select=wikitext-2).

In this work, I downloaded it from kaggle contains 3 (train, valid and test) splits. 
WikiText-2 consists of approximately:

**Train Split:** 36718 lines

**Validation Split:** 3760 lines

**Test Split:** 4358 lines

If you need a rough estimate, here are the approximate token counts based on the original dataset documentation:

**Train Split:** Approximately 2,088,628 tokens

**Validation Split:** Approximately 217,646 tokens

**Test Split:** Approximately 245,569 tokens


**Tokenizing**
For tokenizing the dataset, I used a simple generator method using **yield**.

    def text_read_iterator(token_path ):
      with io.open(token_path, encoding = 'utf-8') as f:
        for line in f:
          yield line
          
tokenizer= get_tokenizer('basic_english')

vocabs = build_vocab_from_iterator(map(tokenizer,text_read_iterator('/content/wikitext-2/wiki.train.tokens'))
,min_freq=1, specials=["<unk>"])

### 4.2. Model
In this subsection, the architecture and specifics of the deep learning model employed for the segmentation task are presented. It describes the model's layers, components, libraries, and any modifications made to it.

### 4.3. Configurations
This part outlines the configuration settings used for training and evaluation. It includes information on hyperparameters, optimization algorithms, loss function, metric, and any other settings that are crucial to the model's performance.

### 4.4. Train
Here, you'll find instructions and code related to the training of the segmentation model. This section covers the process of training the model on the provided dataset.

### 4.5. Evaluate
In the evaluation section, the methods and metrics used to assess the model's performance are detailed. It explains how the model's segmentation results are quantified and provides insights into the model's effectiveness.
