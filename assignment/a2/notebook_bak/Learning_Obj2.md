### Assignment 2: Text Classification Write-Up

This report summarizes your work on the text classification assignment using Deep Averaging Networks (DAN), Convolutional Neural Networks (CNN), and BERT.

---

### 1. Submitted Answers

Here is a consolidated list of the answers you provided in the `answers` file.

#### **Part 1: Neural Network DAN Text Classification**

* [cite_start]**1.a (Keras Functional API):** True [cite: 7]
* [cite_start]**1.1.a (Train Set Positive %):** 0.49845 [cite: 9]
* [cite_start]**1.1.b (Test Set Positive %):** 0.5026 [cite: 10]
* [cite_start]**1.2.a (Validation Accuracy, Unshuffled):** 1.0 [cite: 12]
* [cite_start]**1.2.b (Validation Accuracy, Shuffled):** 0.63574 [cite: 13]
* [cite_start]**1.3.a (Static Word2Vec Accuracy):** 0.75700 [cite: 15]
* [cite_start]**1.3.b (Fine-tuned Word2Vec Accuracy):** 0.79299 [cite: 16]
* [cite_start]**1.3.c (Randomly Initialized Accuracy):** 0.78200 [cite: 17]

#### **Part 2: CNN Text Classification**

* [cite_start]**2.1.a (Val Loss, Epoch 1):** 0.46626 [cite: 21]
* [cite_start]**2.1.b (Val Loss, Epoch 2):** 0.42402 [cite: 23]
* [cite_start]**2.1.c (Val Loss, Epoch 3):** 0.41153 [cite: 25]
* [cite_start]**2.1.d (Val Loss, Epoch 4):** 0.40233 [cite: 27]
* [cite_start]**2.1.e (Val Loss, Epoch 5):** 0.40067 [cite: 29]
* [cite_start]**2.1.f (num_filters):** `[64, 64, 32]` [cite: 30]
* [cite_start]**2.1.g (kernel_sizes):** `[3, 4, 5]` [cite: 31]
* [cite_start]**2.1.h (dense_layer_dims):** `[64]` [cite: 32]
* [cite_start]**2.1.i (dropout_rate):** 0.6 [cite: 33]
* [cite_start]**2.1.j (embeddings_trainable):** False [cite: 34]
* [cite_start]**2.1.k (batch_size):** 64 [cite: 35]

#### **Part 3: BERT Text Classification**

* [cite_start]**3.1.a (Attention Mask Zeros):** For the first example 4 positions are padded while for the second one it is only one. [cite: 38]
* [cite_start]**3.1.b (Number of Outputs):** 2 [cite: 39]
* [cite_start]**3.1.c (Output for Token-level Embeddings):** the first [cite: 41]
* [cite_start]**3.1.d (Input ID for 'bank'):** 3085 [cite: 42]
* [cite_start]**3.1.e (Index of 'bank' in Sentence 1):** 2 [cite: 43]
* [cite_start]**3.1.f (Index of 'bank' in Sentence 2):** 4 [cite: 44]
* [cite_start]**3.1.g (Cosine Similarity of 'bank'):** 0.74783 [cite: 45]
* [cite_start]**3.1.h (Cosine Similarity of 'this'/'the'):** 0.81103 [cite: 46]
* [cite_start]**3.2.a (RoBERTa Checkpoint Name):** `"cardiffnlp/twitter-roberta-base-sentiment-latest"` [cite: 48]
* [cite_start]**3.2.b (Tweets Trained On):** `~124M` [cite: 49]
* [cite_start]**3.2.c (Paper Title):** `"TimeLMs: Diachronic Language Models from Twitter"` [cite: 50]
* [cite_start]**3.2.d (RoBERTa Validation Accuracy):** 0.896 [cite: 51]
* [cite_start]**3.2.e (RoBERTa vs. BERT-base):** RoBERTa better [cite: 53]
* [cite_start]**3.3.a (Low-level Unfrozen Accuracy):** 0.80800 [cite: 55]
* [cite_start]**3.3.b (Better Model Train Loss):** 0.28260 [cite: 56]
* [cite_start]**3.3.c (Better Model Val Loss):** 0.32886 [cite: 57]
* [cite_start]**3.3.d (Better Model Loss Ratio):** 0.85934 [cite: 59]
* [cite_start]**3.3.e (Better Model Val Accuracy):** 0.85900 [cite: 60]
* [cite_start]**3.3.f (Overfitting Model Train Loss):** 0.14960 [cite: 61]
* [cite_start]**3.3.g (Overfitting Model Val Loss):** 0.38383 [cite: 62]
* [cite_start]**3.3.h (Overfitting Model Loss Ratio):** 0.38976 [cite: 64]
* [cite_start]**3.3.i (Overfitting Model Val Accuracy):** 0.86460 [cite: 65]

---

### 2. Summary of Observations

#### **Notebook 1: Deep Averaging Networks (DAN)**
Your work in this notebook demonstrated a solid grasp of fundamental deep learning concepts for NLP.

* **Data Shuffling:** You correctly observed the critical importance of shuffling your training data. By training on sorted data, the validation set contained only positive examples, leading to a misleading **100% accuracy**. In contrast, shuffling produced a more realistic accuracy of **~64%**, highlighting how a representative validation set is essential for true model evaluation.
* **Embedding Strategies:** Your experiments showed that **fine-tuning pre-trained Word2Vec embeddings** yielded the best performance (**~79%** accuracy). This was slightly better than training randomly initialized embeddings from scratch (**~78%**) and significantly better than using static, frozen embeddings (**~76%**). This confirms that allowing embeddings to adapt to the specific task is highly beneficial.

#### **Notebook 2: Convolutional Neural Networks (CNN)**
This section focused on a common challenge in deep learning: overfitting. Your approach to mitigating it was very effective.

* **Identifying Overfitting:** The initial CNN model clearly overfit, with the validation loss increasing while the training loss decreased.
* **Applying Regularization:** You successfully reduced overfitting by implementing several key regularization strategies:
    * **Reducing Model Complexity:** You decreased the number of filters and dense layers.
    * **Increasing Dropout:** You raised the dropout rate to 0.6.
    * **Freezing Embeddings:** You kept the embeddings non-trainable, which reduces the number of parameters the model needs to learn.
* **Results:** Your adjusted model showed excellent convergence, with the validation loss consistently decreasing from **0.466** to **0.400** over five epochs. This demonstrates a strong practical understanding of how to build more robust and generalizable models.

#### **Notebook 3: BERT**
In this final notebook, you explored the power and nuances of large pre-trained transformer models.

* **Contextual Embeddings:** You demonstrated that BERT produces contextual embeddings. The cosine similarity for the word "bank" in different contexts (**0.748**) was lower than for two function words ("this"/"the") with similar roles (**0.811**). This is a key insight: BERT understands that a word's meaning changes with its context.
* **Transfer Learning:** You correctly identified a more specialized model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) and showed that it outperformed the general `bert-base-cased` model (**~90%** vs. **~86%** accuracy). This highlights the power of using models that have already been fine-tuned on a similar task.
* **Layer Freezing:** Your experiments with layer freezing were insightful. Unfreezing only the **top layers** of BERT (`encoder.layer.10` and `11`) provided the best balance of performance and overfitting control (loss ratio **~0.86**). In contrast, unfreezing all layers led to significant overfitting (loss ratio **~0.39**), showing that strategic fine-tuning is crucial for adapting these large models effectively.

---

### 3. Overall Learning Objectives

Across all three notebooks, this assignment was designed to teach several core concepts in modern NLP and deep learning.

* **Model Progression and Complexity:** The assignment took you on a journey from a strong baseline (DAN) to more sophisticated architectures. This progression illustrates how NLP models have evolved to better capture linguistic features:
    * **DAN:** Averages all word embeddings, treating the text as a "bag of embeddings."
    * **CNN:** Captures local patterns and n-gram features through its convolutional filters.
    * **BERT:** Uses attention to understand the relationships between all words in a sequence, creating deep contextual representations.

* **The Power of Embeddings:** A central theme was the importance of word embeddings. You saw firsthand that **contextual**, **fine-tuned embeddings** from large pre-trained models like BERT and RoBERTa deliver state-of-the-art performance, significantly outperforming non-contextual embeddings like Word2Vec.

* **Tackling Overfitting:** You practiced essential skills for building robust models. The assignment demonstrated that overfitting is a constant challenge, but it can be controlled through a variety of techniques, including **dropout**, **reducing model size**, and, most powerfully, **strategic layer-freezing** during fine-tuning.

