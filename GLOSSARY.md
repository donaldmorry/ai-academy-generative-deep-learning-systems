# Ecosystem Glossary

Click any term in sections to jump here. Terms are grouped by course of first introduction.

---

## Course 1: Applied ML & AI

### A–C

<a id="bag-of-words"></a>
### Bag-of-Words
Text representation that counts word occurrences per document, discarding grammar and word order. See [Section 4.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-04-text-classification/section-03-bag-of-words-and-tf-idf.md).

<a id="accuracy"></a>
### Accuracy
Fraction of predictions that are correct:

$$
\text{Accuracy} = \frac{\text{correct}}{\text{total}}
$$

> **Readable form:** accuracy = number correct divided by total predictions Misleading on imbalanced data. See [Section 3.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-03-classification-models/section-03-classification-accuracy-measures.md).

<a id="classification"></a>
### Classification
[Supervised learning](#supervised-learning) predicting a **discrete category** (spam/ham, digit 0–9). See [Chapter 03](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-03-classification-models/README.md) and [Section 3.1](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-03-classification-models/section-01-classification-fundamentals.md).

<a id="clustering"></a>
### Clustering
[Unsupervised learning](#unsupervised-learning) that groups similar points. Example: [k-means](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-01-machine-learning/section-04-unsupervised-learning-k-means-clustering.md).

<a id="cosine-similarity"></a>
### Cosine Similarity
Similarity measure based on the angle between two vectors: $\cos(\theta) = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$. Length-invariant; used for content-based recommenders. See [Section 4.7](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-04-text-classification/section-07-cosine-similarity-and-recommender-systems.md).

<a id="cross-validation"></a>
### Cross-Validation
Technique that splits data into k folds, trains on k−1, tests on 1, repeats. More reliable than a single train/test split. See [Section 2.7](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-02-regression-models/section-07-model-comparison-and-cross-validation.md).

<a id="anchor-box"></a>
### Anchor Box
Predefined bounding-box template (width, height, aspect ratio) at each feature-map location. Detectors predict **offsets** from anchors rather than absolute coordinates. See [Section 12.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-04-single-shot-detectors-and-yolo.md).

<a id="azure-cognitive-services"></a>
### Azure Cognitive Services
Microsoft's managed AI API portfolio — vision, language, speech, decision — callable via REST/SDK without training custom models. See [Chapter 14](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-14-azure-cognitive-services/README.md).

<a id="azure-custom-vision"></a>
### Azure Custom Vision
Cloud service for training custom **image classifiers** and **object detectors** via web UI and Prediction API. See [Section 12.7](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-07-azure-custom-vision.md).

<a id="azure-language"></a>
### Azure Language
Azure AI service for **sentiment analysis**, **key phrases**, **named entity recognition**, **PII detection**, and custom text classification. See [Section 14.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-14-azure-cognitive-services/section-04-azure-language-services.md).

<a id="azure-speech"></a>
### Azure Speech
Azure AI service for **speech-to-text** and **neural text-to-speech**. See [Section 14.6](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-14-azure-cognitive-services/section-06-azure-speech-services.md).

<a id="azure-translator"></a>
### Azure Translator
Azure neural machine translation API supporting 100+ languages. See [Section 14.5](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-14-azure-cognitive-services/section-05-azure-translator.md).

<a id="bert"></a>
### BERT
**Bidirectional Encoder Representations from Transformers** — pretrained encoder-only [transformer](#transformer) fine-tuned for downstream NLP tasks. Uses `[CLS]` token for classification. See [Section 13.7](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-07-bert-and-transfer-learning-for-nlp.md).

<a id="bleu-score"></a>
### BLEU Score
**Bilingual Evaluation Understudy** — n-gram overlap metric for machine translation quality, range 0–100. See [Section 13.5](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-05-sequence-to-sequence-models.md).

<a id="bounding-box"></a>
### Bounding Box
Axis-aligned rectangle $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ localizing an object in an image. See [Section 12.1](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-01-from-classification-to-detection.md).

<a id="cls-token"></a>
### [CLS] Token
Special BERT input token whose final hidden state summarizes the sequence for **classification** tasks. See [Section 13.7](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-07-bert-and-transfer-learning-for-nlp.md).

<a id="computer-vision-api"></a>
### Computer Vision API
Azure service for image **tagging**, **captions**, **OCR (Read)**, and content moderation. See [Section 14.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-14-azure-cognitive-services/section-03-azure-computer-vision.md).

<a id="confidence-score"></a>
### Confidence Score
Model probability $s \in [0,1]$ that a detection or prediction is correct. Filtered before [NMS](#non-maximum-suppression). See [Section 12.5](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-05-yolov3-in-practice.md).

<a id="contoso-travel"></a>
### Contoso Travel
Prosise Chapter 14 capstone travel assistant chaining Azure Speech, Language, Translator, and Computer Vision. See [Section 14.7](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-14-azure-cognitive-services/section-07-contoso-travel-multi-service-app.md).

### D–G

<a id="deep-learning"></a>
### Deep Learning
[Machine learning](#machine-learning) using neural networks with many layers. Subset of ML, not all of [AI](#artificial-intelligence). See [Section 1.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-01-machine-learning/section-02-machine-learning-vs-ai-vs-deep-learning.md).

<a id="dropout"></a>
### Dropout
[Regularization](#regularization) technique that randomly disables a fraction of neuron connections during training, forcing redundant representations that generalize better. Disabled at inference. See [Section 9.7](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-09-neural-networks/section-07-dropout-and-regularization.md).

<a id="early-stopping"></a>
### Early Stopping
Keras callback that halts training when a monitored metric (e.g., `val_loss`) stops improving for `patience` epochs. With `restore_best_weights=True`, reverts weights to the best epoch. See [Section 9.8](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-09-neural-networks/section-08-callbacks-training-control-and-model-persistence.md).

<a id="epoch"></a>
### Epoch
One complete pass of the entire training dataset through a neural network during `fit()`. See [Section 9.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-09-neural-networks/section-02-your-first-keras-model.md).

<a id="keras"></a>
### Keras
High-level Python API for building and training [neural networks](#neural-network) on [TensorFlow](#tensorflow). `tensorflow.keras` is the recommended import path. See [Chapter 09](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-09-neural-networks/README.md).

<a id="tensorflow"></a>
### TensorFlow
Open-source framework for fast tensor computation at scale, with GPU support and SavedModel export. Keras runs on top of TensorFlow 2.x. See [Section 9.1](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-09-neural-networks/section-01-keras-and-tensorflow-setup.md).

<a id="feature"></a>
### Feature
One input variable (column) used for prediction. "Miles traveled" is a feature for taxi fare.

<a id="gradient-boosting"></a>
### Gradient Boosting
Ensemble that builds trees **sequentially**, each correcting the previous tree's errors. See [Section 2.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-02-regression-models/section-04-ensemble-methods-random-forests-and-gradient-boosting.md).

<a id="embedding-layer"></a>
### Embedding Layer
Keras layer mapping token indices to dense vectors $\mathbf{e}_i \in \mathbb{R}^d$. Trainable or initialized from GloVe/Word2Vec. See [Section 13.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-03-embedding-layers.md).

<a id="encoder-decoder"></a>
### Encoder-Decoder
Sequence-to-sequence architecture: **encoder** compresses input sequence; **decoder** generates output autoregressively. Used in NMT. See [Section 13.5](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-05-sequence-to-sequence-models.md).

<a id="glove"></a>
### GloVe
**Global Vectors** — pretrained word embeddings from co-occurrence statistics. Common initialization for Keras `Embedding` layers. See [Section 13.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-03-embedding-layers.md).

### H–L

<a id="hyperparameter"></a>
### Hyperparameter
Setting chosen **before** training (e.g., k in k-NN, tree depth). Not learned from data. Opposite of **model parameters**.

<a id="kernel-trick"></a>
### Kernel Trick
Technique that computes dot products in a high-dimensional feature space **without explicitly building** the expanded features. Enables nonlinear [SVM](#svm) boundaries via functions like RBF: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$. See [Section 5.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-05-support-vector-machines/section-02-kernels-and-the-kernel-trick.md).

<a id="k-means"></a>
### k-Means
Clustering algorithm minimizing within-cluster variance. See [Section 1.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-01-machine-learning/section-04-unsupervised-learning-k-means-clustering.md).

<a id="k-nearest-neighbors"></a>
### k-Nearest Neighbors (k-NN)
Classifies by majority vote of k closest training points. See [Section 1.5](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-01-machine-learning/section-05-supervised-learning-k-nearest-neighbors.md).

<a id="label"></a>
### Label
The correct output (target) for a training example. "Spam" or "ham" for an email.

<a id="linear-regression"></a>
### Linear Regression
Models target as linear combination of features: $\hat{y} = w_0 + w_1 x_1 + \cdots + w_n x_n$. See [Section 2.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-02-regression-models/section-02-linear-regression.md).

<a id="faster-r-cnn"></a>
### Faster R-CNN
Two-stage detector replacing Selective Search with a **Region Proposal Network (RPN)** on shared CNN features. See [Section 12.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-03-faster-r-cnn-and-mask-r-cnn.md).

<a id="fine-tuning"></a>
### Fine-Tuning
Continuing training of a **pretrained** model (e.g., BERT) on a downstream task with a small learning rate. See [Section 13.7](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-07-bert-and-transfer-learning-for-nlp.md).

<a id="gru"></a>
### GRU (Gated Recurrent Unit)
Simplified gated RNN variant with reset and update gates — fewer parameters than [LSTM](#lstm). See [Section 13.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-04-recurrent-neural-networks.md).

<a id="lstm"></a>
### LSTM (Long Short-Term Memory)
Gated RNN architecture with forget, input, and output gates preserving gradient flow over long sequences. See [Section 13.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-04-recurrent-neural-networks.md).

### M–O

<a id="eigenvector"></a>
### Eigenvector
A nonzero vector $\mathbf{v}$ that only **scales** (does not rotate) when a matrix $\mathbf{A}$ multiplies it: $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$. In [PCA](#pca), eigenvectors of the covariance matrix define principal component directions; eigenvalues $\lambda$ give variance along each axis. See [Section 6.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-06-principal-component-analysis/section-02-pca-theory-covariance-eigenvectors-and-principal-components.md).

<a id="explained-variance"></a>
### Explained Variance
Fraction of total dataset variance captured by a principal component:

$$
\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^{m}\lambda_j}
$$

> **Readable form:** explained variance ratio for component $k$ = that component's eigenvalue divided by the sum of all eigenvalues.

Cumulative explained variance guides component count (scree plots, 95% rule). See [Section 6.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-06-principal-component-analysis/section-04-choosing-the-number-of-components.md).

<a id="machine-learning"></a>
### Machine Learning
Systems that **learn from data** without explicit rules for every case. See [Section 1.1](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-01-machine-learning/section-01-what-is-machine-learning.md).

<a id="mae"></a>
### MAE (Mean Absolute Error)
$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$. Average error in original units.

<a id="model"></a>
### Model
The learned function mapping inputs to outputs. In k-NN, the training set *is* the model.

<a id="naive-bayes"></a>
### Naive Bayes
Probabilistic classifier applying Bayes' theorem with a "naive" independence assumption between features. `MultinomialNB` is a standard baseline for text counts. See [Section 4.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-04-text-classification/section-04-naive-bayes-for-text.md).

<a id="mlops"></a>
### MLOps
**ML operations** — practices and tooling for deploying, monitoring, versioning, and maintaining models in production (model registry, CI/CD, drift detection). Introduced in [Section 7.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-07-operationalizing-models/section-02-model-serialization-with-pickle-and-joblib.md); deepened in Course 2.

<a id="mse"></a>
### MSE (Mean Squared Error)
$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$. Penalizes large errors heavily.

<a id="onnx-open-neural-network-exchange"></a>
<a id="onnx"></a>
### ONNX
**Open Neural Network Exchange** — platform-agnostic model format for cross-language inference via ONNX Runtime (Python, C#, Java, C++, JavaScript). Export sklearn pipelines with `skl2onnx`. See [Section 7.5](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-07-operationalizing-models/section-05-onnx-export-and-inference.md).

<a id="ordinary-least-squares"></a>
### Ordinary Least Squares (OLS)
Fits [linear regression](#linear-regression) by minimizing sum of squared residuals.

<a id="overfitting"></a>
### Overfitting
Model memorizes training data, fails on new data. Like a student who memorizes answers but can't solve new problems.

<a id="pca"></a>
### Principal Component Analysis (PCA)
**Linear dimensionality reduction** that finds orthogonal directions of maximum variance, projects data onto the top $k$ directions, and enables compression, denoising, visualization, anonymization, and anomaly detection via reconstruction error. See [Chapter 06](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-06-principal-component-analysis/README.md) and [Section 6.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-06-principal-component-analysis/section-02-pca-theory-covariance-eigenvectors-and-principal-components.md).

<a id="instance-segmentation"></a>
### Instance Segmentation
Pixel-level labeling that distinguishes **individual object instances** (separate masks per person), unlike semantic segmentation. [Mask R-CNN](#mask-r-cnn) is a standard architecture. See [Section 12.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-03-faster-r-cnn-and-mask-r-cnn.md).

<a id="intersection-over-union-iou"></a>
### Intersection over Union (IoU)
Overlap metric for bounding boxes:

$$
\text{IoU} = \frac{\text{Area}(\text{pred} \cap \text{GT})}{\text{Area}(\text{pred} \cup \text{GT})}
$$

> **Readable form:** IoU = overlap area divided by union area

Used in detection matching and [NMS](#non-maximum-suppression). See [Section 12.1](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-01-from-classification-to-detection.md).

<a id="mask-r-cnn"></a>
### Mask R-CNN
Extends [Faster R-CNN](#faster-r-cnn) with a parallel **mask branch** predicting $28 \times 28$ binary masks per instance. See [Section 12.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-03-faster-r-cnn-and-mask-r-cnn.md).

<a id="mean-average-precision-map"></a>
### Mean Average Precision (mAP)
Detection benchmark metric — mean of per-class **Average Precision** (area under precision-recall curve), often averaged over IoU thresholds (COCO: @0.5:0.95). See [Section 12.8](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-08-choosing-a-detection-architecture.md).

<a id="multi-head-attention"></a>
### Multi-Head Attention
Parallel [self-attention](#self-attention) heads with different learned projections, concatenated and projected — captures diverse token relations. See [Section 13.6](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-06-the-transformer-architecture.md).

<a id="named-entity-recognition-ner"></a>
### Named Entity Recognition (NER)
Extracting typed entities (Location, Person, DateTime) from text. Azure Language and custom models provide NER. See [Section 14.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-14-azure-cognitive-services/section-04-azure-language-services.md).

<a id="neural-machine-translation"></a>
### Neural Machine Translation (NMT)
Sequence-to-sequence [deep learning](#deep-learning) approach to translation. Lab: [Section 13.5](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-05-sequence-to-sequence-models.md); production: [Azure Translator](#azure-translator).

<a id="non-maximum-suppression"></a>
### Non-Maximum Suppression (NMS)
Post-processing that removes duplicate overlapping detections — keeps highest-score box, suppresses neighbors with IoU above threshold. See [Section 12.6](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-06-non-maximum-suppression.md).

<a id="object-detection"></a>
### Object Detection
Computer vision task predicting **bounding boxes** and **class labels** for every object instance in an image. See [Chapter 12](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/README.md).

<a id="ocr-optical-character-recognition"></a>
### OCR (Optical Character Recognition)
Extracting text from images. Azure Computer Vision **Read** API handles scene and document OCR. See [Section 14.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-14-azure-cognitive-services/section-03-azure-computer-vision.md).

<a id="out-of-vocabulary-oov"></a>
### Out-of-Vocabulary (OOV)
Token not in training vocabulary — mapped to `[UNK]` in Keras [TextVectorization](#textvectorization). See [Section 13.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-02-text-preprocessing-with-keras.md).

### R–S

<a id="r-cnn"></a>
### R-CNN
**Regions with CNN features** — pioneering two-stage detector: region proposals → CNN features → SVM classifier. Evolved into Fast/Faster R-CNN. See [Section 12.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-02-r-cnn-and-fast-r-cnn.md).

<a id="rate-limiting"></a>
### Rate Limiting
API quota enforcement — HTTP **429 Too Many Requests** when exceeded. Requires exponential backoff. See [Section 14.8](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-14-azure-cognitive-services/section-08-production-considerations.md).

<a id="recurrent-neural-network-rnn"></a>
### Recurrent Neural Network (RNN)
Neural network maintaining hidden state $\mathbf{h}_t$ across sequence timesteps. Simple RNNs suffer [vanishing gradients](#vanishing-gradient); [LSTM](#lstm)/[GRU](#gru) address this. See [Section 13.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-04-recurrent-neural-networks.md).

<a id="region-proposal"></a>
### Region Proposal
Candidate bounding box that might contain an object — from Selective Search (R-CNN) or [RPN](#region-proposal-network). See [Section 12.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-02-r-cnn-and-fast-r-cnn.md).

<a id="region-proposal-network"></a>
### Region Proposal Network (RPN)
Convolutional subnetwork in [Faster R-CNN](#faster-r-cnn) predicting objectness and box offsets at [anchor](#anchor-box) locations. See [Section 12.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-03-faster-r-cnn-and-mask-r-cnn.md).

<a id="resource-group"></a>
### Resource Group
Azure container organizing related resources (Cognitive Services, storage) for lifecycle and billing management. See [Section 14.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-14-azure-cognitive-services/section-02-azure-setup-and-authentication.md).

<a id="roi-pooling"></a>
### RoI Pooling
**Region of Interest Pooling** — converts variable-size feature-map regions to fixed-size tensors (e.g., $7 \times 7$) for classification heads. See [Section 12.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-02-r-cnn-and-fast-r-cnn.md).

<a id="random-forest"></a>
### Random Forest
Ensemble of decision trees trained on random data/feature subsets, averaged. Reduces [overfitting](#overfitting).

<a id="regression"></a>
### Regression
[Supervised learning](#supervised-learning) predicting a **continuous number** (price, temperature). See [Chapter 02](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-02-regression-models/README.md).

<a id="regularization"></a>
### Regularization
Penalty on model complexity to prevent [overfitting](#overfitting). Ridge (L2) and Lasso (L1) are examples.

<a id="rmse"></a>
### RMSE (Root Mean Squared Error)
$\text{RMSE} = \sqrt{\text{MSE}}$. Same units as target; punishes big mistakes.

<a id="r-squared"></a>
### R² (Coefficient of Determination)
Fraction of variance explained: $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$. 1.0 = perfect; 0 = no better than mean.

<a id="supervised-learning"></a>
### Supervised Learning
Learning from labeled (input, output) pairs. See [Section 1.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-01-machine-learning/section-03-supervised-vs-unsupervised-learning.md).

<a id="svm"></a>
### Support Vector Machine (SVM)
Classifier (or regressor) that finds the **maximum-margin** hyperplane between classes. Predictions depend only on **support vectors** — training points on the margin boundary. Extended to nonlinear data via the [kernel trick](#kernel-trick). See [Chapter 05](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-05-support-vector-machines/README.md) and [Section 5.1](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-05-support-vector-machines/section-01-svm-intuition-maximum-margin-and-support-vectors.md).

<a id="tf-idf"></a>
### TF-IDF (Term Frequency–Inverse Document Frequency)
Weighting scheme that emphasizes terms frequent in a document but rare across the corpus: $\text{tf-idf}(t,d) = \text{tf}(t,d) \times \text{idf}(t)$. See [Section 4.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-04-text-classification/section-03-bag-of-words-and-tf-idf.md).

<a id="tokenization"></a>
### Tokenization
Splitting raw text into tokens (words, punctuation units). First step in most NLP pipelines. See [Section 4.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-04-text-classification/section-02-text-preprocessing.md).

<a id="self-attention"></a>
### Self-Attention
Mechanism where each token attends to all tokens in the same sequence: $\text{softmax}(QK^\top/\sqrt{d_k})V$. Core of [transformers](#transformer). See [Section 13.6](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-06-the-transformer-architecture.md).

<a id="sentiment-analysis"></a>
### Sentiment Analysis
Classifying text polarity (positive/negative/neutral). Classical: [Chapter 04](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-04-text-classification/section-05-sentiment-analysis.md); neural: [Chapter 13](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/README.md); API: [Azure Language](#azure-language).

<a id="single-shot-detector"></a>
### Single-Shot Detector
Object detector predicting all boxes in **one forward pass** (e.g., [YOLO](#yolo)) — no separate proposal stage. See [Section 12.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-04-single-shot-detectors-and-yolo.md).

<a id="speech-to-text"></a>
### Speech-to-Text (STT)
Automatic transcription of audio to text. Azure Speech SDK provides real-time and batch STT. See [Section 14.6](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-14-azure-cognitive-services/section-06-azure-speech-services.md).

<a id="teacher-forcing"></a>
### Teacher Forcing
Seq2seq training technique feeding **ground-truth** previous tokens as decoder input instead of model predictions. See [Section 13.5](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-05-sequence-to-sequence-models.md).

<a id="text-to-speech"></a>
### Text-to-Speech (TTS)
Synthesizing spoken audio from text — Azure **neural voices** with SSML control. See [Section 14.6](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-14-azure-cognitive-services/section-06-azure-speech-services.md).

<a id="textvectorization"></a>
### TextVectorization
Keras layer converting strings to integer sequences (or multi-hot/TF-IDF) for neural NLP pipelines. See [Section 13.2](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-02-text-preprocessing-with-keras.md).

<a id="transfer-learning"></a>
### Transfer Learning
Reusing pretrained model weights (ImageNet CNN, BERT, GloVe) on a new task with fine-tuning or frozen feature extraction. See [Section 13.7](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-07-bert-and-transfer-learning-for-nlp.md).

<a id="transformer"></a>
### Transformer
Attention-based architecture replacing recurrence — parallel token processing via [self-attention](#self-attention) and [multi-head attention](#multi-head-attention). See [Section 13.6](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-06-the-transformer-architecture.md).

### U–Z

<a id="underfitting"></a>
### Underfitting
Model too simple to capture patterns. High error on both train and test.

<a id="unsupervised-learning"></a>
### Unsupervised Learning
Learning from unlabeled data — finding structure. See [Section 1.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-01-machine-learning/section-03-supervised-vs-unsupervised-learning.md).

<a id="vanishing-gradient"></a>
### Vanishing Gradient
Gradients shrinking toward zero when backpropagating through many RNN timesteps — early tokens receive little learning signal. [LSTM](#lstm) gates mitigate this. See [Section 13.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-04-recurrent-neural-networks.md).

<a id="word-embedding"></a>
### Word Embedding
Dense vector representation $\mathbf{e}_w \in \mathbb{R}^d$ capturing semantic relationships — "king" − "man" + "woman" ≈ "queen". See [Section 13.1](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-01-beyond-bag-of-words.md).

<a id="yolo"></a>
### YOLO (You Only Look Once)
**Single-shot** object detector predicting boxes and classes in one network pass via grid-based regression. See [Section 12.4](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-12-object-detection/section-04-single-shot-detectors-and-yolo.md).

<a id="positional-encoding"></a>
### Positional Encoding
Sinusoidal (or learned) vectors added to token embeddings so [transformers](#transformer) know word order without recurrence. See [Section 13.6](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-06-the-transformer-architecture.md).

---

## Course 2: AI Modern Approach

<a id="artificial-intelligence"></a>
### Artificial Intelligence
Study and construction of **rational agents** that perceive and act. Broader than [ML](#machine-learning). See [Chapter 01](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-01-introduction/README.md).

<a id="rational-agent"></a>
### Rational Agent
Entity that selects actions to maximize **expected** [performance measure](#performance-measure), given its [percepts](#percept) and prior knowledge — not omniscience. Central framework of AIMA. See [Section 2.2](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-02-intelligent-agents/section-02-good-behavior-and-rationality.md).

$$
a^* = \arg\max_{a} \mathbb{E}[\text{Performance} \mid a, \text{percepts}, \text{knowledge}]
$$

> **Readable form:** Pick the action with the highest expected score given what you've sensed and what you already know.

<a id="peas"></a>
### PEAS
Framework for specifying a **task environment**: **P**erformance measure, **E**nvironment, **A**ctuators, **S**ensors. Design checklist before choosing agent architecture. See [Section 2.3](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-02-intelligent-agents/section-03-the-nature-of-environments.md).

<a id="percept"></a>
### Percept
The agent's sensory input at one instant, produced by **sensors**. May differ from the true environment state (especially in partially observable settings). See [Section 2.1](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-02-intelligent-agents/section-01-agents-and-environments.md).

<a id="performance-measure"></a>
### Performance Measure
External criterion scoring how well an agent is doing based on **environment states** (not just percepts). Rationality is defined relative to this measure. See [Section 2.2](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-02-intelligent-agents/section-02-good-behavior-and-rationality.md).

<a id="turing-test"></a>
### Turing Test
If a human cannot distinguish machine from human in conversation, machine is "intelligent." Necessary but not sufficient definition.

<a id="search-algorithm"></a>
### Search Algorithm
Systematic exploration of states to find a goal. BFS, UCS, A*, etc. See [Chapter 03](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-03-solving-problems-by-searching/README.md).

<a id="heuristic"></a>
### Heuristic
Function $h(n)$ estimating remaining cost from state $n$ to a goal. Used by informed search to prioritize promising nodes. Quality depends on domain knowledge. See [Section 3.5](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-03-solving-problems-by-searching/section-05-heuristic-functions.md).

<a id="admissible-heuristic"></a>
### Admissible Heuristic
Heuristic that never overestimates true remaining cost: $h(n) \leq h^*(n)$ for all $n$. Required for A* optimality (tree search). Straight-line distance on Romania and Manhattan distance on the 8-puzzle are admissible. See [Section 3.4](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-03-solving-problems-by-searching/section-04-informed-search.md).

<a id="a-star-search"></a>
### A* Search
Best-first search with evaluation $f(n) = g(n) + h(n)$ — path cost so far plus heuristic estimate. Optimal when $h$ is admissible (and consistent for graph search). See [Section 3.4](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-03-solving-problems-by-searching/section-04-informed-search.md).

$$
f(n) = g(n) + h(n)
$$

> **Readable form:** f(n) = cost paid so far + estimated cost remaining.

<a id="local-search"></a>
### Local Search
Algorithms that keep one or few current states and move to neighbors by objective value — no path memory. Used for optimization and CSPs. See [Section 4.1](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-04-search-complex-environments/section-01-local-search.md).

<a id="hill-climbing"></a>
### Hill Climbing
Greedy local search: move to the best neighbor; stop at a local maximum. Variants include stochastic, first-choice, and random-restart. See [Section 4.2](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-04-search-complex-environments/section-01-local-search.md).

<a id="simulated-annealing"></a>
### Simulated Annealing
Local search that sometimes accepts worse moves with probability $e^{-\Delta E/T}$, with temperature $T$ decreasing over time — escapes local optima. See [Section 4.3](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-04-search-complex-environments/section-01-local-search.md).

<a id="genetic-algorithm"></a>
### Genetic Algorithm
Population-based search using selection, crossover, and mutation on encoded solutions. See [Section 4.4](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-04-search-complex-environments/section-01-local-search.md).

<a id="belief-state"></a>
### Belief State
Set (or distribution) over physical states the agent considers possible under partial observability. Search in belief-state space plans despite uncertainty. See [Section 4.7](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-04-search-complex-environments/section-04-partial-observability.md).

<a id="minimax"></a>
### Minimax
Game-tree algorithm: MAX maximizes backed-up utility, MIN minimizes it — optimal play vs. rational opponent. See [Section 5.2](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-05-adversarial-search-games/section-02-minimax.md).

<a id="alpha-beta-pruning"></a>
### Alpha-Beta Pruning
Prunes game-tree branches that cannot change the minimax root value; same result as minimax with fewer node expansions. See [Section 5.3](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-05-adversarial-search-games/section-03-alpha-beta-pruning.md).

<a id="mcts"></a>
### MCTS (Monte Carlo Tree Search)
Select → expand → simulate → backpropagate loop with UCT balancing exploitation and exploration. Powers modern Go and general game AI. See [Section 5.6](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-05-adversarial-search-games/section-06-monte-carlo-tree-search.md).

<a id="csp"></a>
### CSP (Constraint Satisfaction Problem)
Problem defined by variables, domains, and constraints; solution is a consistent complete assignment. See [Section 6.1](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-06-constraint-satisfaction/section-01-csp-formulation.md).

<a id="arc-consistency"></a>
### Arc Consistency
For every value in $X_i$'s domain, some value in $X_j$'s domain satisfies the binary constraint on arc $(X_i, X_j)$. Enforced by AC-3. See [Section 6.3](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-06-constraint-satisfaction/section-03-constraint-propagation.md).

<a id="forward-checking"></a>
### Forward Checking
After assigning a variable, remove inconsistent values from unassigned neighbors' domains — lightweight constraint propagation. See [Section 6.3](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-06-constraint-satisfaction/section-03-constraint-propagation.md).

<a id="knowledge-base"></a>
### Knowledge Base (KB)
Set of sentences representing what an agent knows; updated with TELL, queried with ASK. See [Section 7.1](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-07-logical-agents/section-01-knowledge-based-agents.md).

<a id="entailment"></a>
### Entailment
$KB \models \alpha$ means $\alpha$ is true in every model of $KB$. Semantic basis for logical inference. See [Section 7.3](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-07-logical-agents/section-03-inference-and-entailment.md).

<a id="resolution"></a>
### Resolution
Inference rule: from $(L \lor A)$ and $(\neg L \lor B)$ derive $(A \lor B)$. Complete for propositional refutation. See [Section 7.4](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-07-logical-agents/section-04-theorem-proving.md).

<a id="horn-clause"></a>
### Horn Clause
Clause with at most one positive literal; enables linear-time forward and backward chaining. See [Section 7.5](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-07-logical-agents/section-05-forward-chaining.md).

<a id="forward-chaining"></a>
### Forward Chaining
Data-driven inference on Horn KB: fire rules whose premises are known until query derived. See [Section 7.5](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-07-logical-agents/section-05-forward-chaining.md).

<a id="wumpus-world"></a>
### Wumpus World
Grid cave environment for logical agents: gold, pits, Wumpus; percepts stench/breeze/glitter drive KB inference. See [Section 7.7](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-07-logical-agents/section-07-wumpus-world.md).

<a id="utility"></a>
### Utility
Numeric preference assigned to outcomes so an agent can compare uncertain choices by expected value. Decision-theoretic agents maximize expected utility when actions have probabilistic consequences.

<a id="bayesian-inference"></a>
<a id="posterior"></a>
<a id="prior"></a>
<a id="bayes-rule"></a>
### Bayes' Rule

The fundamental identity for **updating beliefs** when evidence arrives. Also called **Bayes' theorem** or **Bayes' law**.

**Standard form:**

$$
P(h \mid e) = \frac{P(e \mid h)\,P(h)}{P(e)}
$$

> **Readable form (posterior):** Posterior = (likelihood × prior) / evidence.

**Named components:**

| Symbol | Name | Role |
|--------|------|------|
| $P(h \mid e)$ | **Posterior** | Belief in hypothesis $h$ after observing evidence $e$ |
| $P(e \mid h)$ | **Likelihood** | How probable the evidence is if $h$ is true |
| $P(h)$ | **Prior** | Belief in $h$ before seeing $e$ |
| $P(e)$ | **Evidence** (marginal) | Overall probability of observing $e$ |

> **Readable form (likelihood):** Likelihood = how well the hypothesis explains the evidence.

> **Readable form (prior):** Prior = baseline belief before data.

> **Readable form (evidence):** Evidence = normalizer making posteriors sum to 1.

**Alternate forms:**

$$
P(h \mid e) = \alpha \, P(e \mid h)\, P(h), \quad \alpha = \frac{1}{P(e)}
$$

$$
P(h \mid e) = \frac{P(e \mid h)\, P(h)}{\sum_{h'} P(e \mid h')\, P(h')}
$$

**Medical diagnosis example:** Rare disease ($P(\text{disease}) = 1/50{,}000$), symptom in 70% of cases ($P(\text{symptom} \mid \text{disease}) = 0.7$), symptom base rate 1% ($P(\text{symptom}) = 0.01$):

$$
P(\text{disease} \mid \text{symptom}) = \frac{0.7 \times (1/50000)}{0.01} \approx 0.0014
$$

Only **0.14%** of patients with the symptom have the disease — the **base rate** dominates despite a strong likelihood.

**Course 2, Chapter 12 sections:**

- [Section 12.3 — Conditional Probability](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-12-quantifying-uncertainty/section-03-conditional-probability.md) (product rule, chain rule)
- [Section 12.5 — Bayes' Rule Applications](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-12-quantifying-uncertainty/section-05-bayes-rule-applications.md) (diagnosis, sensor fusion)
- [Section 12.6 — Naive Bayes Classifier](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-12-quantifying-uncertainty/section-06-naive-bayes-classifier.md)

Also see [Course 3, Section 3.4](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/section-04-bayes-rule.md).

---

## Course 3: Deep Learning

<a id="eigenvalue"></a>
### Eigenvalue
Scalar $\lambda$ in $A\mathbf{v} = \lambda\mathbf{v}$ — scale factor along eigenvector direction. Covariance eigenvalues equal PCA variances. See [Section 2.6](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-02-linear-algebra/section-06-eigendecomposition.md).

<a id="eigenvector"></a>
### Eigenvector
Nonzero vector $\mathbf{v}$ unchanged in direction by matrix $A$; only scaled by [eigenvalue](#eigenvalue). See [Section 2.6](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-02-linear-algebra/section-06-eigendecomposition.md).

<a id="svd"></a>
### Singular Value Decomposition (SVD)
Factorization $A = U\Sigma V^\top$ for any matrix; truncated SVD gives optimal low-rank approximation. See [Section 2.7](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-02-linear-algebra/section-07-singular-value-decomposition.md).

<a id="singular-value"></a>
### Singular Value
Non-negative diagonal entries $\sigma_i$ in [SVD](#svd); spectral norm equals largest singular value. See [Section 2.7](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-02-linear-algebra/section-07-singular-value-decomposition.md).

<a id="pseudo-inverse"></a>
### Pseudo-inverse (Moore-Penrose)
Generalized inverse $A^+$ via SVD; solves least squares when $A$ is singular or rectangular. See [Section 2.8](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-02-linear-algebra/section-08-pseudo-inverse-and-principal-components.md).

<a id="norm"></a>
### Norm
Function measuring vector or matrix size; L1, L2, Frobenius norms connect to [regularization](#regularization). See [Section 2.5](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-02-linear-algebra/section-05-norms.md).

<a id="entropy"></a>
### Entropy
$H(X) = -\mathbb{E}[\log p(x)]$ — uncertainty in a distribution (bits or nats). See [Section 3.6](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/section-06-information-theory.md).

<a id="kl-divergence"></a>
### KL Divergence
$D_{\mathrm{KL}}(P\|Q) = \mathbb{E}_P[\log\frac{P(x)}{Q(x)}]$ — asymmetric measure of distribution difference; non-negative. See [Section 3.7](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/section-07-kl-divergence-and-cross-entropy.md).

<a id="cross-entropy"></a>
### Cross-Entropy
$H(P,Q) = H(P) + D_{\mathrm{KL}}(P\|Q)$ — standard classification loss when $P$ is empirical labels. See [Section 3.7](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/section-07-kl-divergence-and-cross-entropy.md).

<a id="softmax"></a>
### Softmax
Maps logits $\mathbf{z}$ to a probability vector: $\mathrm{softmax}(z_i) = e^{z_i}/\sum_j e^{z_j}$. Use log-sum-exp for stability. See [Section 4.5](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-04-numerical-computation/section-05-numerically-stable-softmax.md) and [Section 6.7](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-06-deep-feedforward-networks/section-07-softmax-and-multinomial-outputs.md).

<a id="bias-variance"></a>
### Bias-Variance Tradeoff
Decomposition of generalization error into systematic error (bias), sensitivity to training set (variance), and irreducible noise. See [Section 5.3](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-05-machine-learning-basics/section-03-estimation-bias-and-variance.md).

<a id="capacity"></a>
### Capacity
Ability of a model family to fit diverse functions; high capacity risks [overfitting](#overfitting). See [Section 5.2](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-05-machine-learning-basics/section-02-capacity-and-overfitting.md).

<a id="mlp"></a>
### MLP (Multilayer Perceptron)
Feedforward network with one or more hidden layers of nonlinear units. See [Chapter 06](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-06-deep-feedforward-networks/README.md).

<a id="relu"></a>
### ReLU
Rectified Linear Unit: $\max(0, z)$. Default hidden activation in modern networks. See [Section 6.3](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-06-deep-feedforward-networks/section-03-hidden-units.md).

<a id="pmf"></a>
### PMF (Probability Mass Function)
Discrete distribution $p(x)$ with $\sum_x p(x) = 1$. See [Section 3.2](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/section-02-random-variables.md).

<a id="pdf"></a>
### PDF (Probability Density Function)
Continuous density with $\int p(x)\,dx = 1$; probabilities are integrals over intervals. See [Section 3.2](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/section-02-random-variables.md).

<a id="cdf"></a>
### CDF (Cumulative Distribution Function)
$F(x) = P(X \leq x)$. See [Section 3.2](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/section-02-random-variables.md).

<a id="bayesian-statistics"></a>
### Bayesian Statistics
Treats parameters as random variables; learning updates prior to posterior via [Bayes' rule](#bayes-rule). See [Section 5.7](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-05-machine-learning-basics/section-07-bayesian-statistics.md).

<a id="map-estimation"></a>
### MAP Estimation
Maximum a posteriori estimation chooses the parameter value with the largest posterior probability after combining likelihood and prior. It often behaves like maximum likelihood plus a regularization term.

<a id="bagging"></a>
### Bagging
Bootstrap aggregating trains multiple models on resampled datasets and averages or votes their predictions. It reduces variance when individual models make partially independent errors.

<a id="empirical-risk"></a>
### Empirical Risk
Average loss on training set: $\frac{1}{m}\sum_i L(f(\mathbf{x}^{(i)}), y^{(i)})$. Minimized by [gradient descent](#gradient-descent). See [Section 5.1](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-05-machine-learning-basics/section-01-learning-algorithms.md).

<a id="stochastic-gradient-descent"></a>
### Stochastic Gradient Descent (SGD)
[Gradient descent](#gradient-descent) on mini-batches; adds noise that can help generalization. See [Section 5.9](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-05-machine-learning-basics/section-09-stochastic-gradient-descent.md).

<a id="backpropagation"></a>
### Backpropagation
Algorithm computing gradients of loss w.r.t. all parameters via chain rule. See [Section 6.5](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-06-deep-feedforward-networks/section-05-backpropagation-derivation.md).

<a id="gradient-descent"></a>
### Gradient Descent
Iterative optimization: $\theta \leftarrow \theta - \eta \nabla_\theta L$. See [Section 4.3](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-04-numerical-computation/section-03-gradient-based-optimization.md).

<a id="learning-rate"></a>
### Learning Rate
Step-size hyperparameter controlling how far parameters move in response to a gradient update. Too large can diverge; too small can waste computation or stall.

<a id="batch-normalization"></a>
### Batch Normalization
Layer operation that normalizes intermediate activations using batch statistics and then applies learned scale and shift parameters. It often stabilizes and accelerates deep-network training.

<a id="neural-network"></a>
### Neural Network
Composable layers of neurons learning hierarchical [representations](#representation-learning). In Course 1, built with Keras `Dense` layers for tabular and flattened-pixel inputs. See [Chapter 09](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-09-neural-networks/README.md) and [Chapter 06](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-06-deep-feedforward-networks/README.md).

<a id="dense-layer"></a>
### Dense Layer
Fully connected neural-network layer where each output unit receives input from every unit in the previous layer. Dense layers are the default building block for MLPs.

<a id="convolutional-neural-network-cnn"></a>
### Convolutional Neural Network (CNN)
Neural network using convolutional filters, spatial weight sharing, and often pooling to process images and grid-like data efficiently. CNNs power many vision and image-generation models.

<a id="representation-learning"></a>
### Representation Learning
Automatically discovering useful features from raw data. Core idea of [deep learning](#deep-learning). Course 3, Chapter 15.

<a id="condition-number"></a>
### Condition Number
Ratio $\kappa(A) = \|A\|\|A^{-1}\|$ measuring sensitivity to perturbations; large $\kappa$ means [ill-conditioning](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-04-numerical-computation/section-02-ill-conditioning.md). See [Section 4.2](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-04-numerical-computation/section-02-ill-conditioning.md).

<a id="log-sum-exp"></a>
### Log-Sum-Exp Trick
Stable computation of $\log\sum_i e^{z_i}$ by subtracting $\max_i z_i$ before exponentiating. See [Section 4.5](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-04-numerical-computation/section-05-numerically-stable-softmax.md).

<a id="universal-approximation"></a>
### Universal Approximation
Theorem: sufficiently wide single-hidden-layer networks can approximate continuous functions on compact domains. See [Section 6.6](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-06-deep-feedforward-networks/section-06-universal-approximation.md).

<a id="autoencoder"></a>
### Autoencoder
[Neural network](#neural-network) trained to reconstruct input through code $\mathbf{h}=f(\mathbf{x})$, $\hat{\mathbf{x}}=g(\mathbf{h})$. Course 3, [Chapter 14](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-14-autoencoders/README.md).

<a id="elbo"></a>
### ELBO (Evidence Lower Bound)
Variational lower bound on $\log p(\mathbf{v})$: $\mathcal{L}=\mathbb{E}_q[\log p(\mathbf{v},\mathbf{h})-\log q(\mathbf{h})]$. Course 3, [Chapter 19](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-19-approximate-inference/README.md).

<a id="variational-inference"></a>
### Variational Inference
Approximate $p(\mathbf{h}\mid\mathbf{v})$ with tractable $q(\mathbf{h})$ by minimizing $D_{\mathrm{KL}}(q\|p)$. Course 3, Chapter 19.

<a id="partition-function"></a>
### Partition Function
$Z=\sum_{\mathbf{x}}\exp(-E(\mathbf{x}))$ normalizes undirected models; intractable in high dimensions. Course 3, [Chapter 18](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-18-partition-function/README.md).

<a id="contrastive-divergence"></a>
### Contrastive Divergence (CD)
RBM training approximation using short Gibbs chains between data and model statistics. Course 3, Chapters 18–20.

<a id="rbm"></a>
### Restricted Boltzmann Machine (RBM)
Bipartite undirected [generative model](#generative-model) with visible $\mathbf{v}$ and hidden $\mathbf{h}$ layers. Course 3, [Chapter 20](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-20-deep-generative-models/README.md).

<a id="ica"></a>
### Independent Component Analysis (ICA)
Linear factor model recovering statistically independent non-Gaussian sources (blind source separation). Course 3, [Chapter 13](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-13-linear-factor-models/README.md).

<a id="monte-carlo"></a>
### Monte Carlo Method
Estimate $\mathbb{E}[f(\mathbf{x})]$ by averaging over random samples. Course 3, [Chapter 17](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-17-monte-carlo-methods/README.md).

<a id="mcmc"></a>
### MCMC (Markov Chain Monte Carlo)
Construct Markov chains whose stationary distribution is the target $p(\mathbf{x})$; includes Metropolis-Hastings and Gibbs. Course 3, Chapter 17.

<a id="ctc-loss"></a>
### CTC Loss (Connectionist Temporal Classification)
Aligns variable-length label sequences to frame-level predictions without explicit segmentation. Used in speech recognition. Course 3, [Section 12.3](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-12-applications/section-03-speech-recognition.md).

<a id="bayesian-probability"></a>
### Bayesian Probability
Interpretation of probability as a **degree of belief** on $[0,1]$, used when events are not repeatable (e.g., medical diagnosis). Contrasts with **frequentist probability** (long-run relative frequency). Both obey the same axioms per Ramsey (1926). See [Section 3.1](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/section-01-why-probability.md).

<a id="bayesian-network"></a>
### Bayesian Network
Directed acyclic graphical model factorizing $p(\mathbf{x})=\prod_i p(x_i\mid \mathrm{Pa}(x_i))$. Course 3, [Chapter 16](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-16-structured-probabilistic-models/README.md).

<a id="markov-random-field"></a>
### Markov Random Field (MRF)
Undirected graphical model with energy $E(\mathbf{x})$ and $p(\mathbf{x})\propto\exp(-E(\mathbf{x}))$. Course 3, Chapter 16.

---

## Course 4: Generative Deep Learning

<a id="discriminative-model"></a>
### Discriminative Model
Models $P(y|x)$ — boundary between classes. "Is this a cat?"

<a id="generative-model"></a>
### Generative Model
Models $P(x)$ or $P(x|z)$ — can **create** new samples. See [Section 1.1](./chapters/chapter-01-generative-modeling/section-01-what-is-generative-modeling.md).

<a id="latent-variable"></a>
### Latent Variable
Hidden factor $z$ generating observed data $x$. The "compressed essence" of an image or sentence.

<a id="autoregressive-model"></a>
### Autoregressive Model
Generative model that factorizes a joint distribution into an ordered product of conditional probabilities, predicting each next token, pixel, or value from previous ones.

<a id="normalizing-flow"></a>
### Normalizing Flow
Generative model built from invertible transformations with tractable Jacobian determinants, allowing exact likelihood and efficient sampling.

<a id="maximum-likelihood-estimation"></a>
### Maximum Likelihood Estimation (MLE)
Choose parameters $\theta$ that maximize $P(\text{data}|\theta)$. See [Section 3.9](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/section-09-maximum-likelihood.md) and [Section 5.6](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-05-machine-learning-basics/section-06-maximum-likelihood-estimation.md).

<a id="variational-autoencoder"></a>
### Variational Autoencoder (VAE)
Generative model learning latent distribution. Course 4, Chapter 03.

<a id="generative-adversarial-network-gan"></a>
<a id="gan"></a>
### GAN (Generative Adversarial Network)
Generator vs discriminator game. Course 4, Chapter 04.

<a id="diffusion-model"></a>
### Diffusion Model
Learns to reverse gradual noise addition. Powers Stable Diffusion. Course 4, Chapter 08.

<a id="ddpm"></a>
### DDPM
Denoising Diffusion Probabilistic Model: a diffusion model that learns to reverse a fixed Gaussian noising process step by step. See [Course 4, Chapter 08](./chapters/chapter-08-diffusion-models/README.md).

<a id="latent-diffusion"></a>
### Latent Diffusion
Diffusion performed in a compressed VAE latent space rather than directly in pixel space, reducing compute while preserving high-resolution synthesis. See [Stable Diffusion](./chapters/chapter-13-multimodal-models/section-06-stable-diffusion.md).

<a id="stable-diffusion"></a>
### Stable Diffusion
Text-to-image system combining a text encoder, latent diffusion U-Net, scheduler, and VAE decoder. See [Section 13.6](./chapters/chapter-13-multimodal-models/section-06-stable-diffusion.md).

<a id="classifier-free-guidance"></a>
### Classifier-Free Guidance
Sampling technique that combines conditional and unconditional denoising predictions to trade off prompt adherence against diversity in diffusion models.

<a id="clip"></a>
### CLIP
Contrastive Language-Image Pretraining model that embeds images and text into a shared space. Used for retrieval, guidance, and multimodal alignment. See [Section 13.3](./chapters/chapter-13-multimodal-models/section-03-clip-deep-dive.md).

<a id="dalle-2"></a>
### DALL-E 2
Text-to-image architecture using CLIP-style text/image representations plus a prior and decoder stack. See [Section 13.4](./chapters/chapter-13-multimodal-models/section-04-dalle-2-prior-and-decoder.md).

<a id="imagen"></a>
### Imagen
Text-to-image system emphasizing large frozen language encoders and cascaded diffusion super-resolution. See [Section 13.5](./chapters/chapter-13-multimodal-models/section-05-imagen.md).

<a id="flamingo"></a>
### Flamingo
Vision-language model family that connects visual encoders to language models for few-shot multimodal generation. See [Section 13.7](./chapters/chapter-13-multimodal-models/section-07-flamingo.md).

<a id="stylegan"></a>
### StyleGAN
GAN family using style vectors, mapping networks, and progressive image synthesis controls for high-quality image generation. See [Section 10.3](./chapters/chapter-10-advanced-gans/section-03-stylegan-architecture.md).

<a id="realnvp"></a>
### RealNVP
Normalizing-flow architecture using affine coupling layers for exact likelihood and invertible sampling. See [Section 6.3](./chapters/chapter-06-normalizing-flow-models/section-03-realnvp-architecture.md).

<a id="energy-based-model"></a>
### Energy-Based Model (EBM)
Model that assigns low energy to plausible samples and high energy to implausible samples, often requiring approximate sampling. See [Course 4, Chapter 07](./chapters/chapter-07-energy-based-models/README.md).

<a id="flow-matching"></a>
### Flow Matching
Generative modeling approach that learns a continuous vector field transporting simple noise distributions into data distributions. It is related to diffusion and continuous normalizing flows.

<a id="consistency-model"></a>
### Consistency Model
Generative model trained so points along a diffusion trajectory map consistently to the same clean sample, enabling fewer-step sampling.

<a id="diffusion-transformer"></a>
### Diffusion Transformer (DiT)
Diffusion architecture replacing or augmenting U-Net denoisers with transformer blocks, especially for latent image generation.

<a id="gpt"></a>
### GPT
Generative Pretrained Transformer: decoder-only autoregressive language model trained to predict the next token. See [Course 4, Chapter 09](./chapters/chapter-09-transformers/README.md).

<a id="transformer"></a>
### Transformer
Neural architecture based on self-attention, feed-forward blocks, residual connections, and positional information. See [Course 4, Chapter 09](./chapters/chapter-09-transformers/README.md).

<a id="causal-language-modeling"></a>
### Causal Language Modeling
Training objective that predicts the next token from previous tokens only, using causal masking to prevent future-token leakage.

<a id="instruction-tuning"></a>
### Instruction Tuning
Fine-tuning a language model on instruction-response examples so it follows user tasks more reliably.

<a id="rlhf"></a>
### RLHF
Reinforcement Learning from Human Feedback: alignment method using human preference data, a reward model, and policy optimization to improve helpfulness or safety.

<a id="dpo"></a>
### DPO
Direct Preference Optimization: preference-optimization method that trains from chosen/rejected answer pairs without a separate reward-model reinforcement loop. See [Section 14.9](./chapters/chapter-14-conclusion/section-09-modern-llm-engineering.md).

<a id="peft"></a>
### PEFT
Parameter-Efficient Fine-Tuning: adapting a pretrained model by training a small subset of parameters or adapter weights instead of all model weights.

<a id="lora"></a>
### LoRA
Low-Rank Adaptation: PEFT method that freezes base weights and trains low-rank update matrices. See [Section 14.9](./chapters/chapter-14-conclusion/section-09-modern-llm-engineering.md).

<a id="qlora"></a>
### QLoRA
Quantized LoRA: LoRA fine-tuning with the frozen base model loaded in low precision to reduce memory requirements. See [Section 14.9](./chapters/chapter-14-conclusion/section-09-modern-llm-engineering.md).

<a id="adapter"></a>
### Adapter
Small trainable chapter inserted into or alongside a pretrained network to specialize behavior without retraining the full model.

<a id="retrieval-augmented-generation-rag"></a>
<a id="rag"></a>
### RAG (Retrieval-Augmented Generation)
Pattern that retrieves relevant source documents at query time and conditions the model on that context before generating an answer. See [Section 14.9](./chapters/chapter-14-conclusion/section-09-modern-llm-engineering.md).

<a id="vector-database"></a>
### Vector Database
Index optimized for nearest-neighbor search over embedding vectors, commonly used in RAG systems.

<a id="embedding-model"></a>
### Embedding Model
Model that maps text, images, or other objects into vectors where semantic similarity corresponds to geometric closeness.

<a id="reranker"></a>
### Reranker
Model that reorders retrieved candidates using a more expensive relevance model after a fast first-stage search.

<a id="prompt-injection"></a>
### Prompt Injection
Attack where user or retrieved text tries to override system instructions, exfiltrate hidden context, or misuse tools.

<a id="llm-as-judge"></a>
### LLM-as-Judge
Evaluation pattern where a language model grades outputs. Useful for triage, but it must be calibrated against human labels and failure cases.

<a id="evaluation-harness"></a>
### Evaluation Harness
Repeatable test suite for model behavior, including prompts, expected properties, metrics, safety cases, and regression checks.

<a id="kv-cache"></a>
### KV Cache
Stored attention keys and values from previous tokens during autoregressive decoding, used to avoid recomputing the entire prefix. See [Section 14.9](./chapters/chapter-14-conclusion/section-09-modern-llm-engineering.md).

<a id="quantization"></a>
### Quantization
Representing model weights or activations with fewer bits to reduce memory and speed inference, sometimes with a small quality tradeoff.

<a id="speculative-decoding"></a>
### Speculative Decoding
Inference acceleration where a smaller draft model proposes tokens and a larger model verifies them in batches.

<a id="flashattention"></a>
### FlashAttention
Efficient attention algorithm family that reduces memory traffic by computing attention in tiled kernels instead of materializing the full attention matrix.

<a id="long-context"></a>
### Long Context
Ability of a model or system to process long prompts. Useful context length depends on retrieval, position robustness, memory, and latency, not only the advertised token limit.

<a id="rope"></a>
### RoPE
Rotary Position Embedding: positional method that rotates query/key vectors by position-dependent angles and supports several long-context scaling variants.

<a id="alibi"></a>
### ALiBi
Attention with Linear Biases: positional method that adds distance-based attention penalties, often discussed in long-context model design.

<a id="mixture-of-experts-moe"></a>
<a id="moe"></a>
### Mixture of Experts (MoE)
Architecture that routes tokens through a subset of expert networks, increasing total parameters while activating only part of the model per token. See [Section 14.9](./chapters/chapter-14-conclusion/section-09-modern-llm-engineering.md).

<a id="agentic-llm"></a>
### Agentic LLM
LLM system that can plan, call tools, observe results, and iterate toward a goal. Connects modern LLM practice to Course 2 rational agents.
