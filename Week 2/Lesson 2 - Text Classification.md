# Text Classification

### Classification: Generative and Discriminative
Generative models: Learn data distribution taking prior probability into account * x probability given y
Discriminative models: Learn class separating boundary
![[Pasted image 20250620105244.png]]
(Yeah, generative means different thing here.)
What's crazy is that a lot of modern **generative models** are actually based on discriminative models. bruh.
### Classical Methods
#### Naive Bayes
Naive Bayes treats words as a bag of words and independently (this is why it's naive).
For each word it calculates a probability of it occurring in any class (positive and negative sentiment etc.)
To predict sentiment of a sentence it multiples probability of word occurring in positive sentiment over a probability of word occurring in negative sentiment.
To prevent division by zero we apply Laplacian smoothing and add 1 to every word count (meaning we saw a word at least once). Don't forget about scaling it by prior probability!
If our product is >1 it's a positive sentiment, otherwise it's negative.
To prevent numerical underflow working with probabilities we take logarithm of them.
Problem simplifies to the sum of logarithm (because log of product is a sum of logs).
If our score is > 0 it's positive, otherwise it's negative (well, technically 0 is neutral)
**Since we learn (actually naive bayes doesn't learn anything iteratively, it simply builds distribution from the data, but anyway)  distribution of our data from labels (we calculate probability of each word occurring in positive class) this algorithm is GENERATIVE**
>>🤔 Naive Bayes is well interpretable, non-parametric but prone to sarcasm and other adversarial attacks. (Naive Bayes is that one boring dude with no sense of humor)

Features for Naive Bayes are simply tokens.
To use Naive Bayes for multiclass classification we should calculate log-probabilities, sum them up (for each class) and pick argmax.

#### Logistic Regression
For logistic regression we learn how positive (or negative) our word is (I am using sentiment analysis because it's a really simple example). We minimize cross-entropy loss between our target score (1 or 0) and predicted score (let's say 0.63)
**Logistic regression is a Discriminative model because it learns positive/negative boundary**
![[Pasted image 20250620113813.png]]
For multiclass classification we can use Softmax Regression.
#### SVM
Simply SVM taking features.
![[Pasted image 20250620114132.png]]

### Neural Methods
#### General Pipeline
This is simple on the level of idea.
We pass learned word embeddings as input to the neural network (let's say it's a simple Feed-Forward network for now), do as many transformations as we want to extract features and finish this network with a linear layer and softmax/sigmoid activation.
We use Cross-Entropy loss as an objective.
We can replace Neural network with anything else.
Let's see what we can use!
#### BOE and Weighted BOE
BOE - Bag of Embeddings.
This is a silly idea of just summing up embeddings of all words and using it as a feature vector for the final linear-softmax layer.
Weighted BOE does nothing, but introduces weights. It's more interesting, we can either learn or set weights manually as tf-idf.
![[Pasted image 20250620120006.png]]
#### Convolutional Neural Networks
This is more interesting now. We use 1D convolution patch to search for patterns in text.
Here's how it may look:
![[Pasted image 20250620120656.png]]
How do we perform 1D convolution, if we have a matrix? Technically 3 word embedding form a matrix, but we **concatenate** these vectors.
![[Pasted image 20250620121157.png]]
Cool! But what is pooling? What if we have different sequence length?
It is an ordinary max pooling for all the features. Simple as that.
![[Pasted image 20250620121512.png]]
Text length (horizontal axis number of columns on this scheme) doesn't matter, because we end up forming a one vector!
This is called a **Global Pooling** because we end up with 1 single vector from the entire sequence. We do not pool to form groups, we pool to summarize it, to pool everything.
This is insane! We can either train multiple convnets and concatenate resulting vectors or we can train a deeper CNN with intermediate pooling (not global) and final representation.
#### RNNs
We feed entire sequence to the RNN word-by-word and take the final output as our entire sequence encoding. This is a simplest approach.
![[Pasted image 20250620122638.png]]
To hold longer context we can either use LSTM or use a bidirectional RNN.
To do so, we take final outputs from both RNNs (one is moving from left to right while other is moving from right to left)
### Practical Tips
#### Word Embeddings
There are 3 ways to deal with word embeddings here:
1) Learn embeddings simultaneously during model training
2) Take Pretrained embeddings (from Word2Vec or GloVe)
3) Initialize with pretrained and fine-tune
3rd approach is the best IMO because it starts with general understanding of words (e.g python - snake) but during fine-tuning on our smaller dataset we adjust meaning to fit our task (e.g python - programming language)
Also we fine tune antonyms, because in embeddings they are close (similar context)
We tune it to fix it.
#### Data Augmentation
**Word Dropout** - replace some words with UNK randomly
**Synonym Replacement** - replace words with their synonyms (requires dictionary)
**Rephrasing** - translate translation of the word (perform identity transformation)