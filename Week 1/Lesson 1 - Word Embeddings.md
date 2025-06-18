# Word Embeddings

### Intro
We can not pass text to the machine learning model (it operates with numbers, not words)
Facing some unknown "krakazyabras" we use UNK token to replace all unknown words.
Traditionally UNK is the second row.
### One-Hot vectors
Bad way of managing things. Treats each word as a categorical feature with one-hot operation applied (sucks).
**Problems**: Sparse representation is memory inefficient, no semantic meaning captured.
### Meaning
But what is meaning of a word? How does meaning appear?
**Context!** (Nice thing to do is to debug your own thoughts, observe them and try to project the idea to the numbers)
Our assumption is **similar words appear in similar contexts**
And technically if we manage to capture these context-defined meanings for words we win.
![[Pasted image 20250618142118.png]]
## Count-Based Methods (pre neural network era)
The general principle of count-based representations is captured in their meaning.
We build a matrix where each row represents a word and columns stand for some context correspondence. Intersection captures (with some numerical variable) whether the word fits this context or not.
![[Pasted image 20250618142653.png]]
>> 💡Typically we use SVD to truncate this matrix (Oh, I didn't know about that!)

What's context?
Well... We slide a L-sized window over our corpus calculating how many times this particular word is in window with that particular word and fill the matrix.
![[Pasted image 20250618143428.png]]
In other words, we treat surrounding words as a context forming $V\times V$ matrix.
There's also such thing called PPMI but I need a **further research here**
**Latent Semantic Analysis (LSA)**
Okay, this one is pretty interesting (and also new to me), because we have a corpus of documents of text. We construct a matrix where each row is a word and each column is a document. The intersection of these is a **tf-idf**. What?
>💡tf-idf Explained
>> tf stands for **term frequency** and this is simply the number of times word appears in document
>> idf - **inverse document frequency** and this is computed as $$idf = \log{D \over d_fx}$$
>> The higher the number the more unique and context specific the word is (for example: DNA) and the smaller this score is the more common the word is (for example: he)

**But typically we use it for document summarization (by topics)**
## Prediction Based Methods (Neural Network era)
### Word2Vec
**idea:**
Learn Word vectors by training a model to predict context from words or vice-a-versa.
This task requires some thoughtful contextual representations to emerge, so after training we take those optimized parameters (weight matrix) and use it as a word embedding.
**skip-gram pipeline overview**
We slide a context window over the corpus.
![[Pasted image 20250618150641.png]]
We define a central word and context words. **We learn contexts from central words**
and it's... well... kind of intuitive (Think of a previous count-based $V\times V$ matrix except now you are more flexible with number of context features).
**skip-gram technical details**
Since this is an optimization problem we need to figure out our objective function, which is naturally a **Negative Log-Likelihood**.
Here's likelihood formula:
$$L(\theta) = \prod_{t=1}^T \prod_{-m \le j \le m}P(w_{t+j}|w_t, \theta)$$
Negative log likelihood is naturally:
$$J(\theta) = -{1\over T}\log L(\theta) = -{1\over T} \sum_{t=1}^T \sum_{-m \le j \le m} \log P(w_{t+j}|w_t, \theta)$$
In other words, we maximize the probability of context words in this window being context of central word (because they are).
**What's the probability?**
For each word we have two vector representations for two roles (either as central or a context word)
$v_w$ when the word is a central word
$u_w$ when the word is a context word
We separate these roles because they've different roles.

$$P(o|c) = {\exp(u_o^Tv_c) \over \sum_{w \in V} \exp(u_w^Tv_c)}$$
It is a softmax activated dot-product of contextual representation of a context word c and  a central representation of a central word.
$c$ - central
$o$ - outside
>>⚠ we normalize over entire vocabulary. Otherwise it doesn't make sense.

We increase dot-product for actual (true) context words.
But my god, it's really time consuming. This many updates + softmax over entire vocabulary?!?
Can we optimize it? Absolutely!
**Negative Sampling**
Instead of using all vocabulary we use K random negative words from it.
This simple!
Now instead of optimizing V + 1 vectors we optimize K + 2 vectors where K is a drastically smaller number than V.
![[Pasted image 20250618154103.png]]
It works because our corpus is large (presumably) and typically negative samples achieve enough updates. 
Centrals or Contexts are equally possible to use as embedding. I like the idea of using centrals (green) embeddings more.

**CBOW**
CBOW learns word from context, not context from word. CBOW tends to work better with unique words from what I've heard, but this difference is really subtle and neglectable.
I like skip-gram more because it links us back to count-based methods and feels like a natural step-up.
![[Pasted image 20250618165618.png]]
>🤔 Skip-Gram with negative sampling is default implementation of Word2Vec
>>Negative sampling factor: for small datasets we use 15-20, for huge datasets: 2-5.
>>Embedding dimensionality: frequently set to 300 (but 100 and 50 are also used)
>> Sliding window size: 5-10

>😎 The effect of context window size
>>Larger window: More topical similarities (dog, bark, leash)
>>Smaller window: More functional similarities (walking, running, approaching)

PMI with SVD applied is almost identical in results to Skip-Gram with negative sampling.
### GloVe
GloVe mixes count-based and parametric approaches.
1) Calculate corpus statistics table ($V\times V$)
2) Form same 2 embedding matrices (just like in Word2Vec)
3) Optimize cost function provided below:
$$J(\theta) = \sum_{w, c \in V} f(N(w, c)) \cdot (u_c^T + b_c + \bar{b_w} - \log N(w,c))^2$$
It's MSE loss with weighting function $f$ to penalize rare events and not to over-weight frequent events (we do not really consider optimizing rare events)
![[Pasted image 20250618171426.png]]
remember, N(w, c) - number of times word occurs.
### Evaluation
#### Extrinsic
How useful these embeddings are in our task. Technically we can compare performance of random embeddings and learned embeddings in our task to find out how useful they are. 
#### Intrinsic
Visualization and vector operations to see whether a meaning of a word is captured or not.
## Thinking about questions (from the website)
Okay, so this is the most interesting part because we are pushed to think (really uncommon in courses lol)
### Count-Based Methods
![[Pasted image 20250618191655.png]]
#### Are context words at different distances equally important? If not, how can we modify co-occurrence counts?
> I think they are not equally important. More distant words have a higher chance of being related to the other central word (For example in the first context window example we have a word playing, which is related to the cat, not to the gray.) I do not have a proof for it though, so I am uncertain whether it's a big problem or not. Thinking about theoretically possible, yet completely impractical window of size $V$ it makes total sense, because words closer to the center capture the core meaning and define a word. Final verdict:
> I personally do not know if it has a significant impact on our embeddings, but it definitely is a thing IMO. Simplest approach is to weigh importance of these context words with an inverse proximity to the center. In simpler terms, just subtract context index from center index, take module of it (to avoid negative scores for prior words to the center) and raise to the power of -1. $$1\over |context_i - center_i|$$



### Word2Vec
![[Pasted image 20250618195035.png]]
#### Are all context words equally important?  
Which word types give more/less information than others? Think about some characteristics of words that can influence their importance. Do not forget the previous exercise!
>Okay, the distance inverse scaling idea is natural (IMO, I didn't check my answers) so what we could do is to apply the same thing.
>However I have a strong desire to filter words by their meanings. The idea of attention is there already and works really well, but the things I'm about to suggest are super fucking stupid. Beyond filtering stopwords... It would be great to understand characteristics of each word (damn... I'm naturally coming to attention...) like what part of speech it is etc. etc.
>Well... We could form some sort of a handmade (because parametric solution is attention)



crazy shit to google/research/understand that this shit is too crazy:
denoising words (wtf?) is it possible to apply vae to word embeddings?
