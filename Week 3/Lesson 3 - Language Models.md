# Language Modeling
### What does it mean to model language?
Language model captures all key properties of real human language simplifying some aspects but preserving overall structure and principles.
Language models estimate the probability of each token being in text.
### Sentence Probability
We can not simply estimate probability of entire sentence. Not only because of how diverse our language is but also because of how we treat unknown sentences. We equally give them 0 probability.
A good solution to the problem is a sentence representation word by word.
<img src="https://lena-voita.github.io/resources/lectures/lang_models/general/i_saw_a_cat_prob.gif">
>> 💡 This is left-to-right language modeling framework

### Count based methods
> **Markov Assumption**: the probability of a word only depends on a fixed number of previous words.

Markov Assumption eliminates older context leaving only n-1-grams of context
![[Pasted image 20250625083155.png]]
But how do we treat zero frequencies?
Our probability is calculated the following way:
$$P(mat | cat,on,a) = {N(cat, on, a, mat) \over N(cat, on, a)}$$
What if "cat on a mat" or "cat on a" occurs 0 times?
One way to address this problem is to reduce n-gram size (n)
But it treats probabilities unfairly giving higher probability to unigram.
Laplace smoothing is also a thing here.
Such count based models generate text, which is... fine, but could be better.
Fixed context window should be kept small to work fine but on practice this leads to veeery short memory of the model.
>> Fixed Context Sucks

### Neural Approach
We have pretty much the same framework except instead of classifying groups (like positive or negative) we aim to classify the correct word from context.
Such models are **discriminative classifiers** because they learn separation.
RNNs are (were) popular for language modeling, then guys started using deeper rnns, lstms and grus.
#### Meeting LM requirements
Our language model is good when it is:
- Coherent - text makes sense
- Diverse - variety of produced samples is high enough
Our answer is **Sampling With Temperature!**
Temperature is simply a division factor before softmax layer.
>Low temperature value - More conservative (greedy sampling - simply picking the most probable class like in argmax)
>High temperature value - Classes get more equal probabilities => More randomness and chaos.
>![[Pasted image 20250625094353.png]]
>![[Pasted image 20250625094431.png]]

Greedy decoding yields a lot of cycles
"The end is never The end is never The end is never The end is never The end is never"

Another approach is **TOP-K Sampling** when we samply only from K most probable tokens.
This way we do not generate ultra rare stuff.
Problem: K choice. For some context K should be small (image top probabilities like 0.45, 0.44, 0.01, ....). Here k of 2 is a good option, but is it good in other cases? Not really, diversity needed.

**Nucleus Sampling (Top-p sampling)**, which samples p% of the probability mass improves over top k.
![[Pasted image 20250625095500.png]]
### Evaluation

Perplexity - measure of uncertainity.
$$Perplexity(y_{1:M}) = 2^{-{1\over M}L(y_{1:M})}$$
The best perplexity score is a perplexity of 1, because it indicates 0 cost over our data.
In other words, model is 100% correctly predicts words. It is impossible score, because it means 0 probability to all tokens except the right one - the right one gets score of 1. 
Worst perplexity is |V| - vocabulary size.
This indicates random prediction where each word is equally probable.
State of the art (SOTA) LLMs have perplexity around 17, which is really good for their large vocabs.
Also, take into account that perplexity of a character-level model is different from perplexity of a token-level model (like LLM), because of the vocabulary size.
### Practical Tips
For small-medium size LMs we can use weight tying
![[Pasted image 20250625102002.png]]
When input word embeddings are equal to output word embeddings and gradient flows through it twice.
> Language Models trained on big datasets learn to understand deeper level ideas.
> For example, it can learn to understand sentiment, because the task somehow required it to predict a next token.

That's all folks! I am not gonna do seminar this week because it's only about n-grams + I want to practice.