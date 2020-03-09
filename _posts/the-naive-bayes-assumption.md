---
layout: article
title: The Naive Bayes Assumtion
tags: [Machine Learning]
mathjax: true
---
The Naïve Bayes classifier is a popular machine learning algorithm. In this post we will discuss why it still works in practice even when the (naïve) conditional independence assumption is violated.
<!--more-->

### Introduction

The Naïve Bayes classifier is a machine algorithm that can be used for classification tasks, similar to the [perceptron algorithm](https://mostafaeissa.github.io/2020/02/26/the-perceptron.html). It is a probabilistic machine learning method which means it relies on probability theory to calculate the probability of a specific class given the input feature. 

For example, consider an input instance $X=(x_1,x_2,…,x_n)$ and output labels $Y \in \{+1,-1\}$, the Naïve Bayes classifier calculates $P(Y=+1|x_1,x_2,…,x_n)$ and $P(Y=-1│x_1,x_2,…,x_n)$ and then picks the class that has a higher probability. To calculate these probabilities, the algorithm relies on Bayes rule which allows the probability defined above to be rewritten. Let $C_k \in \{+1,-1\}$ then:

$$
P(Y=C_k│x_1,x_2,…,x_n ) = \frac{P(x_1,x_2,…,x_n│Y=C_k) P(Y=C_k)}{P(x_1,x_2,…,x_n)}
$$

Now, the naïve conditional independence assumption comes into play. Basically the algorithm  assumes that the features are conditionally independent on the output label $C_k$ and this allows us to rewrite the term  $P(x_1,x_2,…,x_n│Y=C_k)$ as a product of easier to compute terms $P(x_1,x_2,…,x_n│Y=C_k)=\sum \limits_{i=1}^{n} P(x_i|Y=C_k)$.

### What is conditional independence?

Before we go further let’s take a detour and give some intuition on what it means to be *conditionally independent*.  For starters, two events A and B are independent if the occurrence of one event has no effect on the occurrence of the other, in probability terms we say $P(A|B) = P(A)$ if $A$ and $B$ are independent. For example, the probability of the event $A$: a coin will land heads and the event $B$: it is raining outside are independent.

However, conditional independence is always with respect to a third event so we say events $A$ and $B$ are conditionally independent given event $C$, in probability terms we say $P(A|B,C)=P(A|C)$ if $A$ and $B$ are conditionally independent given $C$.

To understand it better let us consider this [nice example](https://www.eecs.qmul.ac.uk/~norman/BBNs/Independence_and_conditional_independence.htm). Suppose that we toss the same coin twice where event $A$ is the outcome of the first toss and event $B$ is the outcome of the second toss, both events are dependent on the bias of the coin (its tendency to produce a certain output more) Now if $C$ is the event that the coin is double sided then once we observe the event $C$ the output of event $B$ has no effect on the outcome of event $A$, hence, events $A$ and $B$ are conditionally independent given event $C$.

### What Happens in practice?

In practice, the Naïve assumption is often violated. For example, trying to predict a plant type based on width, height and color of the leaves. The classifier assumes that all these features are independent without consideration of their effect on each other. In this case the width and height are dependent so the assumption is over estimating the true probability.
Another example, in [text classification](https://nlp.stanford.edu/IR-book/html/htmledition/text-classification-and-naive-bayes-1.html), each classification class $C_k$ considers all possible text values for word $i$ regardless on other words within a small window of that word. 
As such, the conditional independence assumption will often either over estimate or underestimate the probability $P(x_1,x_2,…,x_n│Y=C_k)$ which makes the total probability $P(Y=C_k│X)$ incorrect. ***So where is the catch?***

### Why does it work?

### Conclusion

In this post, we looked at the Naïve Bayes algorithm. We then looked at the conditional independence assumption and why Naïve Bayes works so well in practice.

Hope you enjoyed this post.
<br/>
M.
