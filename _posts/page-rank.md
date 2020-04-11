---
layout: article
title: Page Rank Algorithm
tags: [Machine Learning]
mathjax: true
mathjax_autoNumber: true
---
PageRank is an algorithm used to rank web pages in search engines results. It was named after, one of the founders of Google, Larry Page.

<!--more-->

### Introduction

PageRank is an algorithm used to rank web pages that relies on two ideas:

1.  counting the number of in-links a page has where the more in-links the higher the rank.  
2.  In-links from pages having a high rank should be worth more than in-links from pages having a low rank.

This definition seems recursive because if a page A is linked to by three other pages B,C,D then in order to determine the PageRank of page A we need to also know the PageRank of B,C,D. As such, each page distributes its PageRank score equally on its out-links, the more out-links a page has, the less the share of each link. In our earlier example, the PageRank $PR(A) = \frac{PR(B)}{L(B)}+\frac{PR(C)}{L(C)}+\frac{PR(D)}{L(D)}$ where $L(x)$ is the number of out links from page $x$. 

More generally, we can express page rank as :
 
$$PR(p) = \sum \limits_{q \in In(p)} \frac{PR(q)}{L(q)}$$
 
where $In(p)$ is the set of in-links to page $p$. 

We can think of the problem as a surfer who follows the links randomly from a given page. However, the problem with this definition is that some pages do not have any out-links (a sink page) which can make our algorithm get stuck and that is why we need a better model.

### The Random Surfer Model

Similar to how most people surf the web, on a given web page you choose to follow one of the links on the links on the page. If you reach a page without any links (a sink page) then you simply randomly restart on a new page on the web Furthermore, every once in a while, you get bored of following the link and you decide to jump to a totally new page.

In mathematical notation, let $c$ be the probability that the surfer will continue following the links while $(1-c)$ is the probability that the surfer will get bored and jump to a random page. A good value for $c = 0.85.$ This time PageRank score is a combination of the contribution of PageRank scores of the in-links and the contribution of randomly landing on this page when the surfer is bored.

$$
\begin{equation}
\label{page rank}
PR(p) = c \times \sum \limits_{q \in In(p)} \frac{PR(q)}{L(q)} + (1-c) \times \frac{1}{N}
\end{equation}
$$

We can think of the PageRank score $PR(p)$ of a page $p$ as the probability that a random surfer will be at page $p$ at a given point in time. In this interpretation, the PageRank score will be between 0 and 1 and since the surfer has to be on some page at any given time and the sum of the PageRank scores of all the pages sums to 1. 

The Algorithm is as follows:

1. Assign each node an initial page rank
2. repeat until convergence: calculate the page rank of each node using equation  $\eqref{page rank}$

### Worked Example

Consider [this example](https://www.cs.princeton.edu/~chazelle/courses/BIB/pagerank.htm), where we have four web pages pointing to each other as shown in the figure below. Intuitively, Page C has 3 in-links so it should have the highest PageRank and page D should be last because it does not have any in-links.


