---
layout: article
title: The Naive Bayes Assumtion
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


