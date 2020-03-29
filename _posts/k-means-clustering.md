---
layout: article
title: K-means clustering
tags: [Machine Learning]
mathjax: true
---
K-means is a clustering algorithm that is relatively old yet it still in use today because of its simplicity and power.
<!--more-->

## Introduction

Unlike supervised learning algorithm, in unsupervised learning algorithms there is no notion of a label. We are simply given a set of input data and we want to conclude something from it. One idea is clustering. In this post we will focus on K-means algorithm.

In layman terms, clustering is grouping a collection of unlabeled data into similar sets. The resulting clusters will be vastly affected by how you decide if two points are similar to each other. Perhaps the simplest and most obvious measure of similarity is proximity where close by data points should belong to the same cluster.

K-means applies this idea in an iterative way, it starts with a random set of k-points $m_1, m_2, …, m_k$ that act as the centroid of the $k$ clusters to be determined (hence the name k-means) and it proceeds in two steps:

- *Assignment step*: each point in the data set is assigned to the closest cluster based on the Euclidean distance to the cluster centroid. Mathematically speaking, the point $x$ is assigned to the cluster $C_i$ so that each cluster has a set of points $S_i$ such that $\lvert\lvert {x – m_i} \rvert\rvert \leq  \lvert\lvert x – m_j \rvert\rvert \enspace \forall \enspace 1 \leq j \leq k$

- *Update step*: recalculate the cluster centroid based on the new assigned points $m_i=\frac{1}{|S_i |}\sum \limits_{x_j \in S_i} x_j$

The algorithm keeps iterating until no more changes happen in data point assignment and centroid locations.

## Worked Example

Suppose we are given the following dataset in R2 and we want to cluster the data into k=2 clusters.

![k-means example](/assets/images/k-means-clustering/example-iteration1.PNG)

The first step is to pick 2 random initial locations for the cluster centroids. Let $m_1 = (0, 0.5)$ and $m_2 = (0.5, 0.5)$.

**Iteration 1:**

*Assignment step:*

|Distance to Centroid |$m1 (0, 0.5)$ |$m2 (0.5, 0.5)$|
|---|---|--|
$P1 (0, 0)$ | $\sqrt{(0-0)^2+(0-0.5)^2}=\bm{0.5}$ |	$\sqrt{(0-0.5)^2+(0-0.5)^2}=0.707$
$P2 (1,0)$ |	$\sqrt{(1-0)^2+(0-0.5)^2}=1.118$	 | $\sqrt{(1-0.5)^2+(0-0.5)^2}=\bm{0.707}$
$P3 (0, 1)$	| $\sqrt{(0-0)^2+(1-0.5)^2}=\bm{0.5}$ | 	$\sqrt{(0-0.5)^2+(1-0.5)^2}=0.707$
$P4 (1,1)$| 	$\sqrt{(1-0)^2+(1-0.5)^2}=1.118$	 | $\sqrt{(1-0.5)^2+(1-0.5)^2}=\bm{0.707}$

So P1 and P3 will be assigned to centroid $m1 (0, 0.5)$ while P2 and P4 will be assigned to centroid $m2 (0.5,0.5)$

*Update step:*

$$
\begin{aligned}
m_1 &= \frac{1}{2} [(0,0)+(0,1)]=(0,0.5) \\
m_2 &= \frac{1}{2} [(1,0)+(1,1)]=(1,0.5)
\end{aligned}
$$

**Iteration 2:**

*Assignment step:*

|Distance to Centroid |$m1 (0, 0.5)$ |$m2 (0.5, 0.5)$|
|---|---|--|
$P1 (0, 0)$ | $\sqrt{(0-0)^2+(0-0.5)^2}=\bm{0.5}$ |	$\sqrt{(0-1.0)^2+(0-0.5)^2}=1.118$
$P2 (1,0)$ |	$\sqrt{(1-0)^2+(0-0.5)^2}=1.118$	 | $\sqrt{(1-1.0)^2+(0-0.5)^2}=\bm{0.5}$
$P3 (0, 1)$	| $\sqrt{(0-0)^2+(1-0.5)^2}=\bm{0.5}$ | 	$\sqrt{(0-1.0)^2+(1-0.5)^2}=1.118$
$P4 (1,1)$| 	$\sqrt{(1-0)^2+(1-0.5)^2}=1.118$	 | $\sqrt{(1-1.0)^2+(1-0.5)^2 }=\bm{0.5}$

And no new change in assignment will be done and the algorithm terminates.

## Python Implementation


