The perceptron is a learning algorithm. A rather simple one yet surprising, it can acheive very good results as we will explore in this article. <!--more--> 

The perceptron is used for classification task where there are two categories and the objective is to learn how to differentiate between them. For example, whether an email is spam? Or whether a house will sell over a certain asking price?

Traditionally, for each object we want to classify a number of features are collected that best describe the object. In the house example mentioned earlier, the features might be the number of rooms, total area, number of bathrooms, etc. As a result we end up with a number of numerical features describing each object (instance) (Note: categorical features can be converted to numerical using one hot encoding) and our task can be described mathematically as given input instances $X$ of size $n$ where each instance is $d$-dimensional vector $a_1,a_2,a_3,….,a_d$ containing the feature values, design a function $h(x)$ that takes an input instance x and outputs  $+1$ if $x$ belongs to the positive classs and $-1$ otherwise.

$$ h(x) = 
    \left\{
    \begin{array}{l}
      +1 & if \enspace x \in \text{positive class} \\
      -1 & if \enspace x \in \text{negative class }
    \end{array}
  \right.$$

In other words, each input is represented as a vector where each individual component is one of the features and the perceptron is trying to find a hyperplane (think line in 2D) to separate the instances of the $+1$ class from the instances of $-1$ class.

For simplicity, let us focus on the 2D plane and limit inputs to only two features $a_1$ and $a_2$. In this setting, each input corresponds to a point in 2D as seen in the next figure. The goal of the perceptron algorithm is to find a line such that all points on one side belong to the positive class and all points on the other side belong to the negative class. 

![the separating hyperplane](/assets/images/the-perceptron/the-separating-hyperplane.png) 

![the separating hyperplane](..\assets/images/the-perceptron/the-separating-hyperplane.png) 

A plane can be described by a vector $w$ that is normal to the plane. Hence, when we say we want to find a separating plane we mean that we want to find the components of the vector $w$ that describe the plane.

Once the perceptron algorithm has found such hyperplane, we can classify instances based on the following rule. Note that the formula contains a bias term $b$ without this term, the hyperplane will always have to go through the origin

$$ h(x) = 
    \left\{
    \begin{array}{l}
      +1 & if \enspace w.x+b > 0 \\
      -1 & if \enspace w.x+b < 0
    \end{array}
  \right.$$


Here, $w.x$ is the dot product which is the sum component wise components of both vectors $\sum\limits_{i=1}^{d} w_i x_i$ 

### The Perceptron Update Rule

The question now is how can we find such $w$ that classifies the points correctly. A simple idea is to initialize the $w$ vector randomly and go over the points one by one. If the point is correctly classified then we do not need to do anything however, if is misclassified we need to move the hyperplane (update $w$) so that the point will eventually become correctly classified. A good choice, as shown in the next figure, is a vector that is proportional to the misclassified data point but in the opposite direction i.e. $w_{new} =w_{old}+ηyx$. The algorithm keeps iterating until $w$ converges does not change anymore.

![the perceptron update rule](/assets/images/the-perceptron/the-perceptron-update-rule.png) 

![the perceptron update rule](..\assets/images/the-perceptron/the-perceptron-update-rule.png) 

Each iteration of the algorithm consists of a full pass over all the training examples. The algorithm terminates when in a given iteration no point was misclassified.
A natural question at this point is how can we update the bias term $b$ as well. The trick is to augment our training data with all ones vector and treat the bias as another weight component (i.e. weight component of a feature whose value is alaways one).

$$
\begin{aligned}
 h(x) &= w.x+b \\
 &= ∑_{i=1}^d w_i x_i+b \\
 &= ∑_{i=1}^d w_i x_i+b*1 \\
 &= ∑_{i=1}^{d+1} w_i x_i
\end{aligned}
$$

where $x_{d+1}$ always equals one and $w_{d+1}$ is the bias term

### Worked Example

To Understand the algorithm better, let’s go through a toy example. Suppose we have the following positive and negative examples in 3D

|Positive Examples   | Negative Examples |
|:------------------:|:-------------------:|
|$[0 \enspace 0 \enspace 0]$| $[0 \enspace 1 \enspace 1]$|
|$[0 \enspace 0 \enspace 1]$ | $[0 \enspace 1 \enspace 0]$|
|$[1 \enspace 0 \enspace 1]$ | $[1 \enspace 1 \enspace 0]$ 
|$[1 \enspace 0 \enspace 0]$ | $[1 \enspace 1 \enspace 1]$|

![3d example](/assets/images/the-perceptron/3d-example.png) 

![3d example](..\assets/images/the-perceptron/3d-example.png) 

***Iteration 1***

Weight <br> $[w_0 \enspace w_1 \enspace w_2 \enspace b]$	| Data <br> $[x_0 \enspace x_1 \enspace x_2 \enspace 1]$|	$\bold{w.x}$	|Label |	Comment|	New weight|
:-------:|:-------:|------:|:---------|--------|:------:|
$[0 \enspace 0\enspace  0\enspace  0]$|	$[0 \enspace 0 \enspace 0 \enspace 1]$|	$0 \leq 0$|	$+$	|Wrong. Add sample|	$[0 \enspace 0 \enspace 0 \enspace 1]$
$[0 \enspace 0 \enspace 0 \enspace 1]$	|$[0 \enspace 0 \enspace 1 \enspace 1]$|	$1 > 0$	|$+$|	OK	|No Change
$[0 \enspace 0 \enspace 0 \enspace 1]$	|$[0 \enspace 1 \enspace 0 \enspace 1]$	| $1 > 0$ |	$-$	|Wrong. Subtract sample	|$[0 \enspace {-1} \enspace  0 \enspace 0]$
$[0 \enspace {-1} \enspace 0 \enspace 0]$|	$[0 \enspace 1 \enspace 1 \enspace 1]$|	${-1} < 0$|	$-$	|OK	|No Change
$[0 \enspace {-1} \enspace 0 \enspace 0]$|	$[1 \enspace 0 \enspace 0 \enspace 1]$|	$0 \leq 0$	|$+$|	Wrong. Add sample	|$[1 \enspace {-1} \enspace 0 \enspace 1]$
$[1 \enspace {-1} \enspace 0 \enspace 1]$|	$[1 \enspace 0 \enspace 1 \enspace 1]$	|$2 > 0$	|$+$|	OK|	No Change
$[1 \enspace {-1} \enspace 0 \enspace 1]$|	$[1 \enspace 1 \enspace 0 \enspace 1]$|	$1 > 0$|	$-$|	Wrong. Subtract sample|	$[0 \enspace {-2} \enspace 0 \enspace 0]$
$[0 \enspace {-2} \enspace 0 \enspace 0]$|	$[1 \enspace 1 \enspace 1 \enspace 1]$	|$-2 < 0$	|$-$|	OK|	No Change

***Iteration 2***

***Iteration 3***

### Limitations: XOR Example
### Python Implementation
### Conclusion
