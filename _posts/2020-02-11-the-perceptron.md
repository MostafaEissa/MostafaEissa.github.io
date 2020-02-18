The perceptron is a learning algorithm. It is a rather simple algorithm yet surprising we can get very good results as we will later see. <!--more--> It is used for classification problems where we have two categories and we want to learn how to differentiate between them. For example, whether an email is spam? Or whether a house will sell over a certain asking price?
For each object we want to classify we collect a number of features that describe the object. In the house example, the features might be the number of rooms, total area, etc. So we end up with a number of numerical features (Note: categorical features can be converted to numerical using one hot encoding) and our task becomes gives a set of d  features $a_1,a_2,a_3,….,a_d$ of input $X$ will it classify as $+1$ or $-1$?

In other words, each input can now be represented as a vector where each individual component is one of the features and the perceptron is trying to find a hyperplane (think line in 2D) to separate the instances of the $+1$ class from the instances of $-1$ class.

For simplicity, let us consider inputs that only have two features $a_1$ and $a_2$ and as such each input corresponds to a point in 2D as seen in the next figure. The goal of the perceptron algorithm is to find a line such that all points on one side belong to the positive class and all points on the other side belong to the negative class. 

A plane can be described by a vector w that is perpendicular to the plane. As such, when we say we want to find a separating plane we mean that we want to find the components of the vector w that describe the plane.

Once the perceptron algorithm has found such hyperplane, we can classify instances based on the following rule. Note that the formula contains a bias term b without this term, the hyperplane will always have to go through the origin

$$ h(x) = 
    \left\{
    \begin{array}{l}
      +1 & if \enspace w.x+b > 0 \\
      -1 & if \enspace w.x+b < 0
    \end{array}
  \right.$$

Here, $w.x$ is the dot product which is the sum component wise components of both vectors $\sum\limits_{i=1}^{d} w_i x_i$ 

![alt-text](/assets/images/Presentation1.svg) 

![alt-text](..\assets\images\Presentation1.svg) 

### The Perceptron Update Rule

### Worked example
### Proof of limit on number of iterations
### Order of data matter 
### Algorithm and implementation
### Practical Example: MNIST
### Limitations xor example
