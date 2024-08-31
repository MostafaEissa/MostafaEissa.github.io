---
title: "Buy It Again"
date: 2024-08-31
tags: [Machine Learning]
mathjax: true
mathjaxEnableSingleDollar: true
---

Buy-it-again is a popular recommendation model in online retail. In this post we will cover a simple model proposed in [Buy it again: Modeling repeat purchase recommendations](https://dl.acm.org/doi/pdf/10.1145/3219819.3219891) that achieves good results despite its simplicity.

<!--more-->
# Customer Behavior

In online retail, a customer basket consists of :
- **repeat items**: items that the customer has consumed before, in previous baskets
- **explore items**: items that are new to the customer

The ratio of repeat to explore items will differ from customer to customer and also from business to business. For instance, an online grocery store is likely to have a high repeat ratio while an online fashion store will have a higher explore ratio. In this post we will focus on the problem of recommending repeat items (a.k.a buy it again). Our input is the historical customer transactions consisting of previous bought products (an example is shown below).

| TRANSACTION_DT | CUSTOMER_ID | PRODUCT_ID |
| -------------- | ----------- | ---------- |
| 01-Aug-2024    | 110405      | 10372      |
| 01-Aug-2024    | 418983      | 52535      |
| 02-Aug-2024    | 107331      | 54103      |
| 02-Aug-2024    | 110405      | 92129      |
| 03-Aug-2024    | 181995      | 21445      |
| ...            | ...         | ...        |

Before we explore various possible solutions, let's define the **buy it again** recommendation problem. 

# Problem Formulation

We would like to find the probability that a customer $c_i$ who purchased product $p_i$ k times in the past will purchase it again at time $t$. We will assume that the purchase probability of different products are independent of each other. Then, we are are interested in calculating:

$$
P_{c_i, p_i}(t_{k+1}=t | t_1,t_2,...,t_k)
$$

This probability density can be composed into two parts:
- $Q_{c_i,p_i}(A_i= k+1)$: is the probability that a customer purchases the product another time given that they already purchased it k times
- $R_{c_i, p_i}(t = t_{k+1} | t_1,t_2,...,t_k)$: is the probability that the customer makes a purchase at time $t_{k+1}$, conditioned on their repeat purchase history $t_1, t_2, ..., t_k$ of that product.

Then, our probability of interest can be approximated as:

$$
\begin{aligned}
P_{c_i, p_i}(t_{k+1}=t | t_1,t_2,...,t_k) \approx Q_{c_i,p_i}(A_i) \times R_{c_i, p_i}(t | t_1,t_2,...,t_k) \hspace{5em} (1)
\end{aligned}
$$


# Baseline: P-TopFreq

A simple and intuitive approach is using the customer purchase history. **P-TopFreq** finds the most frequent k items in the users' purchase history and use them for recommendation.

```Python
# count how many times each product was bought per customer
num_bought_products_per_customer_df = purchase_history_df
        .group_by(["CUSTOMER_ID", "PRODUCT_ID"])\
        .agg(pl.col("TRANSACTION_DT").count().alias("NUM_BOUGHT"))

# sort products in descending order according to the number they were bought
num_bought_products_per_customer_df = num_bought_products_per_customer_df\
        .sort(["CUSTOMER_ID","NUM_BOUGHT"], descending=True)

# return the top K items per customer as the next recommendation
topk_bought_products_per_customer_df = num_bought_products_per_customer_df\
        .group_by(["CUSTOMER_ID"])\
        .head(topk)
```

According to [A next basket recommendation reality check](https://dl.acm.org/doi/pdf/10.1145/3587153) , many state-of-the-art deep learning based models show modest or even no improvements compared to this  simple frequency-based baseline.

# RCA: Repeat Customer Probability Model

Our first model will completely ignore the time component and assume that $R_{c_i, p_i}(t)$ is a fixed constant across all customers and products. Furthermore, the model will assume that   $Q_{c_i,p_i}(A_i)$ is fixed across all customers (i.e., $Q_{c_i,p_i}(A_i) \approx Q_{p_i}(A_i)$ ). Then, for each product $p_i$ ,we approximate its $Q_{p_i}(A_i)$ by the repeat customer probability ($RCP_{p_i}$):

$$
RCP_{p_i} = \frac{\text{\# customers who bought product} \hspace{.2em} p_i \hspace{.2em} \text{more than once}}{\text{\# customers who bought the product} \hspace{.2em} p_i \hspace{.2em}  \text{at least once}}
$$

We generate recommendations by considering all the repeat purchasable products previously bought by customers and ranking them in the descending order of their RCA.

```Python
# count how many times each product was bought per customer
num_bought_products_per_customer_df = purchase_history_df
        .group_by(["CUSTOMER_ID", "PRODUCT_ID"])\
        .agg(pl.col("TRANSACTION_DT").count().alias("NUM_BOUGHT"))

# sort products in descending order according to the number they were bought
num_bought_products_per_customer_df = num_bought_products_per_customer_df\
        .sort(["CUSTOMER_ID","NUM_BOUGHT"], descending=True)

# for each product calculate the RCA as the ratio between customers 
# who bought the prodct more than once to those who bought it at least once
products_RCA_df = num_bought_products_per_customer_df\
        .group_by("PRODUCT_ID")\
        .agg([
            pl.col("CUSTOMER_ID").count()\
                .alias("NUM_CUSTOMERS_BOUGHT_PRODUCT_AT_LEAST_ONCE"), 

            pl.col("CUSTOMER_ID").filter(pl.col("NUM_BOUGHT") > 1)\
                .count().alias("NUM_CUSTOMERS_BOUGHT_PRODUCT_MORE_THAN_ONCE")
            ])\
        .with_columns(
            RCA=pl.col("NUM_CUSTOMERS_BOUGHT_PRODUCT_MORE_THAN_ONCE")/pl.col("NUM_CUSTOMERS_BOUGHT_PRODUCT_AT_LEAST_ONCE")
        )

# choose products with highest RCA in customer history
topk_rca_products_per_customer_df = purchase_history_df\
        .join(products_RCA_df, on="PRODUCT_ID", how='inner')\
        .sort(["CUSTOMER_ID", "RCA"], descending=True)\
        .group_by(["CUSTOMER_ID"])\
        .head(topk)
```

This model is based on the idea that not all products are repeat purchasable (e.g., fashion vs groceries) and the repeat purchase rate varies among repeat purchasable products. 

# ATD: Aggregate Time Distribution Model

ATD is based on the idea that for each product we have multiple customers who purchased it, if we aggregate their behaviors we can determine the repeat purchase characteristic of that product. For each product, we want to determine the distribution of the repeat purchase time intervals. 

In the paper, they used a log-normal distribution to model $R_{c_i, p_i}(t)$ and assumed it is the same for all customers for the same product. The distribution parameters $\bar{\mu_i}$ and $\bar{\sigma_i}^2$ are empirically estimated from the data.

$$
R_{p_i}(t) = \frac{1}{\sqrt{2\pi}t \bar{\sigma_i}}  exp [ -\frac{(ln t - \bar{\mu_i})^2}{2\bar{\sigma_i}^2}] , t > 0
$$

We still use RCA calculated in the previous model to estimate $Q_{p_i}(A_i)$.  In the paper, they proposed to filter all products with RCA above a certain threshold and then rank products based on their $R_{p_i}(t)$. However, in our implementation we will rank product based on the $Q_{p_i}(A_i) \times R_{p_i}(t)$ value.

```Python
# calculate mean time to repurchase per product
products_mean_time_to_repurchase_df = purchase_history_df\
        .sort(["PRODUCT_ID", "CUSTOMER_ID", "TRANSACTION_DT"], descending=False)\
        .group_by(["PRODUCT_ID", "CUSTOMER_ID"], maintain_order=True)\
        .agg(pl.col("TRANSACTION_DT").count().alias("NUM_BOUGHT"), (pl.col("TRANSACTION_DT").diff()/pl.duration(milliseconds=1)).alias("TIME_TO_NEXT_PURCHASE"))\
        .filter(pl.col("NUM_BOUGHT") > 2)\
        .group_by(["PRODUCT_ID"])\
        .agg(
                pl.col("TIME_TO_NEXT_PURCHASE").explode().drop_nulls().alias("MEAN_TIME_TO_NEXT_PURCHASE"), 
                pl.col("TIME_TO_NEXT_PURCHASE").explode().drop_nulls().log().alias("LOG_MEAN_TIME_TO_NEXT_PURCHASE")
        )

# estimate distribution parameters
products_mean_time_to_repurchase_params_df = products_mean_time_to_repurchase_df
        .with_columns(
                pl.col("LOG_MEAN_TIME_TO_NEXT_PURCHASE").map_elements(lambda x : x.mean()).alias("mean"), 
                pl.col("LOG_MEAN_TIME_TO_NEXT_PURCHASE").map_elements(lambda x : x.std()).alias("std")
        )

# find when was each product last purchased
last_purchase_time_df = purchase_history_df\
        .group_by(["CUSTOMER_ID", "PRODUCT_ID"])\
        .agg(pl.col("TRANSACTION_DT").max().alias("LAST_TRANSACTION_DT"))
```

Now, for any customer, if we want to recommend repeat products at a time $t_{new}$ in the future we can use:

```Python
products_mean_time_to_repurchase_params_df\
        .join(products_RCA_df, on=["PRODUCT_ID"], how="inner")\
        .join(last_purchase_time_df, on=["PRODUCT_ID"], how="inner")\
        .with_columns(((T_NEW - pl.col("LAST_TRANSACTION_DT"))/pl.duration(milliseconds=1)).alias("TIME_TO_NEXT_PURCHASE"))\
        .with_columns(pl.struct(["RCA", "std", "mean", "TIME_TO_NEXT_PURCHASE"]).map_elements(
                        lambda x: x["RCA"] * stats.lognorm.pdf(x["TIME_TO_NEXT_PURCHASE"], s=x["std"], scale=np.exp(x["mean"]))
                ,return_dtype=pl.Float64).alias("score")
        )\
        .filter(pl.col("score").is_not_nan())\
        .sort(["CUSTOMER_ID", "score"], descending=True)\
        .group_by(["CUSTOMER_ID"])\
        .head(topk)
```

This models improves upon the RCA model by incorporating the time to last purchase. Each product has a mean time to repurchase depending on its nature, for example, customers buy milk once a week but buy detergent once a months. ATD model accounts for this fact by adjusting the probability of repurchase to be highest as the time to last purchase is closest to the mean and drops if a new purchase has been recently made (customer has the product and unlikely to buy it again  or a long time has passed since the last purchase (customer is no longer interested in that product). 

# Conclusion

The paper demonstrates that the ATD model, despite its simplicity, delivers a significant boost in key metrics such as precision, recall, and NDCG. It also touches on more sophisticated models like the Modified Poisson-Gamma Model (MPG), which we might explore in a future post. I highly recommend diving into the paper for a deeper understanding.

As data scientists, our primary goal should be to address business challenges, not just to implement the latest complex models. Before delving into intricate solutions, it's essential to assess your business case, analyze your data, and test whether simpler models can effectively meet your objectives.

# References

- Li, M., Jullien, S., Ariannezhad, M., & de Rijke, M. (2023). A next basket recommendation reality check. ACM Transactions on Information Systems, 41(4), 1-29. ([pdf](https://dl.acm.org/doi/pdf/10.1145/3587153))

- Bhagat, R., Muralidharan, S., Lobzhanidze, A., & Vishwanath, S. (2018, July). Buy it again: Modeling repeat purchase recommendations. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 62-70). ([pdf](https://dl.acm.org/doi/pdf/10.1145/3219819.3219891))