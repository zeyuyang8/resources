---
title: Attention Mechanism
layout: default
parent: Deep Learning
grand_parent: Blogs
---

# Attention Mechanism

## Introduction

Attention in machine learning refers to a mechanism that allows models to focus on specific parts of input data.

## Notations

Main variables:

- $$\mathbf{X} \in \mathbb{R}^{m \times v}$$ -  features.
- $$\mathbf{Z} \in \mathbb{R}^{m \times v^{\prime}}$$ - transformed features.
- $$\mathbf{W} \in \mathbb{R}^{v \times v^{\prime}}$$ - attention weights.
- $$\mathbf{Y} \in \mathbb{R}^{m \times l}$$ - labels.
- $$m$$ - number of data points.
- $$v$$ - number of features in the data points.
- $$v^{\prime}$$ - number of features after attention transformation on the data points.
- $$l$$ - number of features in the labels.

Intermediate variables:

- $$\mathbf{Q} \in \mathbb{R}^{m \times q}$$ - queries.
- $$\mathbf{K} \in \mathbb{R}^{m \times q}$$ - keys.
- $$\mathbf{V} \in \mathbb{R}^{m \times v}$$ - values.
- $$\mathbf{W}_q \in \mathbb{R}^{v \times q}$$ - query weights.
- $$\mathbf{W}_k \in \mathbb{R}^{v \times q}$$ - key weights.
- $$\mathbf{W}_v \in \mathbb{R}^{v \times v}$$ - value weights.
- $$\mathbf{W}_h \in \mathbb{R}^{hv^{\prime} \times d}$$ - multi-head attention weights.
- $$q$$ - number of features in the queries and keys.
- $$h$$ - number of attention heads.
- $$d$$ - number of features in the output of multi-head attention.

Functions:

- $$\varphi$$ - non-linear activation function.
- $$\sigma$$ - softmax function.
- $$a$$ - similarity function.

## Mathematical Formulation

The attention mechanism can be formulated as follows,

$$
\mathbf{Z}=\varphi(\mathbf{XW}(\mathbf{X}))
$$

where $$\varphi$$ is a non-linear function. The term $$\mathbf{W}(\mathbf{X})$$ is weights derived from the data matrix, as attention focuses on certain parts of input sequences.

More generally, we can write,

$$
\mathbf{Z}=\varphi(\mathbf{V W}(\mathbf{Q}, \mathbf{K}))
$$

where $$\mathbf{Q}$$, $$\mathbf{K}$$, and $$\mathbf{V}$$ are queries, keys, and values derived from $$\mathbf{X}$$, respectively. The queries are used to describe what each input sequence is "asking about", and keys are used to describe what each input sequence contains. The values are used to describe how each input sequence should be transmitted to the output sequence. Queries, keys, and values are usually computed by linear projections of the input sequence $$\mathbf{X}$$.

$$
\mathbf{Q}=\mathbf{X}\mathbf{W}_q,
\mathbf{K}=\mathbf{X}\mathbf{W}_k,
\mathbf{V}=\mathbf{X}\mathbf{W}_v
$$

When using attention to compute one row of $$\mathbf{Z}$$, denoted by $$\mathbf{z}$$, we first need to use its corresponding query $$\mathbf{q}$$ and compare it to each key $$\mathbf{k}$$ in $$\mathbf{K}$$ to get an array of similarity scores. Then we normalize these similarity scores with softmax function,

$$
\mathbf{z}(
  \mathbf{q}, (\mathbf{k}_1, \mathbf{v}_1),
  \ldots, (\mathbf{k}_m, \mathbf{v}_m)
) = \sum_{i=1}^m \alpha_i(\mathbf{q}, \mathbf{K}) \mathbf{v}_i
$$

Assume that queries and keys have the same length $$q$$, we can compute the similarity scores between queries and keys by dot product. If we assume queries and keys are independent with 0 mean and unit variance, then the mean of the dot product between one key-query pair is 0 and the variance is $$q$$. To ensure the variance is 1, we can divide the dot product by $$\sqrt{q}$$. Thus, the similarity score between one query-key pair can be written as:

$$
a(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^T \mathbf{k}}{\sqrt{q}} \in \mathbb{R}
$$

When computing attention output with batches of data points, we can write,

$$
\mathbf{Z}(\mathbf{X}) = \mathbf{Z}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \sigma(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{q}})\mathbf{V}
$$

where the softmax function is applied row-wise.

## Attention in Neural Networks

### Self-Attention

Given a list of input sequences $$\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_m]$$, we can compute the self-attention output $$\mathbf{Z} = [\mathbf{z}_1, \ldots, \mathbf{z}_m]$$, where $$\mathbf{z}$$ is as follows:

$$
\mathbf{z}(\mathbf{x}, (\mathbf{x}_1, \mathbf{x}_1),
  \ldots, (\mathbf{x}_m, \mathbf{x}_m)
) = \sum_{i=1}^m \alpha_i(\mathbf{x}, \mathbf{X}) \mathbf{x}_i
$$

In other words, self-attention uses the input sequences as queries, keys, and values. The matrix multiplication with batches of data points can be written as,

$$
\mathbf{Z}(\mathbf{X}) = \mathbf{Z}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{Z}(\mathbf{X}, \mathbf{X}, \mathbf{X}) = \sigma(\frac{\mathbf{X} \mathbf{X}^T}{\sqrt{q}})\mathbf{X}
$$

where the softmax function is applied row-wise.

### Multi-Head Attention

To increase the flexibility of attention, we can use multiple attention heads. Each attention head has its own query, key, and value matrices. Let us denote the $$i$$-th attention head as $$\mathbf{Z}^{(i)} \in \mathbb{R}^{m \times v^{\prime}}$$:

$$
\mathbf{Z}^{(i)}(\mathbf{X}) = \mathbf{Z}^{(i)}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \sigma(\frac{\mathbf{Q}^{(i)} \mathbf{K}^{(i)T}}{\sqrt{q}})\mathbf{V}^{(i)}
$$

where $$\mathbf{Q}^{(i)}$$, $$\mathbf{K}^{(i)}$$, and $$\mathbf{V}^{(i)}$$ are queries, keys, and values for the $$i$$-th attention head with weights $$\mathbf{W}_q^{(i)}$$, $$\mathbf{W}_k^{(i)}$$, and $$\mathbf{W}_v^{(i)}$$, respectively. The multi-head attention output is the concatenation of all attention heads multiplied by a weight matrix $$\mathbf{W}_h \in \mathbb{R}^{hv^{\prime} \times d}$$:

$$
\mathbf{Z}(\mathbf{X}) = \mathbf{Z}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \left[ \mathbf{Z}^{(1)}(\mathbf{Q}, \mathbf{K}, \mathbf{V}), \ldots, \mathbf{Z}^{(h)}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \right]\mathbf{W}_h
$$

where $$h$$ is the number of attention heads, and $$d$$ is the number of features in the output of multi-head attention.

### Cross-Attention

When mixing two different input sequences, we can use cross-attention, which uses one sequence to compute queries, and uses the other sequence to compute keys and values. Assume we want to mix features $$\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_m]$$ and labels $$\mathbf{Y} = [\mathbf{y}_1, \ldots, \mathbf{y}_m]$$, and we use the self-attention style to compute queries, keys, values from features and labels. Then, we can compute the cross-attention output $$\mathbf{Z} = [\mathbf{z}_1, \ldots, \mathbf{z}_m]$$, where $$\mathbf{z}$$ is as follows:

$$
\mathbf{z}(
  \mathbf{q}, (\mathbf{k}_1, \mathbf{v}_1),
  \ldots, (\mathbf{k}_m, \mathbf{v}_m)
) = \mathbf{z}(\mathbf{x}, (\mathbf{y}_1, \mathbf{y}_1),
  \ldots, (\mathbf{y}_m, \mathbf{y}_m)
) = \sum_{i=1}^m \alpha_i(\mathbf{x}, \mathbf{Y}) \mathbf{y}_i
$$

## References

1. K. P. Murphy, Probabilistic Machine Learning: An Introduction. MIT press, 2022.
1. K. P. Murphy. Probabilistic machine learning: Advanced topics. MIT press, 2023.
