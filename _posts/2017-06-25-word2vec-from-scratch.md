---
layout: post
title:  word2vec from scratch
date:   2017-06-25 21:10:49 +0900
categories: nlp
---

Great article for overall explanation: [http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

Implement skipgram model of word2vec.

Let sentence is `S I really really like you E`, and vocabulary is `S`, `I`, `really`, `like`, `you`, `E`.

If we set the **window size** to $$1$$, combinations of a center word($$w_c$$) and outside word($$w_o$$) are:

$$
\begin{array}{l|r}
w_c\,idx &   w_o   &   w_c\\
\hline
1     &  S_{=start}&   I \\
1     &  really  &     I \\
2     &  I       & really \\
2     &  really  & really \\
3     &  really  & really \\
3     &  like    & really \\
4     &  really  & like \\
4     &   you    & like \\
5     &  like    & you \\
5     &  E_{=end}& you
\end{array}
$$

Maximizing $$p(w_o \lvert w_c)$$ for every $$w_c$$ is our objective. The objective function $$J(\theta)$$ is

$$
\begin{array}{l}
J(\theta) = & & p(\mathtt{S|I})\, & \times & p(\mathtt{really|I}) \\
            & \times & p(\mathtt{I|really})\, & \times & p(\mathtt{really|really}) \\
            & \times & p(\mathtt{really|really})\, & \times &  p(\mathtt{like|really}) \\
            & \times & p(\mathtt{really|like})\, & \times & p(\mathtt{you|like}) \\
            & \times & p(\mathtt{like|you})\, & \times & p(\mathtt{E|you})
\end{array}
$$

For generalization,

$$ J(\theta) = \prod_{t=1}^T\prod_{-m\le j\le m, j \ne 0 } p(w_{t+j}|w_t) \tag 1 $$

<br><br>

----
# How to find $$p(w_o|w_c)$$?

There are two important hypotheses used in this model. The first one is **distributional hypothesis** which means that words in similar context have a similar meaning. For example, if `smart` and `intelligent` are used in similar context, then we assume that these words have a similar meaning. The second one is that the similarity between two vectors can be represented by the inner product of them.

1. Make **one-hot vector** of words $$w_c$$ and $$w_o$$.

2. Make **center word vector** $$ v_c = \mathcal{V} \cdot w_c $$

3. Make **outer word vector** $$ u_o = \mathcal{U} \cdot w_o $$

4. Get the similarity of two words $$ z = u_o \cdot v_c $$

(The flow is somewhat different from real **word2vec** layers, but we will see this later.)

Use the $$softmax$$ function to normalize and change to probability.

So we can use:

$$
p(w_o|w_c) = softmax(u_o v_c)
           = \frac {exp(u_o v_c)}{\sum_{w=1}^v exp(u_w v_c)} \tag 2
$$


Combine $$(1)$$ and $$(2)$$, we maximize:

$$
J(\theta) = \prod_{c=1}^T\prod_{-m\le j\le m, j \ne 0 }
          \frac {exp(u_{c+j} v_c)}{\sum_{w=1}^v exp(u_w v_c)} \tag 3
$$

For easy calculation, use $$log$$ to change production to summation and multiply $$-1$$ to minimize.


$$ J'(\theta)
= - \sum_{c=1}^T\sum_{-m\le j\le m, j \ne 0 }
\log \frac {exp(u_{c+j} v_c)}{\sum_{w=1}^v exp(u_w v_c)} \tag 4
$$

Then,

$$
\begin{align}
J'(\theta)
&= log \frac{exp(u_{S}v_{I})}
    {
      exp(u_{S}v_{I}) + exp(u_{I}v_{I}) + exp(u_{really}v_{I})
      + exp(u_{like}v_{I}) + exp(u_{you}v_{I}) + exp(u_{E}v_{I})
    } + \\[1ex]
&  log \frac{exp(u_{really}v_{I})}
    {
      exp(u_{S}v_{I}) + exp(u_{I}v_{I}) + exp(u_{really}v_{I})
      + exp(u_{like}v_{I}) + exp(u_{you}v_{I}) + exp(u_{E}v_{I})
    } + \\[3ex]

&  log \frac{exp(u_{I}v_{really})}
    {
      exp(u_{S}v_{really}) + exp(u_{I}v_{really}) + exp(u_{really}v_{really})
      + exp(u_{like}v_{really}) + exp(u_{you}v_{really}) + exp(u_{E}v_{really})
    } + \\[1ex]
&  log \frac{exp(u_{really}v_{really})}
    {
      exp(u_{S}v_{really}) + exp(u_{I}v_{really}) + exp(u_{really}v_{really})
      + exp(u_{like}v_{really}) + exp(u_{you}v_{really}) + exp(u_{E}v_{really})
    } + \\[3ex]

&  log \frac{exp(u_{really}v_{really})}
    {
      exp(u_{S}v_{really}) + exp(u_{I}v_{really}) + exp(u_{really}v_{really})
      + exp(u_{like}v_{really}) + exp(u_{you}v_{really}) + exp(u_{E}v_{really})
    } + \\[1ex]
&  log \frac{exp(u_{like}v_{really})}
    {
      exp(u_{S}v_{really}) + exp(u_{I}v_{really}) + exp(u_{really}v_{really})
      + exp(u_{like}v_{really}) + exp(u_{you}v_{really}) + exp(u_{E}v_{really})
    } + \\[3ex]

&  log \frac{exp(u_{really}v_{like})}
    {
      exp(u_{S}v_{like}) + exp(u_{I}v_{like}) + exp(u_{really}v_{like})
      + exp(u_{like}v_{like}) + exp(u_{you}v_{like}) + exp(u_{E}v_{like})
    } + \\[1ex]
&  log \frac{exp(u_{you}v_{like})}
    {
      exp(u_{S}v_{like}) + exp(u_{I}v_{like}) + exp(u_{really}v_{like})
      + exp(u_{like}v_{like}) + exp(u_{you}v_{like}) + exp(u_{E}v_{like})
    } + \\[3ex]

&  log \frac{exp(u_{like}v_{you})}
    {
      exp(u_{S}v_{you}) + exp(u_{I}v_{you}) + exp(u_{really}v_{you})
      + exp(u_{like}v_{you}) + exp(u_{you}v_{you}) + exp(u_{E}v_{you})
    } + \\[1ex]
&  log \frac{exp(u_{E}v_{you})}
    {
      exp(u_{S}v_{you}) + exp(u_{I}v_{you}) + exp(u_{really}v_{you})
      + exp(u_{like}v_{you}) + exp(u_{you}v_{you}) + exp(u_{E}v_{you})
    }\\[3ex]
\end{align} \tag {4-1}
$$

<br>

If we find $$\mathcal{V}$$ and $$\mathcal{U}$$ that minimize $$J'(\theta)$$, then we are done. $$\mathcal{V}$$ and $$\mathcal{U}$$ are called **parameters** that we optimize. There are another types of parameters called **hyperparameters** that we need to decide(not optimize). The **window size** that we set to $$1$$ is one of **hyperparamters**. We decide another **hyperparameters**: size of $$\mathcal{V}$$ and $$\mathcal{U}$$. Let the size of these matrice be $$3$$, and **vocab size** is $$6$$ in our example, then $$\mathcal{V} \in \mathbf{R}^{6 \times 3 }$$ and $$\mathcal{U} \in \mathbf{R}^{6 \times 3}$$(Actually, **word2vec** use $$\mathcal{U} \in \mathbf{R}^{3 \times 6}$$. We will see this later.)

To calculate $$J(\theta)$$ we set the inital values of $$\mathcal{V} \in \mathbf{R}^{6 \times 3 }$$ and $$\mathcal{U} \in \mathbf{R}^{6 \times 3}$$.

Let $$\mathcal{V} = $$
$$
\begin{pmatrix}
0.01 & 0.01 & 0.01 \\
0.02 & 0.02 & 0.02 \\
0.03 & 0.03 & 0.03 \\
0.04 & 0.04 & 0.04 \\
0.05 & 0.05 & 0.05 \\
0.06 & 0.06 & 0.06
\end{pmatrix}
$$

Let $$\mathcal{U} = $$
$$
\begin{pmatrix}
0.11 & 0.11 & 0.11 \\
0.12 & 0.12 & 0.12 \\
0.13 & 0.13 & 0.13 \\
0.14 & 0.14 & 0.14 \\
0.15 & 0.15 & 0.15 \\
0.16 & 0.16 & 0.16
\end{pmatrix}
$$

$$ one-hot = $$
$$
\begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
\equiv
\begin{pmatrix}
S\\
I\\
really\\
like\\
you\\
E\\
\end{pmatrix}
$$

Then,

$$
\begin{array}{l}
& exp(u_{S} \cdot v_{I}) = exp(v_{I}\cdot u_{S}) = exp(w_{I}V \cdot w_{S}U) =\\[2ex]
& exp\Bigl(\bigl(0, 1, 0, 0, 0, 0 \bigr)
  \cdot
  \begin{pmatrix}
  0.01 & 0.01 & 0.01 \\
  0.02 & 0.02 & 0.02 \\
  0.03 & 0.03 & 0.03 \\
  0.04 & 0.04 & 0.04 \\
  0.05 & 0.05 & 0.05 \\
  0.06 & 0.06 & 0.06
  \end{pmatrix}
\Bigr)
\cdot
\Bigl(\bigl(1, 0, 0, 0, 0, 0 \bigr)
  \cdot
  \begin{pmatrix}
  0.11 & 0.11 & 0.11 \\
  0.12 & 0.12 & 0.12 \\
  0.13 & 0.13 & 0.13 \\
  0.14 & 0.14 & 0.14 \\
  0.15 & 0.15 & 0.15 \\
  0.16 & 0.16 & 0.16
  \end{pmatrix}
\Bigr)
\end{array}
$$

As you can see, the inner product of word one-hot vector and matrix is like just picking the corresponding vector of that word from matrix.

That means **input word matrix** $$\mathcal{V} = $$

$$
\begin{pmatrix}
0.01 & 0.01 & 0.01 \\
0.02 & 0.02 & 0.02 \\
0.03 & 0.03 & 0.03 \\
0.04 & 0.04 & 0.04 \\
0.05 & 0.05 & 0.05 \\
0.06 & 0.06 & 0.06
\end{pmatrix}
$$ is for $$
\begin{pmatrix}
S\\
I\\
really\\
like\\
you\\
E\\
\end{pmatrix}
$$ and index is $$
\begin{pmatrix}
0\\
1\\
2\\
3\\
4\\
5
\end{pmatrix}
$$

, and **output word matrix** $$\mathcal{U} = $$

$$
\begin{pmatrix}
0.11 & 0.11 & 0.11 \\
0.12 & 0.12 & 0.12 \\
0.13 & 0.13 & 0.13 \\
0.14 & 0.14 & 0.14 \\
0.15 & 0.15 & 0.15 \\
0.16 & 0.16 & 0.16
\end{pmatrix}
$$ is also for $$
\begin{pmatrix}
S\\
I\\
really\\
like\\
you\\
E\\
\end{pmatrix}
$$ and index is $$
\begin{pmatrix}
0\\
1\\
2\\
3\\
4\\
5
\end{pmatrix}
$$

Don't forget that we have two vectors for each word.

We will use indexed words instead of real words for convenience. We can rewrite $$\text{(4-1)}$$ to:

$$
\begin{align}
J'(\theta)
&= log \frac{exp(u_{0}v_{1})}
    {
      exp(u_{0}v_{1}) + exp(u_{1}v_{1}) + exp(u_{2}v_{1})
      + exp(u_{3}v_{1}) + exp(u_{4}v_{1}) + exp(u_{5}v_{1})
    } + \\[1ex]
&  log \frac{exp(u_{2}v_{1})}
    {
      exp(u_{0}v_{1}) + exp(u_{1}v_{1}) + exp(u_{2}v_{1})
      + exp(u_{3}v_{1}) + exp(u_{4}v_{1}) + exp(u_{5}v_{1})
    } + \\[3ex]

&  log \frac{exp(u_{1}v_{2})}
    {
      exp(u_{0}v_{2}) + exp(u_{1}v_{2}) + exp(u_{2}v_{2})
      + exp(u_{3}v_{2}) + exp(u_{4}v_{2}) + exp(u_{5}v_{2})
    } + \\[1ex]
&  log \frac{exp(u_{2}v_{2})}
    {
      exp(u_{0}v_{2}) + exp(u_{1}v_{2}) + exp(u_{2}v_{2})
      + exp(u_{3}v_{2}) + exp(u_{4}v_{2}) + exp(u_{5}v_{2})
    } + \\[3ex]

&  log \frac{exp(u_{2}v_{2})}
    {
      exp(u_{0}v_{2}) + exp(u_{1}v_{2}) + exp(u_{2}v_{2})
      + exp(u_{3}v_{2}) + exp(u_{4}v_{2}) + exp(u_{5}v_{2})
    } + \\[1ex]
&  log \frac{exp(u_{3}v_{2})}
    {
      exp(u_{0}v_{2}) + exp(u_{1}v_{2}) + exp(u_{2}v_{2})
      + exp(u_{3}v_{2}) + exp(u_{4}v_{2}) + exp(u_{5}v_{2})
    } + \\[3ex]

&  log \frac{exp(u_{2}v_{3})}
    {
      exp(u_{0}v_{3}) + exp(u_{1}v_{3}) + exp(u_{2}v_{3})
      + exp(u_{3}v_{3}) + exp(u_{4}v_{3}) + exp(u_{5}v_{3})
    } + \\[1ex]
&  log \frac{exp(u_{4}v_{3})}
    {
      exp(u_{0}v_{3}) + exp(u_{1}v_{3}) + exp(u_{2}v_{3})
      + exp(u_{3}v_{3}) + exp(u_{4}v_{3}) + exp(u_{5}v_{3})
    } + \\[3ex]

&  log \frac{exp(u_{3}v_{4})}
    {
      exp(u_{0}v_{4}) + exp(u_{1}v_{4}) + exp(u_{2}v_{4})
      + exp(u_{3}v_{4}) + exp(u_{4}v_{4}) + exp(u_{5}v_{4})
    } + \\[1ex]
&  log \frac{exp(u_{5}v_{4})}
    {
      exp(u_{0}v_{4}) + exp(u_{1}v_{4}) + exp(u_{2}v_{4})
      + exp(u_{3}v_{4}) + exp(u_{4}v_{4}) + exp(u_{5}v_{4})
    }\\[3ex]


&= log \frac{exp(0.0066)}
    {
      exp(0.0066) + exp(0.0072) + exp(0.0078)
      + exp(0.0084) + exp(0.0090) + exp(0.0096)
    } + \\[1ex]
&  log \frac{exp(0.0078)}
    {
      exp(0.0066) + exp(0.0072) + exp(0.0078)
      + exp(0.0084) + exp(0.0090) + exp(0.0096)
    } + \\[3ex]

&  log \frac{exp(0.0108)}
    {
      exp(0.0099) + exp(0.0108) + exp(0.0117)
      + exp(0.0126) + exp(0.0135) + exp(0.0144)
    } + \\[1ex]
&  log \frac{exp(0.0117)}
    {
      exp(0.0099) + exp(0.0108) + exp(0.0117)
      + exp(0.0126) + exp(0.0135) + exp(0.0144)
    } + \\[3ex]

&  log \frac{exp(0.0117)}
    {
      exp(0.0099) + exp(0.0108) + exp(0.0117)
      + exp(0.0126) + exp(0.0135) + exp(0.0144)
    } + \\[1ex]
&  log \frac{exp(0.0126)}
    {
      exp(0.0099) + exp(0.0108) + exp(0.0117)
      + exp(0.0126) + exp(0.0135) + exp(0.0144)
    } + \\[3ex]

&  log \frac{exp(0.0156)}
    {
      exp(0.0132) + exp(0.0144) + exp(0.0156)
      + exp(0.0168) + exp(0.0180) + exp(0.0192)
    } + \\[1ex]
&  log \frac{exp(0.0180)}
    {
      exp(0.0132) + exp(0.0144) + exp(0.0156)
      + exp(0.0168) + exp(0.0180) + exp(0.0192)
    } + \\[3ex]

&  log \frac{exp(0.0210)}
    {
      exp(0.016) + exp(0.0180) + exp(0.0195)
      + exp(0.0210) + exp(0.0225) + exp(0.0240)
    } + \\[1ex]
&  log \frac{exp(0.0240)}
    {
      exp(0.016) + exp(0.0180) + exp(0.0195)
      + exp(0.0210) + exp(0.0225) + exp(0.0240)
    } + \\[5ex]

&= log \frac{1.0066}{6.0488} + log \frac{1.0078}{6.0488} +
   log \frac{1.0109}{6.0734} + log \frac{1.0118}{6.0734} +
   log \frac{1.0118}{6.0734} + \\[2ex]
&  log \frac{1.0127}{6.0734} + log \frac{1.0157}{6.0980} +
   log \frac{1.0181}{6.0980} + log \frac{1.0212}{6.1222} +
   log \frac{1.0242}{6.1222} \\[5ex]
&= -17.9155
\end{align}
$$

<br><br>

To figure out $$-17.9155$$ is the minimum value(maybe not), we use **gradient**. From now on, we do this using `python`.

{% highlight python %}
sentence = "S I really really like you E"
vocab = ['S', 'I', 'really', 'like', 'you', 'E']
one_hot = np.array([
  [1, 0, 0, 0, 0, 0], # S
  [0, 1, 0, 0, 0, 0], # I
  [0, 0, 1, 0, 0, 0], # really
  [0, 0, 0, 1, 0, 0], # like
  [0, 0, 0, 0, 1, 0], # you
  [0, 0, 0, 0, 0, 1]  # E
])
sentence_by_one_hot_idx = [
  0, # S
  1, # I
  2, # really
  2, # really
  3, # like
  4, # you
  5  # E
]
vocab_by_one_hot_idx = [
  0, 1, 2, 3, 4, 5
]


v = len(vocab) # = len(input)
n = 3 # dimension of input word vector
c = 1 # window size
V = np.array([
  [0.01, 0.01, 0.01],
  [0.02, 0.02, 0.02],
  [0.03, 0.03, 0.03],
  [0.04, 0.04, 0.04],
  [0.05, 0.05, 0.05],
  [0.06, 0.06, 0.06]
])
U = np.array([
  [0.11, 0.11, 0.11],
  [0.12, 0.12, 0.12],
  [0.13, 0.13, 0.13],
  [0.14, 0.14, 0.14],
  [0.15, 0.15, 0.15],
  [0.16, 0.16, 0.16]
])

def make_combis(sentence):
  combis = []
  size = len(sentence)
  for idx, v in enumerate(sentence):
    if idx == 0 or idx == size - 1:
      continue
    combis.append([sentence[idx-1], sentence[idx]])
    combis.append([sentence[idx+1], sentence[idx]])
  return np.array(combis)

 # this function is too inefficient to be used in practice.
def get_prob(combis, U, V):
  exps = []
  for j in vocab_by_one_hot_idx:
    exp = []
    for i in vocab_by_one_hot_idx:
      exp.append(np.exp(np.dot(U[i], V[j])))
    exps.append(sum(exp))
  values = []
  for combi in combis:
    o, c = combi
    top = np.exp(np.dot(U[o], V[c]))
    bottom = exps[c]
    value = np.log(top/bottom)
    values.append(value)
  return sum(values)

prob = get_prob(make_combis(sentence_by_one_hot_idx), U, V) # -17.9155

def numerical_gradient(combis, U, V):
  

{% endhighlight %}













































def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))



<br>



<br><br>

----



Adjust parameters to minize errors betweeen `y` and `y_hat`.

Layers of **word2vec**:

$$
\begin{array}{l}
\mathbf{input\,layer}\,w_c &: V\text{-dim} \text{(one-hot vector)}\\
\text{input word matrix $\mathcal{V}$ } &: V \times N\\
\mathbf{projection\,layer}\,h &: N\\
\text{output word matrix $\mathcal{U}$ } &: N \times V\\
\mathbf{output\,layer}\,y &: C \times V\text{-dim} \text{(C: context size)}\\
\end{array}
$$






In neural net, layers are named as input layer, hidden layer and output layer. There is an activation function in a hidden layer to add nonlinearity, but here in **word2vec** we call it **projection layer** since it doesn't have an activation function.


<br><br>

----
# Modelling

Layers of **word2vec**:

$$
\begin{array}{l}
\text{input layer $w_c$ } &: V\text{-dim} \text{(one-hot vector)}\\
\text{input word matrix $\mathcal{V}$ } &: V \times N\\
\text{projection layer $h$ } &: N\\
\text{output word matrix $\mathcal{U}$ } &: N \times V\\
\text{output layer $y$ } &: C \times V\text{-dim} \text{(C: context size)}\\
\end{array}
$$

In neural net, layers are named as input layer, hidden layer and output layer. There is an activation function in a hidden layer to add nonlinearity, but here in **word2vec** we call it **prjection layer** since it doesn't have an activation function.

To make it simple, let dimension of **proejction layer** be $$3$$, then this model use 4-dimensional **one-hot vector** as **input layer**, project it to 3-dimensional **projection layer** and flow it to 4-dimensional **output layer**.

{% highlight python %}
v = len(vocab)
n = 3
c = 1

 # one by one example
w_c = one_hot[0]                    # w_c.shape = (4,)
V = np.zeros((v, n), float) + 0.01  # V.shape   = (4, 3)
h = np.dot(w_c, V)                  # h.shape   = (3,)
U = np.zeros((n, v), float) + 0.01  # U.shape   = (3, 4)
y = np.dot(h, U)                    # y.shape   = (4,)
{% endhighlight %}





true probability
input w_c : [1,0,0,0]
y^{c+1} = [0,1,0,0]

input w_c : [0,1,0,0]
y^{c-1} = [1,0,0,0]
y^{c+1} = [0,0,1,0]





$$
\text{output $y$ :} C \times V\text{-dim} \text{(C: context size)}
$$

$$
\overbrace{p(w_b \lvert w_a)}^{observed} > \overbrace{p(w_c \lvert w_a)}^{unobserved}
$$

가 되어야 한다.

$$p(w_o \lvert w_c)$$


즉 위의 모든 콤비에 대하여 $$p(w_o \lvert w_c)$$의 확률을 최대화하는 것이 목적이다.

일반화하여, 윈도우사이즈를 $$m$$, 전체 어휘 갯수를 $$t$$라고 하면,

$$
= \sum_{t=1}^T\sum_{-m\le j\le m, j \ne 0 }p(w_{t+j}|w_t)
$$

가 되고,

minimize 하기 위한 형태로 바꾸고 $$log$$



minimize 



손실함수를 다음과 같이 정의한다.

$$ J(\theta)
= - \frac{1}{T}\sum_{t=1}^T\sum_{-m\le j\le m, j \ne 0 }
\log p(w_{t+j}|w_t) \tag 1
$$

$$
p(o|c) = \frac {exp(\mathbf{u}_o \mathbf{v}_c)}
               {\sum_{w=1}^v exp(\mathbf{u}_w \mathbf{v}_c)} \tag 2
$$

$$(2)$$의 이유는 잘 안 나와있는데, 내가 생각한 흐름은 다음과 같다.

같은 컨텍스트에 있는 단어는 유사한 뜻을 가진다(distributional hypothesis)고 가정하고,

두 벡터의 내적은 벡터 간의 유사도를 러프하게 측정하므로(내적의 값이 클수록 유사한 벡터)

두 단어 벡터의 내적이 클수록 두 단어는 유사하며 같은 컨텍스트에 있을 확률이 높다.

그래서 $$word_{outside}$$ 벡터 (=$$u_o$$)와 $$word_{center}$$ 벡터 (=$$v_c$$)의 내적은 $$p(o \lvert c)$$와 비례한다.

최종적으로 normalize를 위해 $$softmax$$ 함수를 사용한다.

$$(2)$$가 수학적으로 $$=$$ 기호를 쓸 수 있는 건지 잘 모르겠다.




$$(1)$$을 최소화하기 위해 $$ \log p(w_{t+j}|w_t) $$ 를 미분한 값(gradient)을 사용해야 하므로
다음 식의 미분값을 구한다.

$$
  \log \frac {exp(\mathbf{u}_o \mathbf{v}_c)}
             {\sum_{w=1}^v exp(\mathbf{u}_w \mathbf{v}_c)}
$$

변화시키고자 하는 파라미터는 $$\mathbf{u}_o$$와 $$\mathbf{v}_c$$이므로 이에 대해 편미분을 한다.

먼저 $$\mathbf{v}_c$$ 에 대하여,

$$

  \frac {\partial}{\partial \mathbf{v}_c}
    \left(\log \frac {exp(\mathbf{u}_o \mathbf{v}_c)}
             {\sum_{w=1}^v exp(\mathbf{u}_w \mathbf{v}_c)}\right)

$$

$$
  \begin{align}
  & = \frac {\partial}{\partial \mathbf{v}_c}
    \left(
      \log exp(\mathbf{u}_o \mathbf{v}_c)
      -
      \log \sum_{w=1}^v exp(\mathbf{u}_w \mathbf{v}_c)
    \right)
  \\[2ex]
  & = \frac {\partial}{\partial \mathbf{v}_c}
    \log exp (\mathbf{u}_o \mathbf{v}_c)
    -
    \frac {\partial}{\partial \mathbf{v}_c}
    \log \sum_{w=1}^v exp(\mathbf{u}_w \mathbf{v}_c)
  \\[2ex]
  & = \frac {\partial}{\partial \mathbf{v}_c}
    (\mathbf{u}_o \mathbf{v}_c)
    -
    \frac {\partial}{\partial \mathbf{v}_c}
    \log \sum_{w=1}^v exp(\mathbf{u}_w \mathbf{v}_c)
  \\[2ex]
  & = \mathbf{u}_o
  -
  \frac {\partial}{\partial \mathbf{v}_c}
  \log \sum_{w=1}^v exp(\mathbf{u}_w \mathbf{v}_c)
  \\[2ex]
  & = \mathbf{u}_o
  -
  \frac {1}{\sum_{w=1}^v exp(\mathbf{u}_w \mathbf{v}_c)}
  \cdot
  \frac {\partial}{\partial \mathbf{v}_c}
  \sum_{x=1}^v exp(\mathbf{u}_x \mathbf{v}_c)
  \\[2ex]
  & = \mathbf{u}_o
  -
  \frac {1}{\sum_{w=1}^v exp(\mathbf{u}_w \mathbf{v}_c)}
  \cdot
  \sum_{x=1}^v
  \frac {\partial}{\partial \mathbf{v}_c}
  exp(\mathbf{u}_x \mathbf{v}_c)
  \\[2ex]
  & = \mathbf{u}_o
  -
  \frac {1}{\sum_{w=1}^v exp(\mathbf{u}_w \mathbf{v}_c)}
  \cdot
  \left[
    \sum_{x=1}^v
    exp(\mathbf{u}_x \mathbf{v}_c)
    \cdot
    \frac {\partial}{\partial \mathbf{v}_c}
    \mathbf{u}_x \mathbf{v}_c
  \right]
  \\[2ex]
  & = \mathbf{u}_o
  -
  \frac {1}{\sum_{w=1}^v exp(\mathbf{u}_w \mathbf{v}_c)}
  \cdot
  \left[
    \sum_{x=1}^v
    exp(\mathbf{u}_x \mathbf{v}_c)
    \cdot
    \mathbf{u}_x
  \right]
  \\[2ex]
  & = \mathbf{u}_o
  -
  \sum_{x=1}^v
  \frac {exp(\mathbf{u}_x \mathbf{v}_c)}
        {\sum_{w=1}^v exp(\mathbf{u}_w \mathbf{v}_c)}
  \cdot \mathbf{u}_x
  \\[2ex]
  & = \underbrace{\mathbf{u}_o}_{observed}
  -
  \underbrace{
    \sum_{x=1}^v
    p(x|c)
    \cdot \mathbf{u}_x
  }_{expectation}
  \\[2ex]

  \end{align}
$$



$$
\begin{align}
J(\theta)
& = \prod_{t=1}^T\prod_{-m\le j\le m, j \ne 0 } p(w_{t+j}|w_t) \tag 1\\
& = p(\mathtt{S|I})\, \times p(\mathtt{really|I})\, \times \\
& \,\,\,\,\,\, p(\mathtt{I|really}) \times p(\mathtt{like|really})\, \times \\
& \,\,\,\,\,\, p(\mathtt{really|like}) \times p(\mathtt{you|like})\, \times \\
& \,\,\,\,\,\, p(\mathtt{like|you}) \times p(\mathtt{E|like})
\end{align}
$$











$$a^2 + b^2 = c^2$$

$$ \mathsf{Data = PCs} \times \mathsf{Loadings} $$

\\[ \mathbf{X} = \mathbf{Z} \mathbf{P} \\]

$$ \mathbf{X}\_{n,p} = \mathbf{A}\_{n,k} \mathbf{B}\_{k,p} $$


https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference
