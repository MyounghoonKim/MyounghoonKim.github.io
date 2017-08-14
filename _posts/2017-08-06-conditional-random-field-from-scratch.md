---
layout: post
title:  conditional random field from scratch
date:   2017-08-06 00:00:01 +0900
categories: nlp
---

일단 [여기](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/) 예제 구현부터 해 보자.

http://www.lsi.upc.edu/~aquattoni/AllMyPapers/crf_tutorial_talk.pdf

POS Tagging

$$
\begin{array}
& x_1 & x_2 & x_3 & x_4 & x_5 & x_6 & x_7 & x_8 & x_9\\
\mathsf{He} & \mathsf{reckons} & \mathsf{the} & \mathsf{current} & \mathsf{account} & \mathsf{deficit} & \mathsf{will} & \mathsf{narrow} & \mathsf{significantly.}\\
y_1 & y_2 & y_3 & y_4 & y_5 & y_6 & y_7 & y_8 & y_9\\
[PRP] & [VB] & [DT] & [JJ] & [NN] & [NN] & [MD] & [VB] & [RB]\\
\end{array}
$$
