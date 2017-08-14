---
layout: post
title:  "pmi for NLP"
date:   2017-07-07 00:00:00 +0900
categories:
---
----
# 점별 상호정보량(PMI:Pointwise Mutual Information)
```
문서1: 페더러, 나달, 윔블던, 페더러, 잔디
문서2: 페더러, 나달, 나이키
문서3: 페더러, 롤렉스
문서4: 롤렉스, 론진
```

위와 같은 문서 분포가 있을 때 같은 문서에 자주 등장하는 단어들은 서로 연관성이 높다고 할 수 있다. 예를 들면 **페더러**와 **나달**의 연관성은 **페더러**와 **롤렉스**의 연관성보다 높아야 한다. 그리고 **롤렉스**와 **나달**은 연관성이 없어야 한다. $$pmi$$를 사용하면 두 단어 사이의 연관성을 구할 수 있으며 정의는 다음과 같다.

$${pmi} (x;y)\equiv \log {\frac {p(x,y)}{p(x)p(y)}}=\log {\frac {p(x|y)}{p(x)}}=\log {\frac {p(y|x)}{p(y)}}$$

$$pmi$$끼리의 비교에서는 굳이 $$log$$를 취할 필요가 없으므로 편의상 생략하고 이를 $$pmi_{simple}$$라고 하면

$$
pmi_{simple}(페더러, 나달) =
\frac{p(페더러, 나달)}
     {p(페더러) \times p(나달)}
$$

단어의 확률은 **df(document frequency)**로 구한다. 여기서는 전체 문서 갯수가 4개이고, 3개의 문서에서 **페더러**가 등장하므로, $$p(페더러)$$는 $$0.75$$가 된다. 마찬가지로 $$p(나달)$$은 $$0.5$$이다. 참고로, **df**에서는 한 문서에서 단어가 몇 번 나오든 상관없다.

$$p(페더러,나달)$$은 $$0.5$$이므로

$$pmi_{simple}(페더러, 나달) =
\frac{0.5}
     {0.75 \times 0.5}
= 1.334
$$


같은 방식으로

$$pmi_{simple}(페더러, 롤렉스) =
\frac   {0.25}
        {0.75 \times 0.5}
= 0.667
$$

다르게 생각해 보면 $$pmi$$는 두 단어가 같이 등장할 확률과 따로 등장할 확률의 비율을 보는 것이다. 같이 등장할 확률이 높을수록 $$pmi$$는 커진다 $$=$$ 연관성이 높다.

$$
\begin{align}
pmi(페더러, 나달) &  = \log 1.334 = 0.287 \\[2ex]
pmi(페더러, 롤렉스) & = \log 0.667 = -0.405
\end{align}
$$

<br><br>

----
# 상호정보량(MI: Mutual Information)

$$MI$$는 두 확률변수간의 상호의존성을 보는 척도인데, $$pmi$$의 관점에서 보면 $$pmi$$들의 기대값이다.

$${MI(X,Y)=\sum _{y\in Y}\sum _{x\in X}p(x,y)\log {\left({\frac {p(x,y)}{p(x)\,p(y)}}\right)}}$$

위의 $$pmi$$에서 구한 값은 단어가 등장할 확률로만 구한 값이다.

$$MI$$는 등장할 확률과 등장하지 않을 확률의 $$pmi$$를 모두 계산하여 구한다.

$$페더러$$와 $$나달$$로 확률표를 그려보면
(페더러를 $$F$$, 나달을 $$N$$, 등장을 $$=1$$, 미등장을 $$=0$$으로 표현한다)

$$
\begin{array}{l|r}
        & F=0 & F=1 & 합 \\
\hline
N=0   &  0.25     & 0.25       &   0.5\\
N=1   &   0    & 0.5     &  0.5\\
합      &  0.25  & 0.75    & 1.0
\end{array}
$$

$$
\begin{align}
pmi(F=0,N=0) & = log\frac{p(F=0,N=0)}{p(F=0) \times p(N=0)}
                = log\frac{0.25}{0.25 \times 0.5} & = 0.693\\[2ex]
pmi(F=0,N=1) & = log\frac{p(F=0,N=1)}{p(F=0) \times p(N=1)}
                & = 0\\[2ex]
pmi(F=1,N=0) & = log\frac{p(F=1,N=0)}{p(F=1) \times p(N=0)}
                = log\frac{0.25}{0.75 \times 0.5} & = -0.405\\[2ex]
pmi(F=1,N=1) & = log\frac{p(F=1,N=1)}{p(F=1) \times p(N=1)}
                = log\frac{0.5}{0.75 \times 0.5} & = 0.288\\[2ex]
\end{align}
$$

$$
\begin{align}
MI(F,N) & = 0.25 * 0.693 + 0 + 0.25 * -0.405 + 0.5 * 0.288
        & = 0.216
\end{align}
$$

<br><br>

$$페더러$$와 $$롤렉스$$로도 구해보면,

$$
\begin{array}{l|r}
        & F=0  & F=1 & 합 \\
\hline
R=0     &   0  & 0.5  & 0.5\\
R=1     & 0.25  & 0.25 & 0.5\\
합       & 0.25 & 0.75 & 1.0
\end{array}
$$

$$
\begin{align}
pmi(F=0,R=0) & = log\frac{p(F=0,R=0)}{p(F=0) \times p(R=0)}
                & = 0\\[2ex]
pmi(F=0,R=1) & = log\frac{p(F=0,R=1)}{p(F=0) \times p(R=1)}
                = log\frac{0.25}     {0.25 \times 0.5} & = 0.693\\[2ex]
pmi(F=1,R=0) & = log\frac{p(F=1,R=0)}{p(F=1) \times p(R=0)}
                = log\frac{0.5}{0.75 \times 0.5} & = 0.288\\[2ex]
pmi(F=1,R=1) & = log\frac{p(F=1,R=1)}{p(F=1) \times p(R=1)}
                = log\frac{0.25}{0.75 \times 0.5} & = -0.405\\[2ex]
\end{align}
$$

$$
\begin{align}
MI(F,R) & = 0 + 0.25 * 0.693 + 0.5 * 0.288 + 0.25 * -0.405
        & = 0.216
\end{align}
$$

<br><br>

$$MI(페더러, 나달)$$과 $$MI(페더러, 롤렉스)$$가 동일한 값을 가진다.

이는 실제 단어들의 연관성을 제대로 나타내 주지 못하므로

$$pmi(w_1=1, w_2=1)$$과 $$pmi(w_1=0, w_2=0)$$만 가지고 $$MI$$를 구하고

이를 $$MI_{polar}$$라고 하면

$$
\begin{align}
MI_{polar}(페더러,나달) & = 0.25 * 0.693 + 0.5 * 0.288
        & = 0.317 \\[2ex]
MI_{polar}(페더러,롤렉스) & = 0 + 0.25 * -0.405
        & = -0.101
\end{align}
$$

좀 더 현실적이 되었다.

또 다른 방법으로

$$pmi(w_1=1, w_2=1)$$만 가지고 $$MI$$를 구하고 이를 $$MI_{naive}$$라고 하면

$$
\begin{align}
MI_{naive}(페더러, 나달) & = 0.5 \times 0.287 & = 0.144\\[2ex]
MI_{naive}(페더러, 롤렉스) & = 0.25 \times -0.405 & = -0.101
\end{align}
$$

이 또한 현실적이다.

<br><br>

정리해 보면,

$$
\begin{array}{l|r}
word_1, word_2 & pmi_{simple} & MI_{simple} & pmi & MI_{naive} & MI_{polar} & MI & \#\,pair & \#\,word_1 & \#\,word_2 \\
\hline
\text{윔블던/잔디} & 4.000 & 1.000 & 1.386 & 0.347 & 0.562 & 0.216 & 1 &  1 &  1\\
\text{나달/페더러} & 1.333 & 0.667 & 0.288 & 0.144 & 0.317 & 0.173 & 2 &  2 &  3\\
\text{나달/나이키} & 2.000 & 0.500 & 0.693 & 0.173 & 0.317 & 0.042 & 1 &  2 &  1\\
\text{나달/윔블던} & 2.000 & 0.500 & 0.693 & 0.173 & 0.317 & 0.042 & 1 &  2 &  1\\
\text{나달/잔디} & 2.000 & 0.500 & 0.693 & 0.173 & 0.317 & 0.042 & 1 &  2 &  1\\
\text{론진/롤렉스} & 2.000 & 0.500 & 0.693 & 0.173 & 0.317 & 0.144 & 1 &  1 &  2\\
\text{나이키/페더러} & 1.333 & 0.333 & 0.288 & 0.072 & 0.144 & 0.072 & 1 &  1 &  3\\
\text{윔블던/페더러} & 1.333 & 0.333 & 0.288 & 0.072 & 0.144 & 0.072 & 1 &  1 &  3\\
\text{잔디/페더러} & 1.333 & 0.333 & 0.288 & 0.072 & 0.144 & 0.072 & 1 &  1 &  3\\
\text{롤렉스/페더러} & 0.667 & 0.167 & -0.405 & -0.101 & -0.101 & 0.173 & 1 &  2 &  3
\end{array}
$$

$$MI_{simple}$$이 가장 연관성을 잘 반영하는데 실무에서도 $$MI_{simple}$$과 $$pmi_{simple}$$이 현실의 반영성, 눈으로 보기 편함, 계산상의 편리함 등의 이유로 많이 쓰인다.
연관어를 뽑을 때는 높은 순서대로 정렬한 후 일정 수치 이상만 뽑아서 연관어로 정리하는데 빈도수도 고려하여 이상 영역에 있는 것은 제거하는 것이 좋다. $$pmi$$ 자체를 변형하여 저빈도 단어에 penalty를 주는 방식도 사용되며 각 단어콤비의 샘플을 뽑아 연관어/비연관어로 태그를 단 다음 간단한 기계학습을 적용하여 적당한 가중치를 찾아낼 수도 있겠다.

<br><br>

----
# 쿨백-라이블러 발산(Kullback-Leibler divergence)

$$ D_{KL}(P\|Q)= \sum _{i}P(i)\log \frac{P(i)}{Q(i)} $$

**KL divergence**는 두 확률분포의 유사성을 판단하는 척도라고 생각할 수 있다. 베이지안의 관점에서 보면 사전분포 $$Q$$에서 사후분포 $$P$$로 믿음을 전환함에 따라 얻게 되는 정보의 양이고, 이는 $$Q$$가 $$P$$를 근사함에 따라 잃게 되는 정보의 양으로 생각할 수도 있다.

값이 클수록 덜 유사하며, 0이면 동일한 확률분포이다.

머신러닝에서 많이 쓰는데 실제 확률분포가 있고 우리가 예측한 분포가 있다고 할 때 이 두 분포를 비교하여 그 차이가 작도록(**KL divergence**가 작도록) 파라미터들을 조정하여 예측력을 높일 수 있다.

$$D_{KL}$$을 이용해서 문서간의 유사성을 측정할 수 있다.

`문서1(페더러,나달,페더러,윔블던,잔디)`과 `문서2(나달,페더러,나이키)`의 유사성과

`문서1(페더러,나달,페더러,윔블던,잔디)`과 `문서3(페더러,롤렉스)`의 유사성을 비교해보면,

$$
\begin{align}
D_{KL}(P_{doc1}\|P_{doc2}) & = P_{doc1}(페더러)log\frac{P_{doc1}(페더러)}{P_{doc2}(페더러)} \\[2ex]
& \,\,\,\,+ P_{doc1}(나달)log\frac{P_{doc1}(나달)}{P_{doc2}(나달)} \\[2ex]
& \,\,\,\,+ P_{doc1}(윔블던)log\frac{P_{doc1}(윔블던)}{P_{doc2}(윔블던)} \\[2ex]
& \,\,\,\,+ P_{doc1}(잔디)log\frac{P_{doc1}(잔디)}{P_{doc2}(잔디)} \\[3ex]
& =     0.400 \times log\frac{0.400}{0.333} + 0.200 \times log\frac{0.200}{0.333}
    +   0.200 \times log\frac{0.200}{0.001} + 0.200 \times log\frac{0.200}{0.001} \\[2ex]
& = 0.400 \times 0.183 + 0.200 \times -0.510 + 0.200 \times 5.299 + 0.200 \times 5.299 \\[2ex]
& = 2.090 \\[4ex]

D_{KL}(P_{doc1}\|P_{doc3}) & = P_{doc1}(페더러)log\frac{P_{doc1}(페더러)}{P_{doc3}(페더러)} \\[2ex]
& \,\,\,\,+ P_{doc1}(나달)log\frac{P_{doc1}(나달)}{P_{doc3}(나달)} \\[2ex]
& \,\,\,\,+ P_{doc1}(윔블던)log\frac{P_{doc1}(윔블던)}{P_{doc3}(윔블던)} \\[2ex]
& \,\,\,\,+ P_{doc1}(잔디)log\frac{P_{doc1}(잔디)}{P_{doc3}(잔디)} \\[3ex]
& =     0.400 \times log\frac{0.400}{0.500} + 0.200 \times log\frac{0.200}{0.001}
    +   0.200 \times log\frac{0.200}{0.001} + 0.200 \times log\frac{0.200}{0.001} \\[2ex]
& = 0.400 \times -0.223 + 0.200 \times 5.299 + 0.200 \times 5.299 + 0.200 \times 5.299 \\[2ex]
& = 3.089
\end{align}
$$

$$ P_{doc3}(잔디) $$ 는 0이지만 실제로 문서에서 어떤 단어가 나올 확률을 0이라고 볼 수는 없으므로 작은 값(여기서는 0.001)을 넣어 smoothing한다.

$$ D_{KL}(P_{doc1}\|P_{doc2}) < D_{KL}(P_{doc1}\|P_{doc3}) $$ 이므로

`(문서1,문서2)`가 `(문서1,문서3)`보다 유사하다.


<br><br>

----
# KL divergence와 상호정보량

확률변수 $$x$$, $$y$$가 주어졌을 때 이 두 변수가 독립이라면 $$p(x,y) = p(x)p(y)$$이다. 독립이 아니라면 독립성의 정도는 $$p(x,y)$$와 $$p(x)p(y)$$의 비율로 측정할 수 있다. $$p(x,y)$$는 **결합확률분포**, $$p(x)p(y)$$는 **주변확률분포의 곱**이므로 두 개의 확률분포의 유사도를 측정하는 **KL divergence**로 독립성의 정도를 파악할 수 있다.

$$ D_{KL}(P\|Q)= \sum _{i}P(i)\log \frac{P(i)}{Q(i)} $$ 에서 $$P = p(x,y), Q = p(x)p(y)$$로 바꾸면 $$ D_{KL}(p(x,y)\|p(x)p(y)) = \sum _{i}p(x,y)\log \frac{p(x,y)}{p(x)p(y)}, 즉 $$ **상호정보량**이 된다.

<br><br>

----
# 코드

```scala
// spark using scala
val docs = sc.parallelize(Array(
    Array("페더러", "나달", "페더러", "윔블던", "잔디"),
    Array("나달", "페더러", "나이키"),
    Array("페더러", "롤렉스"),
    Array("롤렉스", "론진")
))

val pairsCount = docs.flatMap { doc =>
    doc.combinations(2).filter { case x =>
        // 같은 단어끼리의 페어는 뺀다
        !x(0).equals(x(1))
    }.map { pair =>
        // 정렬
        if (pair(0) < pair(1))
            (pair(0) + "/" + pair(1), 1)
        else
            (pair(1) + "/" + pair(0), 1)
    }
}.reduceByKey(_ + _)

val words = docs.flatMap { doc =>
    doc.toSet.toArray.map(word => (word, 1))
}.reduceByKey(_ + _)

val docsCount = docs.count()

val result = pairsCount.map {case (pair, count) =>
    (pair.split("/")(0), (pair, count))
}.join(words).map { case (word1, ((pair, pairCount),word1Count)) =>
    // join된 결과를 보기 좋게 정리
    ((pair, pairCount), (word1, word1Count))
}.map { case ((pair, pairCount), (word1, word1Count)) =>
    (pair.split("/")(1), ((pair, pairCount), (word1, word1Count)))
}.join(words).map { case (word2,(((pair,pairCount),(word1,word1Count)),word2Count)) =>
    // join된 결과를 보기 좋게 정리
    ((pair, pairCount), (word1, word1Count), (word2, word2Count))
}.map { case ((pair, pairCount), (word1, word1Count), (word2, word2Count)) =>
    val pPair = pairCount.toFloat / docsCount
    val pWord1 = word1Count.toFloat / docsCount
    val pWord2 = word2Count.toFloat / docsCount

    val pNotWord1 = 1.0 - pWord1
    val pNotWord2 = 1.0 - pWord2
    val pWord1NotWord2 = pWord1 - pPair
    val pNotWord1Word2 = pWord2 - pPair
    val pNotWord1NotWord2 = 1.0 - pPair - pWord1NotWord2 - pNotWord1Word2

    assert(pWord1 == pWord1NotWord2 + pPair)
    assert(pWord2 == pNotWord1Word2 + pPair)
    assert(pNotWord1 == pNotWord1NotWord2 + pNotWord1Word2)
    assert(pNotWord2 == pNotWord1NotWord2 + pWord1NotWord2)

    val pmiSimple = pPair / (pWord1 * pWord2)
    val pmi = math.log(pmiSimple)
    val miSimple = pPair * pmiSimple
    val miNaive = pPair * pmi

    var pmiNotWord1NotWord2 = pNotWord1NotWord2 * math.log(pNotWord1NotWord2 / (pNotWord1 * pNotWord2))
    if (pmiNotWord1NotWord2.isNaN) pmiNotWord1NotWord2 = 0.0
    var pmiWord1NotWord2 = pWord1NotWord2 * math.log(pWord1NotWord2 / (pWord1 * pNotWord2))
    if (pmiWord1NotWord2.isNaN) pmiWord1NotWord2 = 0.0
    var pmiNotWord1Word2 = pNotWord1Word2 * math.log(pNotWord1Word2 / (pNotWord1 * pWord2))
    if (pmiNotWord1Word2.isNaN) pmiNotWord1Word2 = 0.0
    var pmiWord1Word2 = miNaive
    if (pmiWord1Word2.isNaN) pmiWord1Word2 = 0.0

    val miPolar = pmiNotWord1NotWord2 + pmiWord1Word2

    val mi = pmiNotWord1NotWord2 + pmiWord1NotWord2
            + pmiNotWord1Word2 +pmiWord1Word2

    (pair, pmiSimple, miSimple, pmi, miNaive, miPolar, mi, (pairCount, pPair), (word1, word1Count, pWord1), (word2, word2Count, pWord2))
}

val output = result.sortBy { case (pair, pmiSimple, miSimple, pmi, miNaive, miPolar, mi, (pairCount, pPair), (word1, word1Count, pWord1), (word2, word2Count, pWord2)) =>
    (-miSimple, pair)
}.map { case (pair, pmiSimple, miSimple, pmi, miNaive, miPolar, mi, (pairCount, pPair), (word1, word1Count, pWord1), (word2, word2Count, pWord2)) =>
    val pmiSimpleStr = f"$pmiSimple%2.3f"
    val miSimpleStr = f"$miSimple%2.3f"
    val pmiStr = f"$pmi%2.3f"
    val miNaiveStr = f"$miNaive%2.3f"
    val miPolarStr = f"$miPolar%2.3f"
    val miStr = f"$mi%2.3f"
    val raw = f"$pair%-15s\t$pmiSimpleStr%-10s\t$miSimpleStr%-10s\t$pmiStr%-10s\t$miNaiveStr%-10s\t$miPolarStr%-10s\t$miStr%-10s\t" + f"$pairCount%2d\t$word1Count%2d\t$word2Count%2d"
    val mathjax = f"\\text{$pair} & $pmiSimpleStr & $miSimpleStr & $pmiStr & $miNaiveStr & $miPolarStr & $miStr & $pairCount & $word1Count%2d & $word2Count%2d"
    (raw, mathjax)
}

val mathjax = output.map { case (raw, mathjax) => mathjax}.collect().mkString("\\\\\n")

val mathjaxResultStart = """
$$
\begin{array}{l|r}
word_1, word_2 & pmi_{simple} & MI_{simple} & pmi & MI_{naive} & MI_{polar} & MI & \#\,pair & \#\,word_1 & \#\,word_2 \\
\hline
"""

val mathjaxResultEnd = """
\end{array}
$$
"""

// mi
println(mathjaxResultStart + mathjax + mathjaxResultEnd)



val pDoc = docs.zipWithIndex().map { case (doc, docIdx) =>
    val docSize = doc.size
    (docIdx, doc.groupBy(identity).map{case (k,v) =>
        (k,(v.size.toFloat / docSize.toFloat, v.size, docSize))
    })
}
val doc1 = pDoc.filter { case (docIdx, doc) =>
    docIdx == 0
}.map { case (docIdx, doc) => doc }.first()
val doc2 = pDoc.filter { case (docIdx, doc) =>
    docIdx == 1
}.map { case (docIdx, doc) => doc }.first()
val doc3 = pDoc.filter { case (docIdx, doc) =>
    docIdx == 2
}.map { case (docIdx, doc) => doc }.first()

val doc1ToDoc2KLD = doc1.map { case (word, (prob, count, total)) =>
    val doc2Prob = doc2.getOrElse(word, (0.001f, 0, total))._1
    (prob * math.log(prob / doc2Prob))
}.reduceLeft(_ + _)

val doc1ToDoc3KLD = doc1.map { case (word, (prob, count, total)) =>
    val doc3Prob = doc3.getOrElse(word, (0.001f, 0, total))._1
    (prob * math.log(prob / doc3Prob))
}.reduceLeft(_ + _)

// KL divergence between 2 docs
println(doc1ToDoc2KLD)
println(doc1ToDoc3KLD)
```

실무에서는 더 잡다한(..) 클렌징 코드나 효율성을 위한 트릭들이 많이 들어간다.