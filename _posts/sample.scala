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