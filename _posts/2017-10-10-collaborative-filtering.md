---
layout: post
title:  "추천 시스템 리서치"
date:   2017-10-10 00:00:00 +0900
categories:
---

# 추천 시스템 리서치
- 추천 시스템은 기본적으로 협업 필터링과 콘텐트 기반 필터링을 기반으로 한다.

## 협업 필터링(Collaborative Filtering)
- CF란 대규모의 기존 사용자 행동 정보를 분석하여 해당 사용자와 비슷한 성향의
사용자들이 기존에 좋아했던 항목을 추천하는 기술이다.
  - 결과가 직관적, **항목의 구체적인 내용을 분석할 필요가 없다.**
  - SVD, PCA 등으로 차원축소
  - cosim 등으로 유사도 측정
  - k-NN 등으로 군집화
  - 단점
    - 기존 자료가 반드시 필요하다. 즉 신곡 같은 경우 추천이 어렵다. => *콜드스타트 문제*
    - 비인기 항목일 경우 위와 마찬가지로 기존자료가 적어 추천이 어렵다. => *롱테일 문제*

## 콘텐트 기반 필터링
- pass

## 현대적 협업 필터링
- 최근에는 기존 협업 필터링을 고도화한 모델 기반 협업 필터링이 사용된다. 그 중에서 잠재(latent) 모델에 기반을 둔 방법이 많이 사용된다.
  - 일반적으로 잠재 모델이란 드러난 피쳐에 영향을 주는 드러나지 않은(잠재된) 피쳐를 찾아내는 것을 이야기한다.
  - 정확도 상승, 추천이유 명확
  - 계산량이 기하급수적으로 증가
  - LDA, 베이지안 네트워크, 딥러닝 사용
  - 스포티파이도 딥러닝 사용. 잠재요인으로 만들어낸 서브 장르가 1378개라고 함.

## 스포티파이
- 1387개의 서브 장르(잠재요인)
  - ex) Pop, Jazz 가 아니라 Pop Christmas, Deep Sunset Lounge, Acoustic morning
  - 알고리즘만이 아닌 human touch가 들어간다고 함(우리 스타일?)
- 사용자들의 플레이리스트와 취향 프로파일을 가지고 추천 리스트를 만듬
- 카프카 사용
- 음악 블로그와 플레이리스트 제목을 NLP로 분석하여 협업 필터링(이하 CF)와 합쳐서 분석
- 매일 1 TB 가량의 유저 로그가 쌓인다고 함(2015년)
- 1700 하둡 노드(2015년)

http://www.kocca.kr/insight/vol05/vol05_04.pdf(한글)
https://brunch.co.kr/@veloso/2(한글)
https://brunch.co.kr/@hmin0606/7(한글)
https://qz.com/571007/the-magic-that-makes-spotifys-discover-weekly-playlists-so-damn-good/(영문)



https://www.slideshare.net/MrChrisJohnson/from-idea-to-execution-spotifys-discover-weekly
https://benanne.github.io/2014/08/05/spotify-cnns.html
https://www.forbes.com/sites/quora/2017/02/20/how-did-spotify-get-so-good-at-machine-learning/#6f170201665c
http://www.businessinsider.com/inside-spotify-and-the-future-of-music-streaming
http://www.businessinsider.com/how-spotify-taste-profiles-work-2015-9
https://www.thestar.com/entertainment/2016/01/14/meet-the-man-classifying-every-genre-of-music-on-spotify-all-1387-of-them.html
http://www.businessinsider.com/inside-spotify-and-the-future-of-music-streaming
