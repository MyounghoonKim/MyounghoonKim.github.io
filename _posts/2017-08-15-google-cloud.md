---
layout: post
title: "구글 클라우드 플랫폼으로 장고 어플리케이션 만들기"
date: 2017-08-15 00:00:00 +0900
categories:
---

## 구글 클라우드 가입
지금 (2017/8/15) 새로 가입하면 12개월간 $300을 무료로 쓸 수 있도록 해주므로 써보도록 하자.
가입은 알아서..

## 로드 밸런서 런칭
- [빠르게 훑어보는 구글 클라우드 플랫폼](https://www.google.co.kr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjGm7Tr4vHVAhVIi7wKHZDqDGcQFggmMAA&url=http%3A%2F%2Fwww.hanbit.co.kr%2Fstore%2Fbooks%2Flook.php%3Fp_code%3DE5359426070&usg=AFQjCNHq0HAwPPCmluXb_rmWPpAAmVLybg) (이하 **빠르게..**) 4장 참조

### Compute Engine 생성
- `Compute Engine` -> 상단 `Create Instance`
- `Machine Type`은 적당히. `small`정도면..
- `Boot disk`는 `Cent OS 7`
- `Identity and API access`는 `Allow full access to all Cloud APIs`
  - 나중에 CLI 쓸 때 필요함
- `Firewall`은 `HTTP`와 `HTTPS` 모두 `Allow`
- 참고로 언어는 영어로 한다. 검색도 편하고.. 익숙해져야 한다.
- `Create`으로 생성

## SSH 연결

### 키 생성
- 로컬에서 `local$ ssh-keygen -t rsa -f ~/.ssh/[username]_rsa -C [username]`로 공개키와 개인키를 생성한다.

### 공개키 등록
- `local$ cat ~/.ssh/[username]_rsa.pub`를 복사해서 `Compute Engine` -> `Metadata` -> `SSH Keys`에 복사해 넣는다.

### 접속
- `local$ ssh -o ServerAliveInterval=10 -i ~/.ssh/[username]_rsa [username]@[STATIC_IP]`

## 기본프로그램 설치
- [https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-centos-7](https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-centos-7), [https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-centos-7](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-centos-7) 참조
```shell
// python
sudo yum -y update
sudo yum -y install yum-utils
sudo yum -y groupinstall development
sudo yum -y install https://centos7.iuscommunity.org/ius-release.rpm
sudo yum -y install python35u
sudo yum -y install python35u-pip
sudo yum -y install python35u-devel
// ngingx
sudo yum install epel-release
sudo yum install nginx
sudo systemctl start nginx
```
- 브라우저로 접속하여 확인

## 스냅샷 생성 / 로드밸런서 생성
**빠르게..** 참조

## DNS 설정

### GoDaddy에서 도메인 구매
- 알아서..

### IP 매칭
- https://kr.godaddy.com/community/Managing-Domains/linking-my-domain-to-a-google-cloud-project/td-p/13086

### 확인
- 등록한 도메인으로 확인. 요즘에는 빨리 된다. 30초 정도 기다렸음.

## SSL 등록

[https://www.digitalocean.com/community/tutorials/how-to-configure-ssl-termination-on-digitalocean-load-balancers](https://www.digitalocean.com/community/tutorials/how-to-configure-ssl-termination-on-digitalocean-load-balancers)

[https://certbot.eff.org/all-instructions/#centos-rhel-7-nginx](https://certbot.eff.org/all-instructions/#centos-rhel-7-nginx)