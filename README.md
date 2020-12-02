# 목적
    수업받은 학생들에게 다른 튜터를 추천해 알고리즘 시스템 개발

# 언어 및 환경
    언어 : Python3.7
    프레임워크 및 테스트 환경 : Flask, Swagger
    

# 설치 패키지
    pip3 install pandas
    pip3 install sklearn
    pip3 install flask-restplus
    pip3 install werkzeug==0.16.0
    
# 데이터 경로(파일명 중요)
    튜터정보(첨부파일) : data/수업 & 튜터 데이터 (과제용) - 튜터 데이터.csv
    강의정보 : data/수업 & 튜터 데이터 (과제용) - 수업 데이터.csv

# 테스트 방법
    1. 필요 패키지 설치 후 app.py 실행 - python3 app.py
    2. http://127.0.0.1:5000 접속 - swagger
    3. apk /curation/curation 에서 parameter 설정
    4. user : 학생 ID, recommend_count : 튜터 추천 수 (과제조건은 3)
    5. Execute 후 실행
    
# 결과(협업 필터링 구현)
    knn_recommend_tutor : Kdd 유사도 유클리디안 거리 추천 (유저 베이스)
    cos_sim_recommend_tutor : 코사인 유사도 기반 추천 (튜터 베이스)

# 기타
    튜터정보에 임의로 연령대 데이터 추가 하였습니다.
    
   