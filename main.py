import pandas as pd
from math import sqrt
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

csv_class = './data/수업 & 튜터 데이터 (과제용) - 수업 데이터.csv'
csv_tutor = './data/수업 & 튜터 데이터 (과제용) - 튜터 데이터.csv'

def sim_distance(data, n1, n2):
    sum = 0
    # 두 사용자가 모두 본 영화를 기준으로 해야해서 i로 변수 통일(j따로 안 써줌)
    for i in data.loc[n1, data.loc[n1, :] >= 0].index:
        if data.loc[n2, i] >= 0:
            sum += pow(data.loc[n1, i] - data.loc[n2, i], 2)  # 누적합
    return sqrt(1 / (sum + 1))  # 유사도 형식으로 출력





class curationMain :
    def __init__(self):
        print('init')

    def main(self):
        # 유져 id
        student_id = 30406
        # 추천 순위 수
        rank_count = 3

        df_class_ori = pd.read_csv(csv_class)
        df_class_drop_lesid = df_class_ori.drop(columns=['Lesson ID'])
        df_class_uni_tutor = df_class_ori.drop_duplicates(['Tutor ID'])
        # print(df_class.index)
        # 수업을 다르지만 학생과 강사가 같은 경우 평점 평균 처리
        df_class = df_class_ori.groupby([df_class_ori['Student ID'], df_class_ori['Tutor ID']], as_index=False).mean()
        df_tutor = pd.read_csv(csv_tutor)

        df_merge = pd.merge(df_class, df_tutor, on='Tutor ID', how='left')
        df_pivot = df_merge.pivot(index='Student ID', columns='Tutor ID', values='튜터 평가 점수(7점 만점)')
        # nan 처리
        df_pivot.fillna(-1, inplace=True)

        # 강의 받은 교수 최신순(class id가 높으면 최근 수업이라고 가정)
        df_class_ori = df_class_ori.sort_values(by='Lesson ID', ascending=False)
        tutor_array = df_class_ori[df_class_ori['Student ID'] == student_id]

        # 튜터 중복제거
        tutor_array = tutor_array['Tutor ID'].unique()


        # 최근 교육받은 튜터 한명을 가져옴.
        tutor_id = tutor_array[0]


        # 추천 결과
        kdd_tutor_list = self.get_top_content_list(df_pivot, df_tutor, student_id)[0:3]
        kdd_tutor_list = [data[1] for data in kdd_tutor_list]

        print('Kdd 유사도 기반 영화 추천 : %s' % (kdd_tutor_list))

        df_tutor_mean_rating = df_class_drop_lesid.groupby([df_class_drop_lesid['Tutor ID']],
                                                           as_index=False).mean().round(decimals=2)

        # 튜터들의 평점 데이터
        df_tutor_mean_rating = pd.merge(df_class_uni_tutor, df_tutor_mean_rating, left_on='Tutor ID', right_on='Tutor ID')

        df_tutor_mean_rating = df_tutor_mean_rating.drop(columns=['Student ID_y', '튜터 평가 점수(7점 만점)_x']).rename(
            columns = {'Student ID_x': 'Student ID', '튜터 평가 점수(7점 만점)_y': 'mean_rating'})

        df_tutor = pd.merge(df_tutor, df_tutor_mean_rating, left_on='Tutor ID', right_on='Tutor ID', how='left')
        df_tutor['Accent'] = df_tutor['Accent'].fillna(-1).astype(int)
        df_tutor['Gender'] = df_tutor['Gender'].fillna(-1)
        df_tutor['Major'] = df_tutor['Major'].fillna(-1.0).astype(int)
        df_tutor['mean_rating'] = df_tutor['mean_rating'].fillna(-1.0)

        features = ['Accent', 'Gender', 'Major']
        cos_sim = cosine_similarity(df_tutor[features][df_tutor['Tutor ID'] == tutor_id], df_tutor[features])
        cos_sim = pd.Series(cos_sim[0], name='cos_sim')

        df_tutor = pd.concat([df_tutor, cos_sim], axis=1)

        # 튜터 id로부터 메타정보 get (사용자기반)
        knn_df_tutor = df_tutor[df_tutor['Tutor ID'].isin(kdd_tutor_list)].drop(columns='cos_sim')
        knn_df_tutor = knn_df_tutor.drop(columns=['Student ID'])
        # 강의 들었던 튜터는 제외 (상품기반) - 유사 > 평점 높은 순으로 정렬
        cos_sim_df_tutor = df_tutor[~df_tutor['Tutor ID'].isin(tutor_array)].sort_values(by=['cos_sim', 'mean_rating'], ascending=False)[
                           0:rank_count]
        cos_sim_df_tutor = cos_sim_df_tutor.drop(columns=['Student ID'])

        print(knn_df_tutor)
        print(cos_sim_df_tutor)
        return


    def cos_sim(self, A, B):
        return dot(A, B) / (norm(A) * norm(B))

    def top_match(self, data, name, rank=5, simf=sim_distance):
        simList = []
        for i in data.index[-10:]:
            if name != i:
                simList.append((simf(data, name, i), i))
        simList.sort()
        simList.reverse()
        return simList[:rank]

    # 추천 시스템 함수
    def recommendation(self, data, person):
        res = self.top_match(data, person, len(data))
        score_dic = {}
        sim_dic = {}
        myList = []
        for sim, name in res:
            if sim < 0:
                continue
            for movie in data.loc[person, data.loc[person, :] < 0].index:
                simSum = 0
                if data.loc[name, movie] >= 0:
                    simSum += sim * data.loc[name, movie]

                    score_dic.setdefault(movie, 0)
                    score_dic[movie] += simSum

                    sim_dic.setdefault(movie, 0)
                    sim_dic[movie] += sim
        for key in score_dic:
            myList.append((score_dic[key] / sim_dic[key], key))
        myList.sort()
        myList.reverse()

        return myList



    def get_top_content_list(self, contents, tutors, student_id):
        movieList = []
        # Kdd 유사도 기반 영화 추천 알고리즘
        for rate, m_id in self.recommendation(contents, student_id):
            movieList.append((rate, tutors.loc[tutors['Tutor ID'] == m_id, 'Tutor ID'].values[0]))
        print(movieList)

        return movieList



if __name__ == '__main__':
    cm = curationMain()
    cm.main()


