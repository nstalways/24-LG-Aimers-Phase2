# built-in library
import re
from typing import List

# basic library
import numpy as np
import pandas as pd

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(tr_path: str = "./Data/train.csv",
              tt_path: str = "./Data/submission.csv") -> tuple:
    """학습 및 테스트 데이터를 불러옵니다.

    Args:
        tr_path (str, optional): 학습용 데이터의 경로입니다. Defaults to "./Data/train.csv".
        tt_path (str, optional): 테스트용 데이터의 경로입니다. Defaults to "./Data/submission.csv".

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame)
    """
    
    tr_data = pd.read_csv(tr_path)
    tr_data.drop_duplicates(inplace=True) # 중복 제거

    tt_data = pd.read_csv(tt_path)

    return (tr_data, tt_data)


def label_encoding(tr_data: pd.DataFrame, tt_data: pd.DataFrame,
                   features: list = ["customer_country", "business_subarea",
                                     "business_area", "business_unit", "customer_type",
                                     "enterprise", "customer_job",
                                     "inquiry_type", "product_category",
                                     "product_subcategory", "product_modelname",
                                     "customer_country.1", "customer_position",
                                     "response_corporate","expected_timeline"]) -> tuple:
    """범주형 데이터를 수치형 데이터로 encoding 하는 함수입니다.

    Args:
        tr_data (pd.DataFrame): 학습 데이터입니다.
        tt_data (pd.DataFrame): 테스트 데이터입니다.
        features (list, optional): 학습 데이터의 feature 중 범주형 features 의 이름을 담은 리스트입니다.

    Returns:
        tuple: label encoding을 마친 train, test DataFrame을 반환합니다.
    """
    # train / test data 복사
    x_tr = tr_data.copy()
    x_tt = tt_data.copy()

    for f in features:
        # 데이터 타입이 object (str) 일 때 label encoding 수행
        if x_tr[f].dtype.name == 'object':
            le = LabelEncoder()

            # train + test 데이터를 합쳐서 label encoding
            cur_tr_f = list(x_tr[f].values)
            cur_tt_f = list(x_tt[f].values)

            le.fit(cur_tr_f + cur_tt_f)

            x_tr[f] = le.transform(cur_tr_f)
            x_tt[f] = le.transform(cur_tt_f)

    return x_tr, x_tt


def split_train_and_validation(tr_data: pd.DataFrame, val_size: float = 0.2, seed: int = 2024) -> tuple:
    """주어진 data를 train / validation set으로 나누는 함수입니다.

    Args:
        tr_data (pd.DataFrame): split 할 data 입니다.
        val_size (float, optional): validation data의 비율입니다. Defaults to 0.2.
        seed (int, optional): sampling 시 사용할 seed 값입니다. Defaults to 42.

    Returns:
        tuple: (x_train, y_train, x_validation, y_validation) 을 반환
    """
    
    x_tr, x_val, y_tr, y_val = train_test_split(tr_data.drop(columns=['is_converted'], axis=1),
                                                tr_data['is_converted'],
                                                test_size=val_size,
                                                random_state=seed,
                                                shuffle=True)
    
    return (x_tr, y_tr, x_val, y_val)


def delete_features(tr_data: pd.DataFrame, tt_data: pd.DataFrame,
                    features: list = ['id_strategic_ver', 'it_strategic_ver', 'product_modelname', 'ver_cus', 'ver_pro']) -> tuple:
    """
    주어진 데이터에서 features 에 속하는 feature column 들을 삭제한 뒤 반환합니다.

    Args:
        tr_data (pd.DataFrame): training data 입니다.
        tt_data (pd.DataFrame): test data 입니다.
        features (list, optional): 삭제할 feature list 입니다. Correlation이 높거나 general한 정보를 담고있지 않다고 판단한 features를 기본적으로 삭제합니다.

    Returns:
        tuple: features 를 제거한 (tr_data, tt_data) 를 반환합니다.
    """
    
    tr_data = tr_data.drop(columns=features, axis=1)
    tt_data = tt_data.drop(columns=features, axis=1)

    return (tr_data, tt_data)


def extract_country_name(tr_data: pd.DataFrame, tt_data: pd.DataFrame) -> tuple:
    """customer_country feature로부터 국가명을 추출하여
    주어진 dataframe의 country라는 새로운 feature에 할당하는 함수입니다.

    Args:
        tr_data (pd.DataFrame): training data 입니다.
        tt_data (pd.DataFrame): test data 입니다.

    Returns:
        tuple:
            customer_country, customer_country.1 feature는 삭제되고
            country features는 추가된 (tr_data, tt_data) 를 반환합니다.
    """
    for df in [tr_data, tt_data]:
        nan_val = df[df.isna()].loc[0][0] # 결측값 가져오기

        countries = [] # 추출한 국가명을 저장할 배열
        for name in df['customer_country']:
            flag = False
            try:
                name = name.lower()
                res = name.split("/")
                if re.search("@", res[-1]) or re.search("[0-9]", res[-1]): # 비정상 데이터 예외처리
                    flag = True
                
                else:
                    countries.append(res[-1].strip())

            except AttributeError: # nan value 예외처리
                flag = True

            if flag:
                countries.append(nan_val)

        df['country'] = countries
        df.sort_index(axis=1, inplace=True)

    # correlation이 높은 customer_country, customer_country.1 feature 삭제
    tr_data, tt_data = delete_features(tr_data, tt_data, features=['customer_country', 'customer_country.1'])

    return (tr_data, tt_data)


def regroup(tr_data: pd.DataFrame, tt_data: pd.DataFrame, 
            feature_name: str, regroup_info: List[List],
            except_val: str='others', except_thr: int = 0) -> tuple:
    """regroup_info를 바탕으로 data[feature_name]의 값들을 regroup합니다.

    Args:
        tr_data (pd.DataFrame): training data입니다.
        tt_data (pd.DataFrame): test data입니다.
        feature_name (str): regroup을 적용할 feature의 이름입니다.
        regroup_info (List[List]): regroup 정보입니다. 각각의 리스트는 하나의 새로운 그룹을 의미합니다.
        except_val (str): except_thr 이하만큼 등장하는 값을 처리할 때 사용할 값입니다.
        except_thr (int): 최소 등장 횟수입니다.

    Returns:
        tuple: regroup을 마친 tr_data, tt_data를 반환합니다.
    """
    # 데이터를 연결
    data = pd.concat([tr_data, tt_data])

    # value별 등장 횟수 사전 생성
    freq = data[feature_name].value_counts().to_dict()

    # regroup_info를 바탕으로 regroup 수행
    regroup_results = []
    for val in data[feature_name].values:
        if type(val) == float: # 결측치
            regroup_results.append(val)
            continue
        
        flag = True
        for group_pool in regroup_info:
            if val in group_pool:
                regroup_results.append(group_pool[0].lower())
                flag = False
                break
        
        if flag:
            if freq[val] <= except_thr:
                regroup_results.append(except_val.lower())
            else:
                regroup_results.append(val)

    # 데이터 분리
    data[feature_name] = regroup_results
    tr_data, tt_data = data.iloc[:len(tr_data)].drop(['id'], axis=1), data.iloc[len(tr_data):]

    return tr_data, tt_data