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
    tr_data.drop_duplicates(inplace=True)

    tt_data = pd.read_csv(tt_path)

    return (tr_data, tt_data)


# TODO: 원본 baseline 코드와 label encoding 방식을 다르게 했더니 test 성능 차이가 꽤나 많이 남 (혹시 몰라 기록해둠)
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
                    features: list = ['com_reg_ver_win_rate', 'customer_type', 'customer_country.1',
                                      'historical_existing_cnt', 'id_strategic_ver',
                                      'it_strategic_ver', 'idit_strategic_ver','product_subcategory',
                                      'product_modelname', 'expected_timeline', 'ver_win_rate_x',
                                      'ver_win_ratio_per_bu', 'business_area','business_subarea']) -> tuple:
    """
    주어진 데이터에서 features 에 속하는 feature column 들을 삭제한 뒤 반환합니다.

    Args:
        tr_data (pd.DataFrame): training data 입니다.
        tt_data (pd.DataFrame): test data 입니다.
        features (list, optional): 삭제할 feature list 입니다. 
        기본값은 결측치 비율이 50 % 이상인 feature 들 + 중복 feature 입니다.

    Returns:
        tuple: features 를 제거한 (tr_data, tt_data) 를 반환합니다.
    """
    
    tr_data = tr_data.drop(columns=features, axis=1)
    tt_data = tt_data.drop(columns=features, axis=1)

    return (tr_data, tt_data)


def normalize_country_name(series: pd.Series):
    def normalize(country_info):
        if type(country_info) == float:
            return country_info

        return country_info.split('/')[-1]
    
    return series.apply(normalize)


# TODO: 결측치 처리 함수 구현하기