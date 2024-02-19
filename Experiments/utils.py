# built-in library
import os
import random
from datetime import datetime

# basic library
import numpy as np
import pandas as pd

# torch
import torch

# evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import ConfusionMatrixDisplay

# visualization
import matplotlib.pyplot as plt


def set_seed(seed: int = 2024) -> None:
    """실험 재현을 위해 seed를 설정하는 함수입니다.

    Args:
        seed (int, optional): 설정할 seed 값. Defaults to 2024.
    """
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# TODO: 아카이빙용 네이밍 규칙 변경될 수 있음
def make_submission(dir_name: str, y_pred: np.ndarray,
                    f1_val: float = None, model_name: str = None, 
                    sub_path: str = "./Data/submission.csv") -> None:
    """제출용 csv 파일을 생성하는 함수입니다.

    제출용 csv 파일은 항상 submission.csv 라는 이름으로 생성되며,
    아카이빙용 csv 파일은 {연월일}_0_{validation f1 score 소수점 이하 숫자}.csv 의 형태로 생성됩니다.

    Args:
        dir_name (str): 아카이빙할 디렉토리의 이름입니다.
        y_pred (np.ndarray): test data에 대한 예측 값입니다.
        f1_val (float): validation data에 대한 f1 score입니다. Defaults to None
        model_name (str): csv 파일 이름을 생성할 때 사용할 모델의 이름입니다. Defaults to None
        sub_path (str, optional): test data의 경로입니다. Defaults to "./Data/submission.csv".
    """
    
    # test data를 불러와서, 예측 결과를 'is_converted' column에 저장
    df_sub = pd.read_csv(sub_path)
    df_sub['is_converted'] = y_pred

    # 대회 제출용 csv 파일 생성 (파일 이름이 항상 submission.csv여야 함)
    df_sub.to_csv("submission.csv", index=False)

    # 제출용 파일과 별개로, 보관을 위한 csv 파일 생성
    dir_path = os.path.join("Results", dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    csv_name = datetime.now().strftime("%Y%m%d")

    if f1_val: csv_name += ("_0_" + str(f1_val)[2:])
    else: csv_name += "_none"

    if model_name: csv_name += ("_" + model_name + ".csv")
    else: csv_name += ".csv"

    df_sub.to_csv(os.path.join(dir_path, csv_name), index=False)


# TODO: 수정이 들어갈 수도 있음
def get_clf_eval(y_test: np.ndarray, y_pred: np.ndarray = None):
    """classifier 평가 결과를 출력하는 함수입니다.

    Args:
        y_test (np.ndarray): 정답 데이터입니다.
        y_pred (np.ndarray, optional): 모델의 예측 결과 데이터입니다. Defaults to None.
    """
    confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, labels=[True, False])

    # visualize confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion,
                                  display_labels=[True, False])
    disp.plot()
    plt.show()

    print("\n정확도: {:.4f}".format(accuracy))
    print("정밀도: {:.4f}".format(precision))
    print("재현율: {:.4f}".format(recall))
    print("F1: {:.4f}".format(f1))


# TODO: 실험 setting과 evaluation score 들을 dataframe 형태로 누적해주는 함수 구현하기
def record_experimental_results(model_name: str,
                                test_f1_score: str,
                                description: str) -> None:

    path = "./experimental_records.csv"

    # 새로운 기록 생성
    new_record = pd.DataFrame({
        'date': [datetime.now().strftime("%Y-%m-%d/%H:%M:%S")],
        'model_name': [model_name], 
        'test_f1_score': [test_f1_score],
        'description': [description]})
    
    # 기존 기록이 있는지 확인
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df._append(new_record, ignore_index=True)

    else:
        df = new_record

    df.to_csv(path, index=False)
