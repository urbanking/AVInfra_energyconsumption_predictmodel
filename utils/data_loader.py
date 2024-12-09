# utils/data_loader.py

import yaml

def load_data(file_path):
    """
    주어진 파일 경로에서 데이터를 로드하는 함수입니다.
    현재 예제에서는 데이터 파일이 없지만,
    이 함수는 확장성을 위해 포함되었습니다.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def preprocess_data(data):
    """
    로드된 데이터를 전처리하는 함수입니다.
    필요한 경우 데이터의 형식을 변환하거나
    누락된 값을 처리할 수 있습니다.
    """
    # 예제에서는 특별한 전처리 과정을 수행하지 않습니다.
    return data
