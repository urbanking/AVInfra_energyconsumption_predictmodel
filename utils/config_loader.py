# utils/config_loader.py

import yaml

def load_config(config_path):
    """
    설정 파일을 로드하는 함수
    Parameters:
    - config_path: 설정 파일 경로 (str)
    Returns:
    - config: 설정 내용 (dict)
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# ...existing code...
