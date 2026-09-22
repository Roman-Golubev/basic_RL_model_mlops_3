import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_INPUT_DIR = BASE_DIR / "artifacts" / "data_input"
DATA_OUTPUT_DIR = BASE_DIR / "artifacts" / "data_output"

API_NUM = int(os.getenv("API_NUM", "1"))


API_CONFIGS = {
    1: {
        "output_dir": "first_api_results",
        "params_file": "params_1.csv",
        "output_file": "output_1.pkl"
    },
    2: {
        "output_dir": "second_api_results",
        "params_file": "params_2.csv",
        "output_file": "output_2.pkl"
    },
    3: {
        "output_dir": "third_api_results",
        "params_file": "params_3.csv",
        "output_file": "output_3.pkl"
    },
    4: {
        "output_dir": "fourth_api_results",
        "params_file": "params_4.csv",
        "output_file": "output_4.pkl"
    },
    5: {
        "output_dir": "fifth_api_results",
        "params_file": "params_5.csv",
        "output_file": "output_5.pkl"
    },
    6: {
        "output_dir": "sixth_api_results",
        "params_file": "params_6.csv",
        "output_file": "output_6.pkl"
    }
}
