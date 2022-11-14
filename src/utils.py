from typing import Dict, List
import json
import openpyxl
import csv
import numpy as np
import torch
import random



class FileIO:
    @staticmethod
    def read_text(filename):
        with open(filename, "r", encoding="utf8") as f:
            return f.read()

    @staticmethod
    def write_text(data, filename):
        with open(filename, "w", encoding="utf8") as f:
            f.write(data)

    @staticmethod
    def append_text(data: str, filename):
        with open(filename, "a", encoding="utf8") as f:
            f.write("\n" + data)

    @staticmethod
    def read_json(filename):
        with open(filename, "r", encoding="utf8") as f:
            return json.load(f)

    @staticmethod
    def write_json(data, filename):
        with open(filename, "w", encoding="utf8") as f:
            json.dump(data, f, default=str)

    @staticmethod
    def read_excel(filename, sheet_name="Sheet1"):
        wb_obj = openpyxl.load_workbook(filename)
        return wb_obj[sheet_name]

    @staticmethod
    def read_csv(filename: str) -> List[Dict]:
        with open(filename, "r", encoding="utf8") as f:
            csvreader = csv.reader(f)
            header = next(csvreader)
            return [
                {h: x for h, x in zip(header, row) if h}
                for row in csvreader
            ]

    @staticmethod
    def write_numpy(filename: str, array: np.array) -> None:
        with open(filename, 'wb') as f:
            np.save(f, array)

    @staticmethod
    def read_numpy(filename: str) -> None:
        with open(filename, 'rb') as f:
            return np.load(f)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False