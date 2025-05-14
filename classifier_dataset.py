import sys
import pandas as pd
sys.path.remove('c:\\Users\\you0m\\nlp2025-1')

from datasets import load_dataset
# SST-5 데이터 로드
dataset = load_dataset("SetFit/sst5")

# 데이터 확인 (train, test, validation 나눠져 있음)
print(dataset)



# 폴더 생성
import os
os.makedirs("data", exist_ok=True)

# Train 데이터 저장
pd.DataFrame(dataset['train'])[['text', 'label']].rename(
    columns={'text': 'sentence', 'label': 'sentiment'}
).to_csv("data/ids-sst-train.csv", sep='\t', index=False)

# Dev 데이터 저장
pd.DataFrame(dataset['validation'])[['text', 'label']].rename(
    columns={'text': 'sentence', 'label': 'sentiment'}
).to_csv("data/ids-sst-dev.csv", sep='\t', index=False)
