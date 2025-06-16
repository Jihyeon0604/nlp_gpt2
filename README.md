# 🧠 GPT-2 기반 자연어처리 태스크 실험 프로젝트

본 프로젝트는 OpenAI의 GPT-2 구조를 기반으로 직접 모델을 구현하고, 다양한 Fine-Tuning 전략을 적용하여 **감정 분석**, **패러프레이즈 탐지**, **시 생성** 세 가지 자연어처리(NLP) 태스크에 적용한 실험을 수행하였습니다.

주요 목표는 GPT-2 decoder-only 구조의 특성을 이해하고, 이를 다양한 downstream task에 효과적으로 활용하기 위한 fine-tuning 전략을 설계·비교·분석하는 것입니다.

---

## 🗂 수정된 프로젝트 파일

```

📄 classifier\_baseline.py   → Full fine-tuning 기반 감정분석 모델 실행 파일 (SST, CFIMDB)
📄 classifier\_taskA.py      → ULMFiT 전략 적용 모델 실행 파일
📄 sonnet_generation_baseline.py   → fine-tuning 기반 기본 모델 실행 파일
📄 sonnet_generation_taskA.py      → Unlikelihood Loss 적용 모델 실행 파일
📄 sonnet_generation_taskB.py      → Prefix Tuning 적용 모델 실행 파일
📄 sonnet_generation_taskC.py      → Unlikelihood Loss + Prefix Tuning 적용 모델 실행 파일
📄 sonnet_eval.py                  → 생성된 소넷 평가 파일
📄 paraphrase_detection.py         → paraphrase_detection baseline 파일
📄 paraphrase_detection_taskA.py   → Last Hidden 및 Standard Dropout 기법 적용 파일
📄 paraphrase_detection_taskB.py   → Mean Pooling 및 Multi-Sample Dropout 기법 적용 파일
📄 paraphrase_detection_taskC.py   → Hybrid(Mean + Last Hidden) 및 Dynamic Dropout 기법 적용 파일
📄 predictions/para-dev-output.csv        → 베이스라인 모델의 Dev 셋 예측 결과
📄 predictions/para-dev-output_taskA.csv  → Task A 모델의 Dev 셋 예측 결과
📄 predictions/para-dev-output_taskB.csv  → Task B 모델의 Dev 셋 예측 결과
📄 predictions/para-dev-output_taskC.csv  → Task C 모델의 Dev 셋 예측 결과
📄 predictions/para-test-output.csv       → 베이스라인 모델의 Test 셋 예측 결과
📄 predictions/para-test-output_taskA.csv → Task A 모델의 Test 셋 예측 결과
📄 predictions/para-test-output_taskB.csv → Task B 모델의 Test 셋 예측 결과
📄 predictions/para-test-output_taskC.csv → Task C 모델의 Test 셋 예측 결과

````

---

## 🧪 실험 환경

- **프레임워크**: PyTorch, Huggingface Transformers
- **Python 버전**: 3.8
- **환경**:
  - 감정분석 : Google Colab (T4 GPU 사용)
  - paraphrase detection : Google Colab (L4 GPU 사용)
  - 시 생성 : NVIDIA TITAN RTX 1대  
- **훈련 시간**:
  - 감정분석 :
    - SST-5 (Baseline): 약 25분
    - SST-5 (ULMFiT): 약 20분
    - CFIMDB: 약 15분
  - paraphrase detection :
    - Baseline: Epoch당 약 46분 
    - Task A (Last Hidden 및 Standard Dropout): Epoch당 약 39분
    - Task B (Mean Pooling 및 Multi-Sample Dropout): Epoch당 약 39분
    - Task C (Hybrid(Mean + Last Hidden) 및 Dynamic Dropout ): Epoch당 약 39분
  - 시 생성 :
    - Baseline: Epoch당 약 1초
    - Task A (Unlikelihood Loss): Epoch당 약 7초
    - Task B (Prefix Tuning): Epoch당 약 1초
    - Task C (A+B 혼합): Epoch당 약 7초

---

## 🚀 실행 방법

### 1. 환경 세팅
```bash
# Conda 환경 생성
conda env create -f env.yml
conda activate gpt2-nlp
````

### 2. 감정분석 베이스라인 실행 (SST + CFIMDB)

```bash
python classifier_baseline.py --fine-tune-mode full-model --use_gpu
```

### 3. 감정분석 ULMFiT 전략 실험 (Task A)

```bash
python classifier_taskA.py --fine-tune-mode full-model --use_gpu
```

### 4. Paraphrase detection 베이스라인 실험 (Baseline)

```bash
python python paraphrase_detection.py --use_gpu
```

### 코랩용
```bash
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/Jihyeon0604/nlp_gpt2.git
%cd nlp_gpt2
!pip install -r requirements.txt
!pip install sacrebleu
!pip install huggingface_hub[hf_xet]
!python paraphrase_detection.py --use_gpu
# 경로 지정
!mkdir -p /content/drive/MyDrive/
# 저장
# 파일 복사하면서 이름 바꾸기
!cp predictions/para-dev-output.csv /content/drive/MyDrive/para-dev-output.csv.csv
!cp predictions/para-test-output.csv /content/drive/MyDrive/para-test-output.csv
!cp 10-1e-05-paraphrase_task.pt /content/drive/MyDrive/10-1e-05-paraphrase_task.pt
```

### 5. Paraphrase detection 실험 (Task A,B,C)

```bash
python python paraphrase_detection_taskA.py --use_gpu
python python paraphrase_detection_taskB.py --use_gpu
python python paraphrase_detection_taskC.py --use_gpu
```

### 6. 시 생성 베이스라인 실행

```bash
python sonnet_generation_baseline.py --use_gpu
```

### 7. 시 생성 전략 실험 (Task A, B, C)

```bash
python sonnet_generation_taskA.py --use_gpu
python sonnet_generation_taskB.py --use_gpu
python sonnet_generation_taskC.py --use_gpu
```

### 8. 시 생성 저장 경로 및 평가 방법
#### sonnet_generation_baseline.py 실행 시
가중치는 args.filepath (default = f'{args.epochs}-{args.lr}-sonnet.pt')에 저장\
생성된 소넷은 args.sonnet_out (default = predictions/generated_sonnets.txt)에 저장

#### sonnet_generation_taskA.py 실행 시
가중치는 f'{args.epochs}-{args.lr}-sonnet_A.pt'에 저장\
생성된 소넷은 predictions/generated_sonnets_A.txt에 저장

#### sonnet_generation_taskB.py 실행 시
가중치는 f'{args.epochs}-{args.lr}-sonnet_B.pt'에 저장\
생성된 소넷은 predictions/generated_sonnets_B.txt에 저장 

#### sonnet_generation_taskC.py 실행 시
가중치는 f'{args.epochs}-{args.lr}-sonnet_C.pt'에 저장\
생성된 소넷은 predictions/generated_sonnets_C.txt에 저장

이후 sonnet_eval.py로 실행 생성된 소넷 평가 가능
```bash
python sonnet_eval.py
```

sonnet_eval.py의 161번째 줄 filepath = 'predictions/generated_sonnets.txt'를 위의 args.sonnet_out에 맞게 수정한 후 sonnet_eval 실행
```bash
baseline 평가 시 filepath = 'predictions/generated_sonnets.txt'
taskA 평가 시 filepath = 'predictions/generated_sonnets_A.txt'
taskB 평가 시 filepath = 'predictions/generated_sonnets_B.txt'
taskC 평가 시 filepath = 'predictions/generated_sonnets_C.txt'
```

## 📊 주요 결과 요약

### 감정분석 성능지표
| 모델                 | 데이터셋   | Dev Accuracy | Dev F1 Score |
| ------------------ | ------ | ------------ | ------------ |
| Baseline (Full FT) | SST-5  | 51.8%        | 49.0%        |
| Task A (ULMFiT)    | SST-5  | 50.9%        | 48.0%        |
| Baseline (Full FT) | CFIMDB | 98.8%        | 98.8%        |

### paraphrase detection 성능지표
| 모델                      | 데이터셋  | Dev Accuracy | Dev F1 Score |
| ----------------------- | ----- | ------------ | ------------ |
| Baseline (Full FT)      | Quora | **89.9%**    | **89.2%**    |
| Task A (Last Hidden)    | Quora | 89.6%        | -            |
| Task B (Mean + Dropout) | Quora | 89.3%        | 88.7%        |
| Task C (Mixed Pooling)  | Quora | 89.63%       | 88.99%       |

### 시 생성 성능지표
| 모델   | Perplexity ↓ | Distinct-1 ↑ | Distinct-2 ↑ | Rhyming Accuracy ↑ |
| -------- | ------------ | ------------ | ------------ | ------------------ |
| Baseline | 55.73        | 0.621        | 0.927        | 0.141              |
| Task A    | 51.68        | 0.613        | 0.919        | **0.215**          |
| Task B    | **50.11**    | 0.572        | 0.878        | 0.166              |
| Task C    | 55.70        | **0.641**    | **0.944**    | 0.132              |

📌 분석 및 인사이트
**Task A (Unlikelihood Loss)**는 가장 균형 잡힌 성능을 보임:\
Perplexity 개선 (55.73 → 51.68)\
Rhyming Accuracy 향상 (0.141 → 0.215)\
Distinct 지표 소폭 하락했으나 여전히 우수한 다양성 유지

Task B (Prefix Tuning):\
가장 낮은 Perplexity 기록 (50.11)\
하지만 다양성(Distinct-1/2) 저하

Task C (Unlikelihood + Prefix):\
가장 높은 다양성(Distinct-1: 0.641, Distinct-2: 0.944)\
하지만 Rhyming Accuracy가 가장 낮음 (0.132)

---

## 📌 참고 사항

* 학습 중간 체크포인트 저장은 `.pt` 파일로 구현되어 있으며, 필요 시 `torch.load()`로 불러와 평가할 수 있습니다.
* 모든 실험 결과는 reproducibility를 위해 `random seed = 11711`로 고정하였습니다.
* 모델 구조는 `models/gpt2.py`와 `modules/attention.py` 등으로 모듈화되어 있어 확장 가능하도록 구성되어 있습니다.

---

## 👩‍💻 Contributors

* [Jihyeon0604](https://github.com/Jihyeon0604)
* [2rayjja](https://github.com/2rayija)
* [ha-hyeon](https://github.com/ha-hyeon)

---

