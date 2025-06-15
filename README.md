# 🧠 GPT-2 기반 자연어처리 태스크 실험 프로젝트

본 프로젝트는 OpenAI의 GPT-2 구조를 기반으로 직접 모델을 구현하고, 다양한 Fine-Tuning 전략을 적용하여 **감정 분석**, **패러프레이즈 탐지**, **시 생성** 세 가지 자연어처리(NLP) 태스크에 적용한 실험을 수행하였습니다.

주요 목표는 GPT-2 decoder-only 구조의 특성을 이해하고, 이를 다양한 downstream task에 효과적으로 활용하기 위한 fine-tuning 전략을 설계·비교·분석하는 것입니다.

---

## 🗂 수정된 프로젝트 파일

```

📄 classifier\_baseline.py   → Full fine-tuning 기반 감정분석 모델 실행 파일 (SST, CFIMDB)
📄 classifier\_taskA.py      → ULMFiT 전략 적용 모델 실행 파일


````

---

## 🧪 실험 환경

- **프레임워크**: PyTorch, Huggingface Transformers
- **Python 버전**: 3.8
- **환경**:
  - 감정분석 : Google Colab (T4 GPU 사용)
- **훈련 시간**:
  - 감정분석 :
    - SST-5 (Baseline): 약 25분
    - SST-5 (ULMFiT): 약 20분
    - CFIMDB: 약 15분

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


## 📊 주요 결과 요약

| 모델                 | 데이터셋   | Dev Accuracy | Dev F1 Score |
| ------------------ | ------ | ------------ | ------------ |
| Baseline (Full FT) | SST-5  | 51.8%        | 49.0%        |
| Task A (ULMFiT)    | SST-5  | 50.9%        | 48.0%        |
| Baseline (Full FT) | CFIMDB | 98.8%        | 98.8%        |

---

## 📌 참고 사항

* 학습 중간 체크포인트 저장은 `.pt` 파일로 구현되어 있으며, 필요 시 `torch.load()`로 불러와 평가할 수 있습니다.
* 모든 실험 결과는 reproducibility를 위해 `random seed = 11711`로 고정하였습니다.
* 모델 구조는 `models/gpt2.py`와 `modules/attention.py` 등으로 모듈화되어 있어 확장 가능하도록 구성되어 있습니다.

---

## 👩‍💻 Contributors

* [Jihyeon0604](https://github.com/Jihyeon0604)
* [2rayjja](https://github.com/2rayjja)
* [ha-hyeon](https://github.com/ha-hyeon)

---

