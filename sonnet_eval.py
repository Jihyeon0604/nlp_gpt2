import re
import math
from collections import Counter, defaultdict

def parse_sonnets(text):
    """소네트를 숫자 기준으로 분리"""
    sonnets = {}
    lines = text.strip().split('\n')
    
    current_sonnet = None
    current_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 숫자로 시작하는 라인 체크 (소네트 번호)
        if re.match(r'^\d+$', line):
            # 이전 소네트 저장
            if current_sonnet is not None and current_lines:
                sonnets[current_sonnet] = '\n'.join(current_lines)
            
            current_sonnet = int(line)
            current_lines = []
        elif current_sonnet is not None:
            # <|endoftext|> 제거 및 불완전한 라인 필터링
            if '<|endoftext|>' in line:
                line = line.split('<|endoftext|>')[0].strip()
            if line and not line.startswith('--'):
                current_lines.append(line)
    
    # 마지막 소네트 저장
    if current_sonnet is not None and current_lines:
        sonnets[current_sonnet] = '\n'.join(current_lines)
    
    return sonnets

def calculate_perplexity(text):
    """간단한 perplexity 계산 (단어 빈도 기반)"""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return float('inf')
    
    word_counts = Counter(words)
    total_words = len(words)
    
    # 각 단어의 확률 계산
    log_prob_sum = 0
    for word in words:
        prob = word_counts[word] / total_words
        log_prob_sum += math.log(prob)
    
    # Perplexity = exp(-1/N * sum(log(p(w))))
    avg_log_prob = log_prob_sum / total_words
    perplexity = math.exp(-avg_log_prob)
    
    return perplexity

def calculate_distinct_n(text, n=2):
    """Distinct-n 계산"""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < n:
        return 0.0
    
    # n-gram 생성
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams.append(ngram)
    
    if not ngrams:
        return 0.0
    
    # distinct ratio 계산
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    
    return unique_ngrams / total_ngrams

def get_rhyme_ending(word):
    """단어의 라임 부분 추출 (간단한 음성학적 근사)"""
    word = word.lower().strip('.,!?;:"')
    
    # 기본적인 영어 라임 패턴
    vowels = 'aeiou'
    
    # 단어 끝에서 라임 부분 찾기
    rhyme_part = ''
    for i in range(len(word)-1, -1, -1):
        rhyme_part = word[i] + rhyme_part
        if word[i] in vowels:
            break
    
    return rhyme_part if rhyme_part else word[-2:] if len(word) >= 2 else word

def calculate_rhyming_accuracy(text):
    """라임 정확도 계산"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if len(lines) < 4:
        return 0.0
    
    # 각 라인의 마지막 단어 추출
    last_words = []
    for line in lines:
        words = re.findall(r'\b\w+\b', line)
        if words:
            last_words.append(words[-1])
    
    if len(last_words) < 4:
        return 0.0
    
    # 라임 패턴 확인 (ABAB, ABBA 등을 고려)
    rhyme_endings = [get_rhyme_ending(word) for word in last_words]
    
    # 간단한 라임 매칭 확인
    rhyme_matches = 0
    total_pairs = 0
    
    # 연속된 라인들 간의 라임 체크
    for i in range(0, len(rhyme_endings)-1, 2):
        if i+1 < len(rhyme_endings):
            if rhyme_endings[i] == rhyme_endings[i+1]:
                rhyme_matches += 1
            total_pairs += 1
    
    # 교차 라임 체크 (ABAB 패턴)
    for i in range(0, len(rhyme_endings)-3, 4):
        if i+3 < len(rhyme_endings):
            if rhyme_endings[i] == rhyme_endings[i+2]:
                rhyme_matches += 1
            if rhyme_endings[i+1] == rhyme_endings[i+3]:
                rhyme_matches += 1
            total_pairs += 2
    
    return rhyme_matches / total_pairs if total_pairs > 0 else 0.0

def evaluate_sonnets(text):
    """모든 소네트 평가"""
    sonnets = parse_sonnets(text)
    
    results = {}
    
    for sonnet_id, sonnet_text in sorted(sonnets.items()):
        perplexity = calculate_perplexity(sonnet_text)
        distinct_1 = calculate_distinct_n(sonnet_text, n=1)
        distinct_2 = calculate_distinct_n(sonnet_text, n=2)
        rhyming_acc = calculate_rhyming_accuracy(sonnet_text)
        
        results[sonnet_id] = {
            'perplexity': perplexity,
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'rhyming_accuracy': rhyming_acc,
            'text': sonnet_text
        }
    
    return results

filepath = 'predictions/generated_sonnets.txt'

# 텍스트 파일 읽기 및 평가
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

results = evaluate_sonnets(text)

# 결과 출력
print("Sonnet Evaluation Results")
print("=" * 50)

for sonnet_id in sorted(results.keys()):
    metrics = results[sonnet_id]
    print(f"\nSonnet {sonnet_id}:")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
    print(f"  Distinct-1: {metrics['distinct_1']:.3f}")
    print(f"  Distinct-2: {metrics['distinct_2']:.3f}")
    print(f"  Rhyming Accuracy: {metrics['rhyming_accuracy']:.3f}")

# 전체 평균 계산
print("\n" + "=" * 50)
print("Average Metrics:")
avg_perplexity = sum(r['perplexity'] for r in results.values()) / len(results)
avg_distinct_1 = sum(r['distinct_1'] for r in results.values()) / len(results)
avg_distinct_2 = sum(r['distinct_2'] for r in results.values()) / len(results)
avg_rhyming = sum(r['rhyming_accuracy'] for r in results.values()) / len(results)

print(f"  Average Perplexity: {avg_perplexity:.2f}")
print(f"  Average Distinct-1: {avg_distinct_1:.3f}")
print(f"  Average Distinct-2: {avg_distinct_2:.3f}")
print(f"  Average Rhyming Accuracy: {avg_rhyming:.3f}")