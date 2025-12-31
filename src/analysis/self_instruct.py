import pandas as pd
import json
import random
import os
from openai import OpenAI
from typing import List, Dict
import time
from ast import literal_eval

# OpenAI API 키 설정
client =  OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 설정
SEED_COUNT = 50  # 시드 데이터 개수
TARGET_COUNT = 500  # 생성할 총 데이터 개수
EVAL_CSV_PATH = "src/data/eval.csv"
OUTPUT_CSV_PATH = "output/self_instruct_output.csv"

def detect_domain(example: Dict) -> str:
    """문제의 도메인을 자동으로 감지"""
    text = f"{example.get('paragraph', '')} {example.get('question', '')} {example.get('question_plus', '')}"
    text = text.lower()
    
    # 한국사 관련 키워드
    korean_history_keywords = [
        '현종실록', '고려사', '삼국사기', '조선', '고려', '신라', '백제', '고구려',
        '임진왜란', '병자호란', '갑오개혁', '을사조약', '독립군', '광복', '일제',
        '붕당', '탕평', '과거', '전시과', '노비안검법', '전민변정도감', '12목',
        '만민공동회', '헌의 6조', '홍범 14조', '국채보상운동', '방곡령'
    ]
    
    # 서양사/철학 관련 키워드
    western_history_keywords = [
        '볼테르', '계몽주의', '베이컨', '코페르니쿠스', '에라스무스', '낭만주의',
        '리스본 지진', '프랑스', '영국', '독일', '러시아', '스페인', '이탈리아',
        '르네상스', '종교개혁', '프로테스탄트', '가톨릭', '낭트 칙령', '카를스바트',
        '농노제', '알렉산드르', '스톨리핀', '애디슨 법', '스펜서', '푸리에',
        '메테르니히', '마치니', '비오 9세'
    ]
    
    # 경제 관련 키워드
    economics_keywords = [
        '케인스', '통화', '인플레이션', 'gdp', '금리', '대출', '공급', '수요',
        '균형', '물가', '실업률', '소득', '세금', '지출', '생산량', '시장',
        '가격', '비용', '이윤', '효용', '한계 효용', '자유방임', '상업주의',
        '중금주의', '산업화', '모직물', '기계', '노동자', '임금'
    ]
    
    # 심리학 관련 키워드
    psychology_keywords = [
        '파블로프', '조건화', '심리학', '기억', '학습', '동기', '감정', '인지',
        '행동', '실험', '변수', '타당도', '신뢰도', '설득', '인식', '인지',
        '변연계', '시상', '연수', '소뇌', '단순노출효과', '체계적 둔감화'
    ]
    
    # 도메인 점수 계산
    scores = {
        '한국사': sum(1 for keyword in korean_history_keywords if keyword in text),
        '서양사/철학': sum(1 for keyword in western_history_keywords if keyword in text),
        '경제': sum(1 for keyword in economics_keywords if keyword in text),
        '심리학': sum(1 for keyword in psychology_keywords if keyword in text)
    }
    
    # 가장 높은 점수의 도메인 반환
    max_score = max(scores.values())
    if max_score > 0:
        domain = max(scores, key=scores.get)
        return domain
    
    # 키워드가 없으면 일반 독해로 분류
    return '일반 독해'

def load_seed_data(csv_path: str, n: int = 50) -> List[Dict]:
    """CSV에서 시드 데이터를 무작위로 n개 로드하고 도메인 정보 추가"""
    df = pd.read_csv(csv_path)
    
    # problems 컬럼을 파싱
    seed_data = []
    for _, row in df.iterrows():
        try:
            # problems가 JSON 문자열인 경우 파싱
            if isinstance(row['problems'], str):
                problems = literal_eval(row['problems'])
            else:
                problems = row['problems']
            
            example = {
                'id': row['id'],
                'paragraph': row.get('paragraph', ''),
                'question': problems.get('question', ''),
                'choices': problems.get('choices', []),
                'answer': problems.get('answer', None),
                'question_plus': row.get('question_plus', '') if 'question_plus' in row else ''
            }
            
            # 도메인 자동 감지
            example['domain'] = detect_domain(example)
            seed_data.append(example)
        except Exception as e:
            print(f"데이터 파싱 오류 (id: {row.get('id', 'unknown')}): {e}")
            continue
    
    # 도메인 분포 출력
    domain_counts = {}
    for example in seed_data:
        domain = example.get('domain', '알 수 없음')
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"\n📊 시드 데이터 도메인 분포:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {domain}: {count}개")
    
    # 무작위로 n개 선택
    if len(seed_data) < n:
        print(f"⚠️ 시드 데이터가 {n}개보다 적습니다. {len(seed_data)}개를 사용합니다.")
        return seed_data
    
    return random.sample(seed_data, n)

def format_example(example: Dict) -> str:
    """예시를 프롬프트 형식으로 변환"""
    formatted = f"지문: {example['paragraph']}\n"
    if example.get('question_plus'):
        formatted += f"보기: {example['question_plus']}\n"
    formatted += f"질문: {example['question']}\n"
    formatted += f"선택지: {', '.join(example['choices'])}\n"
    if example.get('answer') is not None:
        formatted += f"정답: {example['answer']}\n"
    return formatted

def create_prompt(seed_examples: List[Dict], num_examples: int = 3, target_domain: str = None) -> str:
    """Self-instruct를 위한 프롬프트 생성 (도메인 정보 포함)"""
    # 도메인별로 예시 선택
    if target_domain:
        # 특정 도메인의 예시 우선 선택
        domain_examples = [ex for ex in seed_examples if ex.get('domain') == target_domain]
        if len(domain_examples) >= num_examples:
            examples = random.sample(domain_examples, num_examples)
        else:
            # 해당 도메인 예시가 부족하면 다른 도메인도 포함
            examples = domain_examples.copy()
            remaining = num_examples - len(examples)
            other_examples = [ex for ex in seed_examples if ex.get('domain') != target_domain]
            if other_examples:
                examples.extend(random.sample(other_examples, min(remaining, len(other_examples))))
    else:
        # 도메인 다양성을 고려하여 선택
        # 각 도메인에서 최소 1개씩 선택 시도
        domain_groups = {}
        for ex in seed_examples:
            domain = ex.get('domain', '일반 독해')
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(ex)
        
        examples = []
        # 각 도메인에서 1개씩 선택
        for domain, domain_exs in domain_groups.items():
            if len(examples) < num_examples and domain_exs:
                examples.append(random.choice(domain_exs))
        
        # 부족하면 랜덤으로 추가
        remaining = num_examples - len(examples)
        if remaining > 0:
            available = [ex for ex in seed_examples if ex not in examples]
            if available:
                examples.extend(random.sample(available, min(remaining, len(available))))
        
        # 여전히 부족하면 중복 허용
        if len(examples) < num_examples:
            examples.extend(random.choices(seed_examples, k=num_examples - len(examples)))
    
    # 도메인 정보 수집
    domains_in_examples = [ex.get('domain', '일반 독해') for ex in examples]
    unique_domains = list(set(domains_in_examples))
    
    prompt = f"""다음은 한국어 독해 문제의 예시입니다. 이 예시들은 {', '.join(unique_domains)} 도메인에 속합니다.

예시:
"""
    
    for i, example in enumerate(examples, 1):
        domain = example.get('domain', '일반 독해')
        prompt += f"\n[예시 {i} - {domain}]\n"
        prompt += format_example(example)
        prompt += "\n"
    
    if target_domain:
        prompt += f"""
위 예시들을 참고하여, {target_domain} 도메인에 속하는 새로운 한국어 독해 문제를 생성해주세요. 
같은 도메인의 특성과 스타일을 유지하면서 새로운 문제를 만들어주세요.
"""
    else:
        prompt += """
위 예시들을 참고하여, 유사한 형식의 새로운 한국어 독해 문제를 생성해주세요.
예시 중 하나 이상과 같은 도메인(한국사, 서양사/철학, 경제, 심리학, 일반 독해 등)의 문제를 생성하는 것을 권장합니다.
"""
    
    prompt += """
다음 형식을 정확히 따라주세요:

{
  "paragraph": "지문 내용",
  "question_plus": "보기 내용 (있는 경우)",
  "question": "질문 내용",
  "choices": ["선택지1", "선택지2", "선택지3", "선택지4", "선택지5"],
  "answer": 정답 번호 (1-5)
}

JSON 형식으로만 응답해주세요. 다른 설명은 포함하지 마세요.
"""
    
    return prompt

def generate_instruction(prompt: str, max_retries: int = 3) -> Dict:
    """OpenAI API를 사용하여 새로운 instruction 생성"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 한국어 독해 문제를 생성하는 전문가입니다. 주어진 예시를 참고하여 유사한 형식의 새로운 문제를 생성합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # JSON 추출 시도
            # 코드 블록이 있는 경우 제거
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # JSON 파싱
            try:
                generated_data = json.loads(content)
                
                # 필수 필드 검증
                required_fields = ['paragraph', 'question', 'choices', 'answer']
                if all(field in generated_data for field in required_fields):
                    # choices가 리스트인지 확인
                    if isinstance(generated_data['choices'], list) and len(generated_data['choices']) >= 2:
                        # answer가 유효한 범위인지 확인
                        if isinstance(generated_data['answer'], int) and 1 <= generated_data['answer'] <= len(generated_data['choices']):
                            return generated_data
                        else:
                            print(f"⚠️ 정답 번호가 유효하지 않습니다: {generated_data.get('answer')}")
                    else:
                        print(f"⚠️ 선택지가 유효하지 않습니다: {generated_data.get('choices')}")
                else:
                    print(f"⚠️ 필수 필드가 누락되었습니다: {generated_data.keys()}")
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON 파싱 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                print(f"응답 내용: {content[:200]}...")
            
            # 재시도 전 대기
            if attempt < max_retries - 1:
                time.sleep(2)
                
        except Exception as e:
            print(f"⚠️ API 호출 오류 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return None

def save_generated_data(generated_data: List[Dict], output_path: str):
    """생성된 데이터를 CSV 형식으로 저장"""
    # 원본 형식에 맞게 변환
    formatted_data = []
    for idx, data in enumerate(generated_data):
        problems_dict = {
            'question': data['question'],
            'choices': data['choices'],
            'answer': data['answer']
        }
        
        if data.get('question_plus'):
            problems_dict['question_plus'] = data['question_plus']
        
        formatted_data.append({
            'id': f"self-instruct-{idx + 1:04d}",
            'paragraph': data.get('paragraph', ''),
            'problems': json.dumps(problems_dict, ensure_ascii=False),
            'question_plus': data.get('question_plus', '')
        })
    
    df = pd.DataFrame(formatted_data)
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # CSV 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 생성된 데이터가 저장되었습니다: {output_path}")
    print(f"총 {len(df)}개의 데이터가 생성되었습니다.")

def main():
    print("=" * 60)
    print("Stanford Alpaca 스타일 Self-Instruct 데이터 생성")
    print("=" * 60)
    
    # 1. 시드 데이터 로드
    print(f"\n[1단계] 시드 데이터 로드 중... ({SEED_COUNT}개)")
    seed_data = load_seed_data(EVAL_CSV_PATH, SEED_COUNT)
    print(f"✅ {len(seed_data)}개의 시드 데이터를 로드했습니다.")
    
    # 2. Self-instruct로 데이터 생성
    print(f"\n[2단계] Self-instruct로 데이터 생성 중... (목표: {TARGET_COUNT}개)")
    generated_data = []
    
    # 도메인별 생성 비율 설정 (시드 데이터의 도메인 분포에 따라)
    domain_counts = {}
    for example in seed_data:
        domain = example.get('domain', '일반 독해')
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    # 도메인별 목표 개수 계산 (비율 유지)
    total_seed = len(seed_data)
    domain_targets = {}
    for domain, count in domain_counts.items():
        domain_targets[domain] = int(TARGET_COUNT * (count / total_seed))
    
    # 나머지는 가장 많은 도메인에 할당
    remaining = TARGET_COUNT - sum(domain_targets.values())
    if remaining > 0:
        max_domain = max(domain_counts, key=domain_counts.get)
        domain_targets[max_domain] = domain_targets.get(max_domain, 0) + remaining
    
    print(f"\n📋 도메인별 생성 목표:")
    for domain, target in sorted(domain_targets.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {domain}: {target}개")
    
    # 진행 상황 표시를 위한 설정
    batch_size = 10
    total_batches = (TARGET_COUNT + batch_size - 1) // batch_size
    
    # 도메인별 생성 카운터
    domain_generated = {domain: 0 for domain in domain_targets.keys()}
    
    for batch in range(total_batches):
        current_batch_size = min(batch_size, TARGET_COUNT - len(generated_data))
        
        print(f"\n배치 {batch + 1}/{total_batches} 처리 중... (현재: {len(generated_data)}/{TARGET_COUNT})")
        
        for i in range(current_batch_size):
            # 도메인 선택 (목표에 맞춰)
            available_domains = [d for d, gen in domain_generated.items() 
                               if gen < domain_targets.get(d, 0)]
            
            if available_domains:
                # 목표에 미달한 도메인 중에서 선택
                target_domain = random.choice(available_domains)
            else:
                # 모든 도메인이 목표 달성했으면 랜덤 선택
                target_domain = random.choice(list(domain_targets.keys()))
            
            # 프롬프트 생성 (도메인 정보 포함)
            prompt = create_prompt(seed_data, num_examples=3, target_domain=target_domain)
            
            # 새로운 instruction 생성
            new_data = generate_instruction(prompt)
            
            if new_data:
                # 생성된 데이터에 도메인 정보 추가
                new_data['domain'] = target_domain
                generated_data.append(new_data)
                domain_generated[target_domain] = domain_generated.get(target_domain, 0) + 1
                print(f"  ✓ 데이터 {len(generated_data)}/{TARGET_COUNT} 생성 완료 ({target_domain})")
            else:
                print(f"  ✗ 데이터 생성 실패 (재시도 중...)")
                # 실패한 경우 한 번 더 시도
                time.sleep(1)
                new_data = generate_instruction(prompt)
                if new_data:
                    new_data['domain'] = target_domain
                    generated_data.append(new_data)
                    domain_generated[target_domain] = domain_generated.get(target_domain, 0) + 1
                    print(f"  ✓ 데이터 {len(generated_data)}/{TARGET_COUNT} 생성 완료 ({target_domain}, 재시도 성공)")
            
            # API rate limit 방지를 위한 대기
            time.sleep(0.5)
        
        # 중간 저장 (배치마다)
        if len(generated_data) > 0:
            save_generated_data(generated_data, OUTPUT_CSV_PATH)
    
    # 3. 최종 저장
    print(f"\n[3단계] 최종 데이터 저장 중...")
    save_generated_data(generated_data, OUTPUT_CSV_PATH)
    
    # 최종 도메인 분포 출력
    final_domain_counts = {}
    for data in generated_data:
        domain = data.get('domain', '알 수 없음')
        final_domain_counts[domain] = final_domain_counts.get(domain, 0) + 1
    
    print("\n" + "=" * 60)
    print("✅ Self-instruct 데이터 생성 완료!")
    print(f"생성된 데이터: {len(generated_data)}개")
    print(f"\n📊 최종 도메인 분포:")
    for domain, count in sorted(final_domain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {domain}: {count}개")
    print(f"\n저장 위치: {OUTPUT_CSV_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()

