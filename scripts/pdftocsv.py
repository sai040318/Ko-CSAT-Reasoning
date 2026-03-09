"""
수능 문제지 PDF → CSV 변환 스크립트
목표 형식: id, paragraph, problems (question/choices/answer), question_plus
"""
import re
import os
import pdfplumber
import pandas as pd

# 정답 원문자 → 숫자 문자열
ANSWER_MAP = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'}
CIRCLE_NUMS = ['①', '②', '③', '④', '⑤']

# 이미지 PDF 정답표 (하드코딩)
HARDCODED_ANSWERS = {
    ('국어', 2025): {
        1:'3', 2:'4', 3:'5', 4:'4', 5:'5', 6:'3', 7:'2', 8:'1', 9:'2', 10:'3',
        11:'1', 12:'5', 13:'3', 14:'1', 15:'2', 16:'2', 17:'3',
        18:'2', 19:'4', 20:'1', 21:'4', 22:'4', 23:'5', 24:'2', 25:'2',
        26:'1', 27:'1', 28:'4', 29:'3', 30:'5', 31:'4', 32:'3', 33:'5', 34:'2',
    },
    ('한국사', 2025): {
        1:'5', 2:'1', 3:'3', 4:'3', 5:'1',
        6:'4', 7:'5', 8:'2', 9:'5', 10:'3',
        11:'5', 12:'4', 13:'1', 14:'3', 15:'2',
        16:'2', 17:'4', 18:'3', 19:'2', 20:'1',
    },
}

# --------------------------------------------------------------------------
# 파일 쌍 정의
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#FILE_PAIRS = 

    #[
    # (문제지, 정답표, 과목, 연도)
    #('국어영역_홀수형_2022.pdf',          '1교시_국어영역_정답표_2022.pdf',    '국어', 2022),
    #('1교시_국어영역_문제지_2023.pdf',    '1교시_국어영역_정답표_2023.pdf',    '국어', 2023),
    #('국어영역_문제지_2024.pdf',          '국어영역_정답표_2024.pdf',          '국어', 2024),
    #('국어영역_문제지_홀수형_2025.pdf',   '국어영역_정답표_2025.pdf',          '국어', 2025),
    #('국어영역_문제지_2026.pdf',          '국어영역_정답표_2026.pdf',          '국어', 2026)]

FILE_PAIRS = [('한국사영역_문제지_2022.pdf',        '4교시_한국사영역_정답표_2022.pdf',  '한국사', 2022),
    ('4교시_한국사영역_문제지_2023.pdf',  '4교시_한국사영역_정답표_2023.pdf',  '한국사', 2023),
    ('한국사영역_문제지_2024.pdf',        '한국사영역_정답표_2024.pdf',        '한국사', 2024),
    ('한국사영역_문제지_홀수형_2025.pdf', '한국사영역_정답표_2025.pdf',        '한국사', 2025),
    ('한국사영역_문제지_2026.pdf',        '한국사영역_정답표_2026.pdf',        '한국사', 2026),
]

# 페이지 추출 시 제거할 아티팩트 (페이지 번호, 형식 표기, 저작권 등)
PAGE_ARTIFACT_PATTERN = re.compile(
    r'^\s*\d{1,2}\s*$'           # 단독 페이지 번호 (2, 3, 20 등)
    r'|홀수형|짝수형'              # 형 표기
    r'|이\s*문제지에\s*관한\s*저작권[^\n]*'  # 저작권 문구
    r'|한국교육과정평가원[^\n]*'
    r'|\d{4}학년도\s*대학수학능력시험\s*문제지'  # 시험 표제
    r'|제\s*\d+\s*교시'
    r'|수학능력시험\s*문제지',
    re.MULTILINE
)

# --------------------------------------------------------------------------
# 유틸리티
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """(cid:N) 등 인코딩 아티팩트 제거 및 공백 정리"""
    if not text:
        return ''
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # 앞쪽 구두점 제거
    text = re.sub(r'^[\s\.\。,]+', '', text)
    return text.strip()


def extract_page_text(page) -> str:
    """2단 컬럼 페이지에서 왼쪽 → 오른쪽 순서로 텍스트 추출.
    페이지 번호·형식 표기·저작권 등 아티팩트를 미리 제거."""
    w = page.width
    h = page.height
    left = page.crop((0, 0, w / 2, h)).extract_text() or ''
    right = page.crop((w / 2, 0, w, h)).extract_text() or ''
    left = PAGE_ARTIFACT_PATTERN.sub('', left)
    right = PAGE_ARTIFACT_PATTERN.sub('', right)
    return left + '\n' + right


def extract_full_text(pdf_path: str, stop_at_jaksu: bool = False) -> str:
    """
    PDF 전체 텍스트 추출 (2단 컬럼 처리).
    stop_at_jaksu=True: 짝수형 섹션 시작 전까지만 추출
    """
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = extract_page_text(page)
            if stop_at_jaksu:
                # 짝수형 헤더 감지
                if re.search(r'짝수.*형|짝수\s*\)', page_text):
                    break
            parts.append(page_text)
    return '\n'.join(parts)

# --------------------------------------------------------------------------
# 정답표 파싱
# --------------------------------------------------------------------------
def parse_answer_sheet(pdf_path: str, subject: str, year: int) -> dict:
    """정답표 PDF → {문항번호(int): 정답(str '1'~'5')}"""
    # 이미지 PDF는 하드코딩 값 사용
    key = (subject, year)
    if key in HARDCODED_ANSWERS:
        return HARDCODED_ANSWERS[key]

    answers = {}
    with pdfplumber.open(pdf_path) as pdf:
        # 첫 번째 페이지가 홀수형
        page = pdf.pages[0]
        text = page.extract_text() or ''
        pattern = r'(\d{1,2})\s+([①②③④⑤])\s+\d'
        for m in re.finditer(pattern, text):
            q_num = int(m.group(1))
            ans = ANSWER_MAP[m.group(2)]
            if q_num not in answers:
                answers[q_num] = ans
    return answers

# --------------------------------------------------------------------------
# 문제지 파싱 공통 유틸
# --------------------------------------------------------------------------
def split_into_questions(full_text: str) -> list:
    """
    전체 텍스트를 문제 단위로 분리.
    Returns list of (q_num: int, q_text: str)
    """
    # 숫자. 이후 숫자나 개행이 아닌 문자가 오면 문제 번호로 판단
    question_pattern = re.compile(
        r'(?:^|\n)\s*(\d{1,2})\s*\.\s+(?=[^\d\n])',
        re.MULTILINE
    )
    matches = list(question_pattern.finditer(full_text))
    if not matches:
        return []

    results = []
    for i, m in enumerate(matches):
        q_num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        q_text = full_text[start:end].strip()
        results.append((q_num, q_text))

    # 번호가 역순이면 앞 문제 텍스트에 합침 (지문 내 번호 목록 오감지 방지)
    merged = []
    for q_num, q_text in results:
        if merged and q_num <= merged[-1][0]:
            prev_num, prev_text = merged[-1]
            merged[-1] = (prev_num, prev_text + '\n' + q_text)
        else:
            merged.append((q_num, q_text))

    return merged


def parse_question_block(q_text: str) -> dict:
    """
    단일 문제 텍스트 파싱.
    Returns {'question': str, 'choices': [str, ...], 'question_plus': str}
    """
    # <보기> 독립 블록 추출 → question_plus (인라인 참조 "<보기>를" 제외)
    bogi_pattern = re.compile(r'\n\s*<보\s*기\s*>\s*\n(.*?)(?=\s*①|\s*$)', re.DOTALL)
    question_plus = ''
    bogi_match = bogi_pattern.search(q_text)
    if bogi_match:
        question_plus = clean_text(bogi_match.group(1))
        q_text_main = q_text[:bogi_match.start()] + q_text[bogi_match.end():]
    else:
        q_text_main = q_text

    # ① 위치 찾기
    choice_start = None
    for c in CIRCLE_NUMS:
        idx = q_text_main.find(c)
        if idx != -1:
            if choice_start is None or idx < choice_start:
                choice_start = idx

    choices = []
    if choice_start is not None:
        question_part = q_text_main[:choice_start].strip()
        choices_part = q_text_main[choice_start:]

        # 선택지 분리: ①~⑤ 기준
        choice_pattern = re.compile(r'([①②③④⑤])\s*(.+?)(?=[①②③④⑤]|$)', re.DOTALL)
        for m in choice_pattern.finditer(choices_part):
            raw = m.group(1) + ' ' + m.group(2)
            choice_text = clean_text(raw)
            if choice_text:
                choices.append(choice_text)
    else:
        question_part = q_text_main.strip()

    question = re.sub(r'^\d{1,2}\s*\.\s*', '', question_part).strip()
    question = clean_text(question)

    return {
        'question': question,
        'choices': choices,
        'question_plus': question_plus,
    }

# --------------------------------------------------------------------------
# 국어영역 문제지 파싱
# --------------------------------------------------------------------------
def parse_koreo_passage_header(text: str):
    """[N~M] 다음 글을 읽고 물음에 답하시오. 패턴 찾기"""
    pattern = re.compile(
        r'\[(\d{1,2})[～~](\d{1,2})\]\s*다음\s*(?:글을|을)\s*읽고\s*물음에\s*답하시오'
    )
    results = []
    for m in pattern.finditer(text):
        q_from = int(m.group(1))
        q_to = int(m.group(2))
        results.append((m.start(), q_from, q_to))
    return results


def parse_koreo_exam(pdf_path: str, answers: dict) -> list:
    """국어영역 문제지 PDF 파싱. Returns list of row dicts."""
    full_text = extract_full_text(pdf_path)
    headers = parse_koreo_passage_header(full_text)
    rows = []

    seen_ranges = set()  # 이미 처리한 (q_from, q_to) 쌍 추적

    for h_idx, (h_start, q_from, q_to) in enumerate(headers):
        # 공통 과목(1~34)만 처리, 중복 헤더 스킵
        if q_from > 34:
            continue
        if (q_from, q_to) in seen_ranges:
            continue
        seen_ranges.add((q_from, q_to))

        # 헤더 끝 위치
        header_end_match = re.search(r'답하시오', full_text[h_start:])
        if not header_end_match:
            continue
        header_end = h_start + header_end_match.end()

        # 다음 헤더까지의 세그먼트
        if h_idx + 1 < len(headers):
            next_start = headers[h_idx + 1][0]
            # 단, 다음 헤더가 선택과목(35~)이면 그냥 거기까지
            segment = full_text[header_end:next_start]
        else:
            segment = full_text[header_end:]

        # 세그먼트에서 지문과 문제 분리
        # q_from번 문제 패턴이 나오는 위치
        q_pattern = re.compile(rf'(?:^|\n)\s*{q_from}\s*\.\s+', re.MULTILINE)
        q_match = q_pattern.search(segment)

        if q_match:
            passage_raw = segment[:q_match.start()]
            questions_text = segment[q_match.start():]
        else:
            # q_from 문제를 못 찾으면 세그먼트 반으로 나눔
            passage_raw = segment[:len(segment) // 2]
            questions_text = segment

        passage_text = clean_text(passage_raw)
        if len(passage_text) < 20:
            continue

        # 문제 파싱 (q_from ~ min(q_to, 34))
        q_blocks = split_into_questions(questions_text)
        max_q = min(q_to, 34)

        for q_num, q_text in q_blocks:
            if not (q_from <= q_num <= max_q):
                continue
            parsed = parse_question_block(q_text)
            if not parsed['choices']:
                continue

            answer = answers.get(q_num, '')
            problems = {
                'question': parsed['question'],
                'choices': parsed['choices'],
                'answer': answer,
            }
            rows.append({
                'paragraph': passage_text,
                'problems': str(problems),
                'question_plus': parsed['question_plus'],
            })

    return rows

# --------------------------------------------------------------------------
# 한국사영역 문제지 파싱
# --------------------------------------------------------------------------
def parse_history_exam(pdf_path: str, answers: dict) -> list:
    """한국사영역 문제지 PDF 파싱. Returns list of row dicts."""
    # 홀수형만 추출 (짝수형 섹션 이전까지)
    full_text = extract_full_text(pdf_path, stop_at_jaksu=True)
    q_blocks = split_into_questions(full_text)

    rows = []
    seen_nums = set()

    for q_num, q_text in q_blocks:
        if not (1 <= q_num <= 20):
            continue
        if q_num in seen_nums:
            continue  # 중복 제거
        seen_nums.add(q_num)

        parsed = parse_question_block(q_text)
        if not parsed['choices']:
            continue

        answer = answers.get(q_num, '')
        paragraph = extract_history_passage(q_text, parsed['question'])

        problems = {
            'question': parsed['question'],
            'choices': parsed['choices'],
            'answer': answer,
        }
        rows.append({
            'paragraph': paragraph,
            'problems': str(problems),
            'question_plus': parsed['question_plus'],
        })

    return rows


def extract_history_passage(q_text: str, question: str) -> str:
    """한국사 문제에서 지문(인용구) 추출."""
    # 문제 번호 제거
    text = re.sub(r'^\d{1,2}\s*\.\s*', '', q_text.strip())

    # 질문 시작 패턴 (한국사에서 자주 쓰이는 표현)
    q_patterns = [
        r'에\s*대한\s*설명으로',
        r'에\s*들어갈\s*내용으로',
        r'에\s*해당하는\s*것은',
        r'시기\s*사이에\s*있었던',
        r'으로\s*가장\s*적절한',
        r'으로\s*옳은\s*것은',
        r'옳은\s*것은',
        r'옳지\s*않은\s*것은',
        r'의\s*활동으로',
        r'밑줄\s*친',
        r'탐구\s*활동으로',
        r'배경으로\s*가장',
    ]

    # 질문 패턴 최초 등장 이전의 마지막 문장을 지문으로
    earliest = len(text)
    for pat in q_patterns:
        m = re.search(pat, text)
        if m and m.start() < earliest:
            earliest = m.start()

    if earliest < len(text):
        # 질문 시작 직전 마침표까지
        q_start = text.rfind('.', 0, earliest)
        if q_start > 5:
            passage = clean_text(text[:q_start + 1])
            if len(passage) > 5:
                return passage

    return clean_text(question)

# --------------------------------------------------------------------------
# 메인
# --------------------------------------------------------------------------
def main():
    all_rows = []

    for exam_file, answer_file, subject, year in FILE_PAIRS:
        exam_path = os.path.join(BASE_DIR, exam_file)
        answer_path = os.path.join(BASE_DIR, answer_file)

        if not os.path.exists(exam_path):
            print(f"[SKIP] 파일 없음: {exam_file}")
            continue
        if not os.path.exists(answer_path):
            print(f"[SKIP] 파일 없음: {answer_file}")
            continue

        print(f"[처리중] {exam_file} ({subject} {year}년)")

        try:
            answers = parse_answer_sheet(answer_path, subject, year)
            print(f"  정답 로드: {len(answers)}개")

            if subject == '국어':
                rows = parse_koreo_exam(exam_path, answers)
            else:
                rows = parse_history_exam(exam_path, answers)

            print(f"  추출된 문제: {len(rows)}개")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    if not all_rows:
        print("추출된 데이터 없음.")
        return

    df = pd.DataFrame(all_rows)
    df.insert(0, 'id', [f'generation-for-nlp-{i}' for i in range(len(df))])
    df = df[['id', 'paragraph', 'problems', 'question_plus']]

    output_path = os.path.join(BASE_DIR, 'output_kr_2022_2026.csv')
    df.to_csv(output_path, encoding='utf-8-sig', index=True)
    print(f"\n완료! {len(df)}개 행 → {output_path}")


if __name__ == '__main__':
    main()
