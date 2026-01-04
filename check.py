import pandas as pd
import sys
from collections import Counter

def check_answer_distribution(csv_file):
    """
    CSV 파일의 answer 컬럼에서 1,2,3,4,5의 분포도를 확인합니다.
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_file)
        
        # answer 컬럼이 있는지 확인
        if 'answer' not in df.columns:
            print(f"❌ Error: 'answer' 컬럼을 찾을 수 없습니다.")
            print(f"사용 가능한 컬럼: {', '.join(df.columns)}")
            return
        
        # answer 컬럼의 분포 계산
        answer_counts = df['answer'].value_counts().sort_index()
        total_count = len(df)
        
        print(f"\n{'='*60}")
        print(f"📊 파일: {csv_file}")
        print(f"{'='*60}")
        print(f"\n총 데이터 개수: {total_count}\n")
        
        # 1,2,3,4,5 각각의 분포 출력
        print(f"{'predict':<10} {'개수':<10} {'비율':<10} {'시각화'}")
        print(f"{'-'*60}")
        
        for i in range(1, 6):
            count = answer_counts.get(i, 0)
            percentage = (count / total_count * 100) if total_count > 0 else 0
            bar = '█' * int(percentage / 2)  # 비율에 따른 막대 그래프
            print(f"{i:<10} {count:<10} {percentage:>6.2f}%   {bar}")
        
        # 기타 값이 있는지 확인
        other_values = df[~df['answer'].isin([1, 2, 3, 4, 5])]['answer'].value_counts()
        if len(other_values) > 0:
            print(f"\n⚠️  1-5 범위 외의 값:")
            for value, count in other_values.items():
                percentage = (count / total_count * 100)
                print(f"   {value}: {count}개 ({percentage:.2f}%)")
        
        # NaN 값 확인
        nan_count = df['answer'].isna().sum()
        if nan_count > 0:
            percentage = (nan_count / total_count * 100)
            print(f"\n⚠️  결측치(NaN): {nan_count}개 ({percentage:.2f}%)")
        
        print(f"\n{'='*60}\n")
        
    except FileNotFoundError:
        print(f"❌ Error: 파일을 찾을 수 없습니다: {csv_file}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # 명령줄 인자로 파일 경로를 받거나, 기본 파일들을 확인
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        check_answer_distribution(csv_file)
    else:
        # 기본적으로 확인할 파일들
        default_files = [
            "output/qwen2.5_shuffle_self_eval_results.csv",
            "output/qwen2.5_32b_base_output.csv",
        ]
        
        print("📁 CSV 파일 경로를 인자로 제공하지 않았습니다.")
        print("기본 파일들을 확인합니다...\n")
        
        for file in default_files:
            try:
                check_answer_distribution(file)
            except:
                pass
        
        print("\n💡 사용법: python check.py <csv_file_path>")
        print("   예시: python check.py output/qwen2.5_shuffle_self_eval_results.csv")

