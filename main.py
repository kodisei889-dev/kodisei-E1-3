"""Mini NPU Simulator
MAC(Multiply-Accumulate) 연산으로 3x3 ~ 25x25 패턴(Cross / X)을 판별한다.
표준 라이브러리(json, re, time)만 사용한다.
"""
import json
import re
import time

EPSILON = 1e-9
REPEAT = 10
DATA_FILE = "data.json"
LABEL_CROSS = "Cross"
LABEL_X = "X"
UNDECIDED = "UNDECIDED"


# ---------------------------------------------------------------------------
# 핵심 연산
# ---------------------------------------------------------------------------
def mac(pattern, flt):
    """패턴과 필터를 같은 위치끼리 곱해 모두 더한다. (반복문 직접 구현)"""
    n = len(pattern)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += pattern[i][j] * flt[i][j]
    return total


def compare_scores(score_a, score_b, eps=EPSILON):
    """허용오차 기반 비교. 'A' / 'B' / None(동점)을 반환한다."""
    if abs(score_a - score_b) < eps:
        return None
    return "A" if score_a > score_b else "B"


def normalize_label(raw):
    """다양한 라벨 표기를 표준 라벨(Cross / X)로 정규화한다. 실패하면 None."""
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    if key in ("+", "cross"):
        return LABEL_CROSS
    if key == "x":
        return LABEL_X
    return None


def judge(cross_score, x_score):
    """Cross / X / UNDECIDED 판정."""
    winner = compare_scores(cross_score, x_score)
    if winner is None:
        return UNDECIDED
    return LABEL_CROSS if winner == "A" else LABEL_X


def validate_matrix(m, name="matrix"):
    """n x n 숫자 2차원 배열인지 검사. 문제가 있으면 ValueError."""
    if not isinstance(m, list) or len(m) == 0:
        raise ValueError(f"{name}: 2차원 배열이 아닙니다")
    n = len(m)
    for r, row in enumerate(m):
        if not isinstance(row, list):
            raise ValueError(f"{name}: {r + 1}행이 배열이 아닙니다")
        if len(row) != n:
            raise ValueError(f"{name}: 정사각형이 아닙니다 ({n}행, {r + 1}행의 열 수 {len(row)})")
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{name}: 숫자가 아닌 값이 있습니다 ({v!r})")
    return n


# ---------------------------------------------------------------------------
# 성능 측정
# ---------------------------------------------------------------------------
def measure_mac_ms(pattern, flt, repeat=REPEAT):
    """MAC 연산 호출 구간만 repeat회 측정해 평균(ms)을 반환한다."""
    mac(pattern, flt)  # 워밍업 (측정에서 제외)
    elapsed = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        mac(pattern, flt)
        elapsed += time.perf_counter() - start
    return elapsed / repeat * 1000.0


def make_bench_matrix(n):
    """성능 측정용 입력 (체크무늬 패턴)."""
    return [[float((i + j) % 2) for j in range(n)] for i in range(n)]


def run_performance(sizes=(3, 5, 13, 25)):
    print("\n#---------------------------------------")
    print("# 성능 분석 (평균/%d회)" % REPEAT)
    print("#---------------------------------------")
    print(f"{'크기':<10}{'평균 시간(ms)':>16}{'연산 횟수(N²)':>16}")
    print("-" * 42)
    for n in sizes:
        m = make_bench_matrix(n)
        avg = measure_mac_ms(m, m)
        print(f"{n}×{n:<7}{avg:>16.4f}{n * n:>16}")


# ---------------------------------------------------------------------------
# 모드 1: 사용자 입력 (3x3)
# ---------------------------------------------------------------------------
def read_matrix_3x3(title):
    """3줄을 공백 구분으로 입력받는다. 형식 오류 시 처음부터 재입력."""
    while True:
        print(f"{title} (3줄 입력, 공백 구분)")
        rows = []
        error = False
        for _ in range(3):
            line = input().strip()
            parts = line.split()
            if len(parts) != 3:
                error = True
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                error = True
        if error or len(rows) != 3:
            print("입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.")
            continue
        return rows


def print_matrix(m):
    for row in m:
        print(" ".join(f"{v:g}" for v in row))


def run_user_mode():
    print("\n#----------------------------------------")
    print("# [1] 필터 입력")
    print("#----------------------------------------")
    filter_a = read_matrix_3x3("필터 A")
    print()
    filter_b = read_matrix_3x3("필터 B")
    print("\n✓ 필터 A, B 저장 완료")
    print("[필터 A]")
    print_matrix(filter_a)
    print("[필터 B]")
    print_matrix(filter_b)

    print("\n#----------------------------------------")
    print("# [2] 패턴 입력")
    print("#----------------------------------------")
    pattern = read_matrix_3x3("패턴")

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    avg_ms = (measure_mac_ms(pattern, filter_a) + measure_mac_ms(pattern, filter_b)) / 2

    print("\n#----------------------------------------")
    print("# [3] MAC 결과")
    print("#----------------------------------------")
    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")
    print(f"연산 시간(평균/{REPEAT}회): {avg_ms:.4f} ms")
    winner = compare_scores(score_a, score_b)
    if winner is None:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON:g})")
    else:
        print(f"판정: {winner}")

    run_performance()


# ---------------------------------------------------------------------------
# 모드 2: data.json 분석
# ---------------------------------------------------------------------------
def load_filters(data):
    """filters 섹션을 읽어 {N: {'Cross': 행렬, 'X': 행렬}} 로 정규화한다."""
    result = {}
    filters = data.get("filters")
    if not isinstance(filters, dict):
        print("✗ 'filters' 섹션이 없거나 형식이 올바르지 않습니다.")
        return result
    for key, group in filters.items():
        m = re.fullmatch(r"size_(\d+)", str(key))
        if not m:
            print(f"✗ {key} 필터 키 형식 오류 (size_N 형태가 아님)")
            continue
        n = int(m.group(1))
        if not isinstance(group, dict):
            print(f"✗ {key} 필터 형식 오류")
            continue
        normalized = {}
        for raw_label, matrix in group.items():
            label = normalize_label(raw_label)
            if label is None:
                print(f"✗ {key}: 알 수 없는 필터 라벨 '{raw_label}'")
                continue
            try:
                size = validate_matrix(matrix, f"{key}/{raw_label}")
                if size != n:
                    raise ValueError(f"{key}/{raw_label}: 크기 {size}x{size} ≠ {n}x{n}")
            except ValueError as e:
                print(f"✗ {e}")
                continue
            normalized[label] = matrix
        result[n] = normalized
        if LABEL_CROSS in normalized and LABEL_X in normalized:
            print(f"✓ {key:<8} 필터 로드 완료 (Cross, X)")
        else:
            print(f"✗ {key:<8} 필터 불완전 (로드된 라벨: {', '.join(normalized) or '없음'})")
    return result


def evaluate_case(case_key, case, filters):
    """한 케이스를 판정한다. (cross_score, x_score, decision, expected) 반환, 오류 시 ValueError."""
    m = re.fullmatch(r"size_(\d+)_\d+", str(case_key))
    if not m:
        raise ValueError("패턴 키 형식 오류 (size_N_idx 형태가 아님)")
    n = int(m.group(1))

    if not isinstance(case, dict) or "input" not in case or "expected" not in case:
        raise ValueError("스키마 오류: input/expected 필드가 필요합니다")

    expected = normalize_label(case["expected"])
    if expected is None:
        raise ValueError(f"expected 라벨 오류: {case['expected']!r}")

    group = filters.get(n)
    if group is None:
        raise ValueError(f"size_{n} 필터가 없습니다")
    if LABEL_CROSS not in group or LABEL_X not in group:
        raise ValueError(f"size_{n} 필터에 Cross/X 중 누락된 항목이 있습니다")

    pattern = case["input"]
    size = validate_matrix(pattern, "input")
    if size != n:
        raise ValueError(f"크기 불일치: 패턴 {size}x{size}, 필터 {n}x{n}")

    cross_score = mac(pattern, group[LABEL_CROSS])
    x_score = mac(pattern, group[LABEL_X])
    return cross_score, x_score, judge(cross_score, x_score), expected


def run_json_mode():
    print("\n#---------------------------------------")
    print("# [1] 필터 로드")
    print("#---------------------------------------")
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"✗ {DATA_FILE} 파일을 찾을 수 없습니다. main.py와 같은 폴더에 두세요.")
        return
    except json.JSONDecodeError as e:
        print(f"✗ {DATA_FILE} JSON 파싱 실패: {e}")
        return
    if not isinstance(data, dict):
        print("✗ 최상위 구조가 객체가 아닙니다.")
        return

    filters = load_filters(data)

    print("\n#---------------------------------------")
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#---------------------------------------")
    patterns = data.get("patterns")
    if not isinstance(patterns, dict):
        print("✗ 'patterns' 섹션이 없거나 형식이 올바르지 않습니다.")
        patterns = {}

    total = passed = 0
    failures = []
    for key, case in patterns.items():
        total += 1
        print(f"--- {key} ---")
        try:
            cross_score, x_score, decision, expected = evaluate_case(key, case, filters)
        except ValueError as e:
            print(f"판정: 오류 | FAIL ({e})")
            failures.append((key, str(e)))
            continue
        except Exception as e:  # 예상하지 못한 오류도 프로그램을 중단시키지 않는다
            print(f"판정: 오류 | FAIL (예상치 못한 오류: {e})")
            failures.append((key, f"예상치 못한 오류: {e}"))
            continue

        print(f"Cross 점수: {cross_score}")
        print(f"X 점수: {x_score}")
        if decision == expected:
            passed += 1
            print(f"판정: {decision} | expected: {expected} | PASS")
        elif decision == UNDECIDED:
            reason = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
            print(f"판정: {decision} | expected: {expected} | FAIL ({reason})")
            failures.append((key, reason))
        else:
            reason = f"판정 {decision} ≠ expected {expected}"
            print(f"판정: {decision} | expected: {expected} | FAIL")
            failures.append((key, reason))

    run_performance()

    print("\n#---------------------------------------")
    print("# [4] 결과 요약")
    print("#---------------------------------------")
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {len(failures)}개")
    if failures:
        print("\n실패 케이스:")
        for key, reason in failures:
            print(f"- {key}: {reason}")
    print('(상세 원인 분석 및 복잡도 설명은 README.md의 "결과 리포트" 섹션 참고)')


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def main():
    print("=== Mini NPU Simulator ===\n")
    print("[모드 선택]\n")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    while True:
        choice = input("선택: ").strip()
        if choice == "1":
            run_user_mode()
            break
        if choice == "2":
            run_json_mode()
            break
        print("1 또는 2를 입력하세요.")


if __name__ == "__main__":
    main()