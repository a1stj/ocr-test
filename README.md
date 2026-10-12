# ocr-test

해외여행 공동비용 정산 서비스에서 쓸 영수증·카드 결제내역 OCR 후보를 **같은 샘플, 같은 채점 기준**으로 비교하는 레포입니다.

## 비교 대상
| 구분 | 후보 | 실행 스크립트 |
| --- | --- | --- |
| 로컬 OCR (무료) | Tesseract 5 (jpn), RapidOCR | `run_local_ocr.py` |
| 영수증 전용 API | Google Document AI Expense Parser | `test_cloud_apis.py docai` |
| 글자 인식 API | Google Cloud Vision + 규칙 파서 | `test_cloud_apis.py vision` |
| 멀티모달 LLM | Claude (Haiku / Sonnet 등) | `test_cloud_apis.py claude --model <모델>` |

## 폴더 구조
```
samples/          테스트 이미지 + 같은 이름의 정답 .json
results/          엔진별 결과 (json, OCR 원문 txt)
make_samples.py   합성 샘플 생성 (오사카 시나리오: 편의점, 이자카야, 테마파크 카드전표, 파리 카페, 카드앱 캡처)
preprocess.py     사진 전처리 (종이 영역 찾기, 원근 보정, 조명 보정, 이진화)
run_local_ocr.py  로컬 OCR + 규칙 파서 채점
test_cloud_apis.py 클라우드 API / LLM 채점 (API 키 필요)
```

## 시작하기
```bash
# 로컬 OCR
sudo apt-get install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-kor fonts-noto-cjk fonts-ipafont-gothic   # macOS: brew install tesseract tesseract-lang
pip install -r requirements.txt
python3 make_samples.py
python3 run_local_ocr.py                      # (선택) --best-dir <tessdata_best 폴더>

# 클라우드 비교: .env.example 을 .env 로 복사해 키를 채운 뒤
set -a; source .env; set +a
python3 test_cloud_apis.py claude --model claude-haiku-5-5
python3 test_cloud_apis.py docai
python3 test_cloud_apis.py vision             # gcloud auth application-default login 필요
```

## 실제 영수증 추가하기
`samples/` 에 `이름.jpg` 와 같은 이름의 정답 `이름.json` 을 넣으면 모든 스크립트가 같이 채점합니다.

```json
{"type": "receipt", "merchant": "居酒屋 なにわ屋", "date": "2026-10-03", "currency": "JPY", "total": 24000,
 "items": [{"name": "生ビール", "qty": 8, "amount": 4800}], "full_text": "(선택) 영수증 전체 글자"}
```

## 채점 기준
- `total_ok` 합계가 정확히 맞았는지, `date_ok` 날짜가 맞았는지
- `item_amount_recall` 정답 품목 금액 중 찾은 개수, `item_name_ok` 품목 이름까지 맞은 개수 (유사도 0.6 이상)
- `extra_items` 품목이 아닌 줄(소계, 세금, 받은 돈 등)이 품목으로 들어온 개수
- `cer` 글자 오류율 (로컬 OCR, Vision)

## 지금까지 결과 (2026-10-12, 합성 샘플 9장, 로컬 OCR)
| 조건 | 엔진 | 합계 | 품목 금액 | 품목 이름 |
| --- | --- | --- | --- | --- |
| 깨끗한 이미지 | Tesseract best | 4/4 | 21/21 | 20/21 |
| 사진판 | Tesseract best | 1/4 | 9/21 | 8/21 |
| 사진판 + 전처리 | Tesseract best | 3/4 | 21/21 | 12/21 |
| 카드앱 캡처 | 모든 엔진 | 결제 5건 모두 맞음 | | |

합성 이미지는 실제 영수증보다 쉽고, 규칙 파서도 이 샘플을 보며 다듬었기 때문에 실제 성능은 더 낮을 수 있습니다.

## 꼭 지킬 것
- **실제 영수증 사진은 카드번호, 이름, 승인번호를 가린 뒤** 올립니다. 원본은 `samples/private/` 에 두면 커밋되지 않습니다.
