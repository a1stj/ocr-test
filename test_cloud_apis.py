"""클라우드 OCR / LLM 후보를 같은 샘플·같은 채점 기준으로 비교하는 스크립트. (키가 있어야 실행됨)

사용법 (BE 또는 팀원 PC에서):
  pip install anthropic google-cloud-vision google-cloud-documentai pydantic
  # 1) Claude (멀티모달 LLM → 우리 JSON 스키마)
  export ANTHROPIC_API_KEY=...
  python3 test_cloud_apis.py claude --model claude-haiku-5-5
  python3 test_cloud_apis.py claude --model claude-sonnet-5-5
  # 2) Google Cloud Vision (글자만) → 우리 규칙 파서
  gcloud auth application-default login
  python3 test_cloud_apis.py vision
  # 3) Google Document AI Expense Parser (콘솔에서 Expense Parser 프로세서를 만들고 ID 확인)
  export DOCAI_PROJECT=... DOCAI_LOCATION=us DOCAI_PROCESSOR_ID=...
  python3 test_cloud_apis.py docai
  # 실제 영수증으로 하려면 samples/ 에 사진(.jpg)과 같은 이름의 정답 .json 을 넣으면 된다 (형식은 기존 json 참고)
결과: results/cloud_<엔진>.json  (키는 .env 에 두고 커밋하지 말 것)
"""
import base64, json, sys, time
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from run_local_ocr import S, R, cer, parse_receipt, parse_card, score_receipt, score_card

# ---------- 우리 서비스 형식 (BE가 FE에 주는 '초안' 형식과 같게) ----------
class Item(BaseModel):
    name: str
    qty: Optional[int] = None
    amount: float

class Receipt(BaseModel):
    merchant: Optional[str] = None
    date: Optional[str] = None  # YYYY-MM-DD
    currency: Optional[str] = None  # ISO 4217 (JPY, EUR, KRW)
    total: Optional[float] = None
    items: List[Item] = []

class CardTx(BaseModel):
    date: str
    merchant: str
    local_currency: Optional[str] = None
    local_amount: Optional[float] = None
    krw: Optional[int] = None

class CardStatement(BaseModel):
    transactions: List[CardTx]

PROMPT_RECEIPT = """이 영수증 이미지에서 결제 정보를 추출해.
- items: 실제로 산 품목만. 小計/合計/消費税/お預り/お釣り/クレジット/TOTAL/TVA 같은 줄은 넣지 마.
- 수량이 따로 찍혀 있으면 qty, amount는 그 줄의 금액(수량×단가)이야.
- total은 손님이 실제로 낸 합계(合計/TOTAL). 받은 돈이나 거스름돈이 아님.
- date는 YYYY-MM-DD, currency는 ISO 코드. 안 보이는 값은 null. 이미지에 없는 숫자는 만들지 마."""
PROMPT_CARD = """이 카드사 앱 캡처에서 결제 건을 모두 추출해. 각 건마다 date(YYYY-MM-DD), merchant(화면 그대로),
local_currency/local_amount(현지 통화 금액), krw(원화 청구액, 숫자만). 안 보이는 값은 null."""

def claude(img: Path, kind: str, model: str):
    import anthropic
    client = anthropic.Anthropic()
    media = "image/png" if img.suffix == ".png" else "image/jpeg"
    resp = client.messages.parse(
        model=model, max_tokens=4000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media,
                                          "data": base64.b64encode(img.read_bytes()).decode()}},
            {"type": "text", "text": PROMPT_CARD if kind == "card" else PROMPT_RECEIPT}]}],
        output_format=CardStatement if kind == "card" else Receipt,
    )
    return resp.parsed_output.model_dump(), {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}

def vision_text(img: Path):
    from google.cloud import vision
    r = vision.ImageAnnotatorClient().document_text_detection(image=vision.Image(content=img.read_bytes()))
    return r.full_text_annotation.text.splitlines()

def docai(img: Path):
    import os
    from google.cloud import documentai
    loc = os.environ.get("DOCAI_LOCATION", "us")
    c = documentai.DocumentProcessorServiceClient(client_options={"api_endpoint": f"{loc}-documentai.googleapis.com"})
    name = c.processor_path(os.environ["DOCAI_PROJECT"], loc, os.environ["DOCAI_PROCESSOR_ID"])
    mime = "image/png" if img.suffix == ".png" else "image/jpeg"
    doc = c.process_document(request=documentai.ProcessRequest(
        name=name, raw_document=documentai.RawDocument(content=img.read_bytes(), mime_type=mime))).document
    out = {"merchant": None, "date": None, "currency": None, "total": None, "items": []}
    num = lambda e: (e.normalized_value.money_value.units + e.normalized_value.money_value.nanos / 1e9) \
        if e.normalized_value and e.normalized_value.money_value.currency_code else None
    for e in doc.entities:
        if e.type_ == "supplier_name": out["merchant"] = e.mention_text
        elif e.type_ == "receipt_date": out["date"] = e.normalized_value.text or e.mention_text
        elif e.type_ == "currency": out["currency"] = e.normalized_value.text or e.mention_text
        elif e.type_ == "total_amount":
            out["total"] = num(e); out["currency"] = out["currency"] or e.normalized_value.money_value.currency_code
        elif e.type_ == "line_item":
            it = {"name": "", "amount": None}
            for p in e.properties:
                if p.type_ == "line_item/description": it["name"] = p.mention_text
                if p.type_ == "line_item/amount": it["amount"] = num(p)
                if p.type_ == "line_item/quantity": it["qty"] = p.mention_text
            out["items"].append(it)
    return out, {"raw_text": doc.text}

def norm_amounts(p):
    """1,000 같은 정수 통화는 int로 맞춰 채점 (JPY 36000.0 == 36000)."""
    if p.get("total") is not None and float(p["total"]).is_integer(): p["total"] = int(p["total"])
    for it in p.get("items", []):
        if it.get("amount") is not None and float(it["amount"]).is_integer(): it["amount"] = int(it["amount"])
    return p

def main():
    eng = sys.argv[1]; model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "claude-opus-5-5"
    rows = []
    for gtf in sorted(S.glob("*.json")):
        gt = json.loads(gtf.read_text()); img = next(S.glob(gtf.stem + ".*[gp]"))
        kind = "card" if gt["type"] == "card_statement" else "receipt"
        t0 = time.time(); extra = {}
        try:
            if eng == "claude":
                p, extra = claude(img, kind, model)
                if kind == "card":
                    sc = score_card(gt, {"krw": [t["krw"] for t in p["transactions"]],
                                         "jpy": [int(t["local_amount"] or 0) for t in p["transactions"]],
                                         "dates": [t["date"][5:].replace("-", ".") for t in p["transactions"]]})
                else: sc = score_receipt(gt, norm_amounts(p))
            elif eng == "vision":
                lines = vision_text(img); extra = {"cer": round(cer(gt["full_text"], "\n".join(lines)), 3)}
                p = parse_card(lines) if kind == "card" else parse_receipt(lines)
                sc = score_card(gt, p) if kind == "card" else score_receipt(gt, p)
            elif eng == "docai":
                if kind == "card": continue  # Expense Parser는 영수증용
                p, extra = docai(img); sc = score_receipt(gt, norm_amounts(p)); extra.pop("raw_text", None)
            else: sys.exit("engine: claude | vision | docai")
        except Exception as e:  # 한 장 실패해도 나머지는 계속
            p, sc = None, {"error": repr(e)}
        row = {"sample": gtf.stem, "engine": eng if eng != "claude" else model, "sec": round(time.time() - t0, 2), **sc, **extra, "parsed": p}
        rows.append(row); print({k: v for k, v in row.items() if k != "parsed"}, flush=True)
    (R / f"cloud_{eng if eng != 'claude' else model}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
