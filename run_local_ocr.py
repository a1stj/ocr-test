"""로컬 OCR 엔진 비교 + 규칙 기반 파서(= 'OCR 텍스트만 받고 우리가 직접 파싱'하는 방식) 정확도 측정.
엔진: Tesseract 5 (jpn 기본 fast 모델 / tessdata_best 모델), RapidOCR(PaddleOCR 계열 ONNX, 기본 중국어+영어 모델)
실행: python3 run_local_ocr.py [--best-dir 경로]  → results/local_ocr_results.json, results/*.txt
"""
import json, re, sys, time, unicodedata, difflib
from pathlib import Path
import pytesseract
from PIL import Image

HERE = Path(__file__).parent; S = HERE / "samples"; R = HERE / "results"; R.mkdir(exist_ok=True)
BEST = sys.argv[sys.argv.index("--best-dir") + 1] if "--best-dir" in sys.argv else None

def norm(s): return unicodedata.normalize("NFKC", s)

def cer(ref, hyp):
    a = re.sub(r"\s+", "", norm(ref)); b = re.sub(r"\s+", "", norm(hyp))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), 1)

# ---------- 엔진 ----------
def tess(img, lang, best=False, psm=4):
    cfg = f"--psm {psm}" + (f' --tessdata-dir "{BEST}"' if best and BEST else "")
    return [l for l in pytesseract.image_to_string(str(img), lang=lang, config=cfg).splitlines() if l.strip()]

_rapid = None
def rapid(img, lang=None):
    global _rapid
    if _rapid is None:
        from rapidocr_onnxruntime import RapidOCR; _rapid = RapidOCR()
    res, _ = _rapid(str(img))
    boxes = [((b[0][1] + b[2][1]) / 2, b[0][0], (b[2][1] - b[0][1]), t) for b, t, _ in (res or [])]
    boxes.sort(); lines = []; cur = []
    for yc, x, h, t in boxes:  # 같은 높이의 글자 조각을 한 줄로 묶기
        if cur and abs(yc - cur[-1][0]) > max(h, cur[-1][2]) * 0.45:
            lines.append(cur); cur = []
        cur.append((yc, x, h, t))
    if cur: lines.append(cur)
    return [" ".join(t for _, _, _, t in sorted(l, key=lambda z: z[1])) for l in lines]

# ---------- 규칙 기반 파서 ----------
# Tesseract가 ¥를 \\, 半, 羊 로 잘못 읽는 경우가 많아 같이 받는다. "36, 000"처럼 쉼표 뒤 공백도 허용.
AMT = re.compile(r"[¥￥\\半羊]\s*(\d{1,3}(?:,\s?\d{3})+|\d+)|(\d{1,3}(?:,\d{3})+|\d+)\s*円|(\d+[\.,]\d{2})\s*EUR")
SKIP = re.compile(r"小計|合計|税|預|釣|対象|クレジット|現金|TOTAL|TVA|CB |VISA|番号|No\.|TEL")
QTY = re.compile(r"^\s*\d+\s*(点|枚|個|コ)|[x×X]\s*[@e0]|^\s*\d+\s*[x×]?\s*$")

def to_num(s):
    s = re.sub(r"[,\s]", "", s)
    try: return float(s) if "." in s else int(s)
    except ValueError: return None

def amounts(line):
    out = []
    for m in AMT.finditer(norm(line)):
        v = to_num(next(g for g in m.groups() if g)); 
        if v is not None: out.append((m.start(), v))
    return out

def parse_receipt(lines):
    L = [re.sub(r"(?<=[^\x00-\x7f]) (?=[^\x00-\x7f])", "", norm(l)) for l in lines]  # 일본어 글자 사이 공백 제거
    total = None; items = []; date = None; total = None; items = []; date = None
    for i, l in enumerate(L):
        if date is None:
            m = re.search(r"(20\d\d)\s*[/年.\-]\s*(\d{1,2})\s*[/月.\-]\s*(\d{1,2})", l) or \
                re.search(r"(\d{2})/(\d{2})/(20\d\d)", l)
            if m:
                g = m.groups(); date = (f"{g[0]}-{int(g[1]):02d}-{int(g[2]):02d}" if len(g[0]) == 4
                                        else f"{g[2]}-{int(g[1]):02d}-{int(g[0]):02d}")
        if total is None and re.search(r"合\s*計|TOTAL", l) and "小" not in l:
            a = amounts(l) or (amounts(L[i + 1]) if i + 1 < len(L) else [])
            if a: total = a[-1][1]
    pending = None
    for i, l in enumerate(L):
        a = amounts(l)
        if SKIP.search(l) or (i > 0 and SKIP.search(L[i - 1]) and not a and False):
            pending = None; continue
        if not a:
            t = l.strip()
            if t and not QTY.search(t): pending = t
            continue
        name = AMT.sub("", l).strip()
        if not name or QTY.search(name):
            name = pending or name
        if i > 0 and SKIP.search(L[i - 1]) and not AMT.sub("", l).strip():
            continue  # '合計' 다음 줄로 밀려난 금액 등
        items.append({"name": name, "amount": a[-1][1]}); pending = None
    return {"date": date, "total": total, "items": items}

def score_receipt(gt, p):
    gt_amts = [i["amount"] for i in gt["items"]]; got = [i["amount"] for i in p["items"]]
    pool = list(got); hit = 0; name_hit = 0
    for g in gt["items"]:
        if g["amount"] in pool:
            pool.remove(g["amount"]); hit += 1
            cand = [x["name"] for x in p["items"] if x["amount"] == g["amount"]]
            if any(difflib.SequenceMatcher(None, re.sub(r"\s", "", norm(g["name"])), re.sub(r"\s", "", norm(c))).ratio() >= 0.6 for c in cand): name_hit += 1
    return {"total_ok": p["total"] == gt["total"], "date_ok": p["date"] == gt["date"],
            "item_amount_recall": f"{hit}/{len(gt_amts)}", "item_name_ok": f"{name_hit}/{len(gt_amts)}",
            "extra_items": len(got) - hit}

def parse_card(lines):
    t = norm("\n".join(lines))
    # '원'이 %, 4 등으로 깨지는 경우가 있어, 가맹점 줄(대문자 영문) 맨 끝 숫자를 원화로 본다
    krw = [int(m.replace(",", "")) for m in re.findall(r"^[A-Z][A-Z .&']+?\s+(\d{1,3}(?:,\d{3})+)", t, re.M)]
    return {"krw": krw,
            "jpy": [int(x.replace(",", "")) for x in re.findall(r"JPY\s*([\d,]+)", t)],
            "dates": re.findall(r"2026\.(\d\d\.\d\d)", t)}

def score_card(gt, p):
    tx = gt["transactions"]
    return {"krw_ok": f"{sum(t['krw'] in p['krw'] for t in tx)}/{len(tx)}",
            "jpy_ok": f"{sum(t['local_amount'] in p['jpy'] for t in tx)}/{len(tx)}",
            "date_ok": f"{sum(t['date'][5:].replace('-', '.') in p['dates'] for t in tx)}/{len(tx)}"}

PREP = HERE / "prep"; PREP.mkdir(parents=True, exist_ok=True)
def prep(img):
    if "_photo" not in img.name: return img
    from preprocess import preprocess
    return preprocess(img, PREP / (img.stem + ".png"))

_easy = {}
def easy(img, kind):
    import easyocr
    langs = ["ko", "en"] if kind == "card" else ["ja", "en"]
    r = _easy.setdefault(kind, easyocr.Reader(langs, gpu=False, verbose=False))
    res = r.readtext(str(img))
    boxes = sorted(((b[0][1] + b[2][1]) / 2, b[0][0], b[2][1] - b[0][1], t) for b, t, _ in res)
    lines, cur = [], []
    for yc, x, h, t in boxes:
        if cur and abs(yc - cur[-1][0]) > max(h, cur[-1][2]) * 0.45: lines.append(cur); cur = []
        cur.append((yc, x, h, t))
    if cur: lines.append(cur)
    return [" ".join(t for *_, t in sorted(l, key=lambda z: z[1])) for l in lines]

ENGINES = {
    "tesseract_fast": lambda img, kind: tess(img, "kor+eng" if kind == "card" else "jpn+eng"),
    "tesseract_best": lambda img, kind: tess(img, "kor+eng" if kind == "card" else "jpn+eng", best=True),
    "rapidocr": lambda img, kind: rapid(img),
    "tesseract_best+prep": lambda img, kind: tess(prep(img), "kor+eng" if kind == "card" else "jpn+eng", best=True),
    "rapidocr+prep": lambda img, kind: rapid(prep(img)),
}
try:
    import easyocr  # noqa
    ENGINES["easyocr"] = easy; ENGINES["easyocr+prep"] = lambda img, kind: easy(prep(img), kind)
except ImportError: pass
if not BEST: ENGINES.pop("tesseract_best"); ENGINES.pop("tesseract_best+prep")

def main():
    results = []
    for gtf in sorted(S.glob("*.json")):
        gt = json.loads(gtf.read_text()); img = next(S.glob(gtf.stem + ".*[gp]"))
        kind = "card" if gt["type"] == "card_statement" else "receipt"
        for name, fn in ENGINES.items():
            if name.endswith("+prep") and "_photo" not in gtf.stem: continue
            t0 = time.time(); lines = fn(img, kind); dt = time.time() - t0
            (R / f"{gtf.stem}__{name}.txt").write_text("\n".join(lines))
            row = {"sample": gtf.stem, "engine": name, "sec": round(dt, 2), "cer": round(cer(gt["full_text"], "\n".join(lines)), 3)}
            if kind == "card": row.update(score_card(gt, parse_card(lines)))
            else:
                p = parse_receipt(lines); row.update(score_receipt(gt, p)); row["parsed"] = p
            results.append(row)
            print({k: v for k, v in row.items() if k != "parsed"}, flush=True)
    (R / "local_ocr_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
