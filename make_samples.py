"""합성 테스트 이미지 생성기: 일본 영수증 3종 + 유럽 영수증 1종 + 한국 카드앱 해외결제 캡처 1종.
각 샘플마다 정답(ground truth) JSON을 같이 만든다. 사진 느낌 변형본(_photo)도 생성.
실행: python3 make_samples.py  (Pillow, numpy 필요 / 폰트: IPAGothic, Noto Sans CJK)
"""
import json, math, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

OUT = Path(__file__).parent / "samples"
OUT.mkdir(exist_ok=True)
JP_FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
KR_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
KR_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
random.seed(7); np.random.seed(7)

def yen(n): return f"¥{n:,}"

def render_receipt(lines, width=640, size=26, font=JP_FONT):
    """lines: list of (left, right, opts). opts: center/big/rule"""
    f = ImageFont.truetype(font, size); fb = ImageFont.truetype(font, int(size * 1.45))
    lh = int(size * 1.55); pad = 36
    h = pad * 2 + sum(int(lh * (1.5 if o.get("big") else 1)) for _, _, o in lines)
    im = Image.new("L", (width, h), 255); d = ImageDraw.Draw(im); y = pad
    for left, right, o in lines:
        ff = fb if o.get("big") else f
        if o.get("rule"):
            d.line((pad, y + lh // 2, width - pad, y + lh // 2), fill=0, width=2)
        elif o.get("center"):
            w = d.textlength(left, font=ff); d.text(((width - w) / 2, y), left, font=ff, fill=0)
        else:
            d.text((pad + o.get("indent", 0), y), left, font=ff, fill=0)
            if right:
                w = d.textlength(right, font=ff); d.text((width - pad - w, y), right, font=ff, fill=0)
        y += int(lh * (1.5 if o.get("big") else 1))
    return im.convert("RGB")

def text_of(lines):
    return "\n".join((l + (" " + r if r else "")).strip() for l, r, o in lines if not o.get("rule"))

def photoize(im, angle=3.5, blur=1.1, fade=1.0, seed=0):
    """폰으로 찍은 느낌: 배경 위에 회전, 원근, 그림자, 흐림, 노이즈, JPEG 압축."""
    rng = np.random.default_rng(seed)
    if fade != 1.0:
        im = ImageEnhance.Contrast(im).enhance(fade)
        im = ImageEnhance.Brightness(im).enhance(1.08)
    w, h = im.size
    bg = Image.new("RGB", (w + 160, h + 160), (92, 84, 74))
    bg.paste(im, (80, 80))
    bg = bg.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(92, 84, 74))
    W, H = bg.size; dx = W * 0.04
    coeffs = _persp([(0, 0), (W, 0), (W, H), (0, H)], [(dx, 0), (W - dx * 0.3, 0), (W, H), (0, H)])
    bg = bg.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(92, 84, 74))
    a = np.asarray(bg).astype(np.float32)
    grad = np.linspace(0.72, 1.05, W)[None, :, None] * np.linspace(1.02, 0.85, H)[:, None, None]
    a = a * grad + rng.normal(0, 7, a.shape)
    bg = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
    return bg

def _persp(pa, pb):
    m = []
    for (x, y), (X, Y) in zip(pa, pb):
        m.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y]); m.append([0, 0, 0, X, Y, 1, -y * X, -y * Y])
    A = np.array(m, dtype=float); B = np.array(pa, dtype=float).reshape(8)
    return np.linalg.solve(A, B).tolist()

def save(name, im, gt, lines=None, photo=None):
    im.save(OUT / f"{name}.png")
    if lines is not None: gt["full_text"] = text_of(lines)
    (OUT / f"{name}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2))
    if photo:
        photoize(im, **photo).save(OUT / f"{name}_photo.jpg", quality=62)
        (OUT / f"{name}_photo.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2))

# ---------- R1 편의점 (경감세율 8%/10% 혼재, 받은돈/거스름돈) ----------
items = [("おにぎり 鮭", 160, "※"), ("緑茶 525ml", 150, "※"), ("からあげ棒", 220, "※"),
         ("のど飴", 238, "※"), ("ビニール傘", 770, "")]
sub = sum(p for _, p, _ in items); tax8 = round(sum(p for _, p, m in items if m) * 8 / 108)
tax10 = round(sum(p for _, p, m in items if not m) * 10 / 110)
L = [("ハッピーマート 大阪難波店", "", {"center": 1}), ("大阪市中央区難波1-2-3", "", {"center": 1}),
     ("TEL 06-1234-5678", "", {"center": 1}), ("登録番号 T1234567890123", "", {"center": 1}),
     ("2026年10月03日(土) 14:22", "", {}), ("レジ02 責No.105", "", {}), ("", "", {"rule": 1})]
L += [(f"{n}{' ' + m if m else ''}", yen(p), {}) for n, p, m in items]
L += [("", "", {"rule": 1}), ("小計", yen(sub), {}), ("(8%対象", yen(sub - 770) + ")", {}),
      ("(内消費税等 8%", yen(tax8) + ")", {}), ("(10%対象", yen(770) + ")", {}),
      ("(内消費税等 10%", yen(tax10) + ")", {}), ("合計", yen(sub), {"big": 1}),
      ("お預り", yen(2000), {}), ("お釣", yen(2000 - sub), {}),
      ("※印は軽減税率対象商品です", "", {}), ("またのご来店をお待ちしております", "", {"center": 1})]
save("r1_konbini", render_receipt(L),
     {"type": "receipt", "merchant": "ハッピーマート 大阪難波店", "date": "2026-10-03", "currency": "JPY",
      "total": sub, "items": [{"name": n, "amount": p} for n, p, _ in items]}, L, photo={"angle": 3.5, "seed": 1})

# ---------- R2 이자카야 (수량 2줄 형식, 내세, 합계 ¥24,000 = 시나리오 B) ----------
items = [("生ビール", 8, 600), ("ハイボール", 4, 550), ("枝豆", 2, 400), ("焼き鳥盛り合わせ", 2, 1800),
         ("刺身五点盛り", 1, 2800), ("鶏の唐揚げ", 2, 780), ("だし巻き玉子", 2, 650), ("お通し", 4, 500),
         ("締めラーメン", 2, 900), ("バニラアイス", 3, 380), ("ウーロン茶", 4, 500)]
tot = sum(q * p for _, q, p in items); assert tot == 24000
L = [("居酒屋 なにわ屋", "", {"center": 1, "big": 1}), ("大阪市中央区道頓堀2-4-6", "", {"center": 1}),
     ("2026/10/03 21:47  卓No.12  4名", "", {}), ("伝票No.0381", "", {}), ("", "", {"rule": 1})]
for n, q, p in items:
    if q == 1: L.append((n, yen(p), {}))
    else: L += [(n, "", {}), (f"{q}点 × @{p:,}", yen(q * p), {"indent": 40})]
L += [("", "", {"rule": 1}), ("小計", yen(tot), {}), ("(内消費税等 10%", yen(round(tot * 10 / 110)) + ")", {}),
      ("合計", yen(tot), {"big": 1}), ("お預り", yen(30000), {}), ("お釣り", yen(6000), {}),
      ("ありがとうございました", "", {"center": 1})]
save("r2_izakaya", render_receipt(L),
     {"type": "receipt", "merchant": "居酒屋 なにわ屋", "date": "2026-10-03", "currency": "JPY", "total": tot,
      "items": [{"name": n, "qty": q, "amount": q * p} for n, q, p in items]}, L,
     photo={"angle": -2.5, "blur": 1.3, "fade": 0.55, "seed": 2})

# ---------- R3 테마파크 카드 결제 (카드번호 마스킹, 시나리오 C) ----------
L = [("テーマパーク チケットセンター", "", {"center": 1}), ("大阪市此花区桜島2-1-33", "", {"center": 1}),
     ("2026年10月04日 09:12", "", {}), ("", "", {"rule": 1}),
     ("1デイ・パス 大人", "", {}), ("4枚 × @9,000", yen(36000), {"indent": 40}), ("", "", {"rule": 1}),
     ("合計", yen(36000), {"big": 1}), ("(内消費税等", yen(3273) + ")", {}),
     ("クレジット", yen(36000), {}), ("カード会社 VISA", "", {}), ("会員番号 ************1234", "", {}),
     ("有効期限 **/**", "", {}), ("承認番号 0123456", "", {}), ("支払区分 一括", "", {}),
     ("お客様控え", "", {"center": 1})]
save("r3_themepark_card", render_receipt(L),
     {"type": "receipt", "merchant": "テーマパーク チケットセンター", "date": "2026-10-04", "currency": "JPY",
      "total": 36000, "items": [{"name": "1デイ・パス 大人", "qty": 4, "amount": 36000}]}, L,
     photo={"angle": 5, "blur": 1.0, "seed": 3})

# ---------- R4 유럽 카페 (소수점 통화 EUR) ----------
items = [("2 x Cafe creme", 9.00), ("Croissant", 2.20), ("Pain au chocolat", 2.60), ("Jus d'orange", 5.50)]
tot = round(sum(p for _, p in items), 2)
L = [("CAFE DU MARCHE", "", {"center": 1, "big": 1}), ("12 Rue Cler, 75007 Paris", "", {"center": 1}),
     ("05/10/2026 08:41", "", {}), ("", "", {"rule": 1})]
L += [(n, f"{p:.2f} EUR", {}) for n, p in items]
L += [("", "", {"rule": 1}), ("TOTAL TTC", f"{tot:.2f} EUR", {"big": 1}), ("dont TVA 10%", "1.75 EUR", {}),
      ("CB VISA", f"{tot:.2f} EUR", {}), ("Merci de votre visite", "", {"center": 1})]
save("r4_paris_cafe", render_receipt(L, font=KR_FONT),
     {"type": "receipt", "merchant": "CAFE DU MARCHE", "date": "2026-10-05", "currency": "EUR", "total": tot,
      "items": [{"name": n, "amount": p} for n, p in items]}, L, photo={"angle": -4, "seed": 4})

# ---------- C1 한국 카드앱 해외결제 내역 캡처 ----------
tx = [("10.06", "TAXI NANIWA OSAKA JP", "JPY 4,000", 36840),
      ("10.04", "USJ LLC OSAKA JP", "JPY 36,000", 331200),
      ("10.03", "NANIWAYA DOTONBORI JP", "JPY 24,000", 220560),
      ("10.03", "HAPPY MART NAMBA JP", "JPY 1,538", 14166),
      ("10.02", "KANSAI AIRPORT EXP JP", "JPY 3,640", 33520)]
W, H = 1080, 1580
im = Image.new("RGB", (W, H), (246, 247, 249)); d = ImageDraw.Draw(im)
fT = ImageFont.truetype(KR_BOLD, 46); fm = ImageFont.truetype(KR_FONT, 34); fs = ImageFont.truetype(KR_FONT, 30)
fb = ImageFont.truetype(KR_BOLD, 38)
d.text((48, 60), "9:41", font=fs, fill=(20, 20, 20)); d.text((48, 140), "이용내역", font=fT, fill=(20, 20, 20))
d.text((48, 220), "해외 이용 · 2026년 10월", font=fm, fill=(110, 110, 120))
y = 300; lines_gt = []
for date, mer, loc, krw in tx:
    d.rounded_rectangle((32, y, W - 32, y + 200), 24, fill=(255, 255, 255))
    d.text((72, y + 30), mer, font=fb, fill=(20, 20, 20))
    d.text((72, y + 92), f"2026.{date}  해외승인 · 일시불", font=fs, fill=(120, 120, 130))
    d.text((72, y + 140), loc, font=fs, fill=(120, 120, 130))
    s = f"{krw:,}원"; d.text((W - 72 - d.textlength(s, font=fb), y + 30), s, font=fb, fill=(20, 20, 20))
    lines_gt.append(f"{mer} {krw:,}원\n2026.{date} 해외승인 · 일시불\n{loc}")
    y += 230
d.text((48, y + 20), "청구 예정 금액은 매입 시점 환율에 따라 달라질 수 있어요", font=fs, fill=(140, 140, 150))
save("c1_card_app_kr", im,
     {"type": "card_statement", "transactions": [{"date": "2026-" + dt.replace(".", "-"), "merchant": m,
       "local_currency": "JPY", "local_amount": int(l.split()[1].replace(",", "")), "krw": k} for dt, m, l, k in tx],
      "full_text": "이용내역\n해외 이용 · 2026년 10월\n" + "\n".join(lines_gt)})
print(sorted(p.name for p in OUT.iterdir()))
