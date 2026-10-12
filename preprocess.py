"""사진 전처리: 영수증 종이 영역 찾기 → 원근 보정 → 대비 보정 → 이진화.
FE(브라우저)에서는 회전·리사이즈·압축까지만 하고, 이 정도 보정은 BE나 OCR API 쪽에 맡기는 걸 권장.
여기서는 '전처리가 로컬 OCR 정확도를 얼마나 바꾸는지' 보려는 용도."""
import cv2, numpy as np

def order(pts):
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)

def find_paper(gray):
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.25 * gray.size: return None
    ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    return order(ap.reshape(-1, 2).astype(np.float32)) if len(ap) == 4 else None

def preprocess(path, out_path):
    img = cv2.imread(str(path)); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    q = find_paper(gray)
    if q is not None:
        w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2])))
        h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
        M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
        gray = cv2.warpPerspective(gray, M, (w, h))
    if gray.shape[1] < 1000:  # 글자가 작으면 키우기
        f = 1000 / gray.shape[1]; gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    bg = cv2.medianBlur(gray, 41)  # 조명 그림자 제거(배경 나누기)
    norm = cv2.divide(gray, bg, scale=255)
    norm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(norm)
    th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    cv2.imwrite(str(out_path), th)
    return out_path
