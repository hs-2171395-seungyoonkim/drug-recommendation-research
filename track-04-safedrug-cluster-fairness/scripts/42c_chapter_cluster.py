"""§ICD-9 Chapter 군집화 — 주진단(SEQ_NUM=1)의 ICD-9 대분류(Chapter)를 그대로 군집으로 사용한다.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

df = pd.read_pickle(OUT / "40_dxtext.pkl") # 원본 사용

def get_chapter(code):
    if not isinstance(code, str):
        return "Unknown"
    code = code.strip().upper()
    if code.startswith('V'):
        return "V_Supp"
    if code.startswith('E'):
        return "E_Supp"
    
    try:
        num = int(code[:3])
    except:
        return "Unknown"
        
    if 1 <= num <= 139: return "01_Infectious"
    elif 140 <= num <= 239: return "02_Neoplasms"
    elif 240 <= num <= 279: return "03_Endocrine"
    elif 280 <= num <= 289: return "04_Blood"
    elif 290 <= num <= 319: return "05_Mental"
    elif 320 <= num <= 389: return "06_Nervous"
    elif 390 <= num <= 459: return "07_Circulatory"
    elif 460 <= num <= 519: return "08_Respiratory"
    elif 520 <= num <= 579: return "09_Digestive"
    elif 580 <= num <= 629: return "10_Genitourinary"
    elif 630 <= num <= 679: return "11_Pregnancy"
    elif 680 <= num <= 709: return "12_Skin"
    elif 710 <= num <= 739: return "13_Musculoskeletal"
    elif 740 <= num <= 759: return "14_Congenital"
    elif 760 <= num <= 779: return "15_Perinatal"
    elif 780 <= num <= 799: return "16_Symptoms"
    elif 800 <= num <= 999: return "17_Injury"
    else: return "Unknown"

df['chapter'] = df['seq1_code'].apply(get_chapter)

# 카테고리를 숫자로 매핑
chapters = sorted(df['chapter'].unique())
c2i = {c: i for i, c in enumerate(chapters)}
df['cluster'] = df['chapter'].map(c2i)

print(f"[.] 주진단 Chapter 분류 완료. 총 {len(chapters)}개 군집 생성")
for c in chapters:
    print(f"    - {c}: {sum(df['chapter'] == c)}명")

df.to_csv(OUT / "dxtext_chapter_assignments.csv", index=False, encoding="utf-8-sig")
print("[+] out/dxtext_chapter_assignments.csv 생성 완료")
