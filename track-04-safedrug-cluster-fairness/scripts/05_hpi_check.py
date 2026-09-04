"""
§9 부수 확인: 랜덤 20개 방문의 diagnose 리스트 / HPI 섹션 / 약물 개수 나란히.
그리고 HPI 단독 단어수 분포.
"""
import textwrap
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

df = pd.read_pickle(OUT / "parsed.pkl").reset_index(drop=True)
sec = pd.read_pickle(OUT / "note_sections.pkl").reset_index(drop=True)
df = df.merge(sec[["SUBJECT_ID", "HADM_ID", "sec_history_of_present_illness"]], on=["SUBJECT_ID", "HADM_ID"])

hpi = df["sec_history_of_present_illness"].fillna("")
w = hpi.str.split().map(len)

lines = []
lines.append("# §9 HPI 부수 확인\n")
lines.append("## HPI 섹션 단독 단어수 분포\n")
lines.append("| min | Q1 | 중앙 | Q3 | p90 | p99 | max | 평균 | 0단어 |")
lines.append("|---|---|---|---|---|---|---|---|---|")
lines.append(
    f"| {w.min()} | {w.quantile(.25):.0f} | **{w.median():.0f}** | {w.quantile(.75):.0f} | "
    f"{w.quantile(.90):.0f} | {w.quantile(.99):.0f} | {w.max()} | {w.mean():.1f} | {(w==0).sum()}건 |"
)
lines.append("")

samp = df.sample(20, random_state=42).sort_values("HADM_ID")
lines.append("## 랜덤 20개 방문\n")
for _, r in samp.iterrows():
    lines.append(f"### HADM_ID {r.HADM_ID}  (SUBJECT_ID {r.SUBJECT_ID})")
    lines.append(f"- **처방 약물 수**: {len(r.drug_id_l)}   |   **진단 수**: {len(r.diag_id_l)}   |   HPI {len(str(r.sec_history_of_present_illness).split())}단어")
    lines.append(f"- **diagnose** ({len(r.diagnose_l)}): {', '.join(r.diagnose_l)}")
    lines.append("- **HPI**:")
    lines.append("  > " + "\n  > ".join(textwrap.wrap(str(r.sec_history_of_present_illness), 110)))
    lines.append("")

txt = "\n".join(lines)
(OUT / "09_hpi_check.md").write_text(txt, encoding="utf-8")
print(txt)
