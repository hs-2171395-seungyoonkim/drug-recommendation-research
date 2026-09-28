# SafeDrug 개입 E: 디코딩 단계 방문별 DDI 예산 리포트

## 1. 개입 설명

- 개입 E는 재학습 없이, 평가된 확률 벡터에 대해 방문별 DDI 예산(b_v)을 디코딩 규칙(REMOVE→ADD)으로 적용한다. b_v는 훈련 데이터의 그룹별 실제 DDI율에서 유도되며(축: ccs_group=주요, long_k10=민감도), 방문 하나에만 적용되므로 다른 그룹으로 새어나갈 수 없다.
- **ccs**: CCS 주진단 축(급성 카테고리)의 예산. **k10**: long_k10 축의 예산(민감도 확인). **global**: 모든 방문에 동일한 r_global 예산을 적용하는 대조군.

## 2. 전체 지표 (test, seed x variant)

| seed | variant | p_floor | jaccard_mean | f1_mean | prauc_mean | ddi_rate | avg_n_med_pred | n_test_visits |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline |  | 0.5078 | 0.6654 | 0.7632 | 0.0619 | 20.1921 | 2264 |
| 0 | ccs | 0.4000 | 0.5137 | 0.6701 | 0.7632 | 0.0605 | 22.1705 | 2264 |
| 0 | k10 | 0.4000 | 0.5139 | 0.6702 | 0.7632 | 0.0603 | 22.1617 | 2264 |
| 0 | global | 0.4000 | 0.5138 | 0.6701 | 0.7632 | 0.0610 | 22.1935 | 2264 |
| 1 | baseline |  | 0.5114 | 0.6682 | 0.7640 | 0.0615 | 20.6409 | 2264 |
| 1 | ccs | 0.4500 | 0.5127 | 0.6689 | 0.7640 | 0.0586 | 21.7858 | 2264 |
| 1 | k10 | 0.4500 | 0.5124 | 0.6686 | 0.7640 | 0.0584 | 21.7708 | 2264 |
| 1 | global | 0.4500 | 0.5125 | 0.6688 | 0.7640 | 0.0588 | 21.8004 | 2264 |
| 2 | baseline |  | 0.5147 | 0.6712 | 0.7642 | 0.0623 | 20.5300 | 2264 |
| 2 | ccs | 0.4500 | 0.5138 | 0.6701 | 0.7642 | 0.0603 | 21.5839 | 2264 |
| 2 | k10 | 0.4500 | 0.5138 | 0.6701 | 0.7642 | 0.0600 | 21.5667 | 2264 |
| 2 | global | 0.4500 | 0.5135 | 0.6697 | 0.7642 | 0.0605 | 21.5932 | 2264 |
| 3 | baseline |  | 0.5102 | 0.6671 | 0.7655 | 0.0603 | 20.3759 | 2264 |
| 3 | ccs | 0.4500 | 0.5107 | 0.6671 | 0.7655 | 0.0583 | 21.3494 | 2264 |
| 3 | k10 | 0.4000 | 0.5113 | 0.6677 | 0.7655 | 0.0591 | 22.1718 | 2264 |
| 3 | global | 0.4000 | 0.5109 | 0.6673 | 0.7655 | 0.0596 | 22.2005 | 2264 |

## 3. 격차(gap) 통계: excess DDI (축별)

| seed | variant | axis | statistic | observed | p_raw |
|---|---|---|---|---|---|
| 0 | baseline | long_k10 | range | 0.0276 | 0.0012 |
| 0 | baseline | long_k10 | weighted_sd | 0.0063 | 0.0003 |
| 0 | baseline | ccs_group | range | 0.0550 | 0.0004 |
| 0 | baseline | ccs_group | weighted_sd | 0.0109 | 0.0001 |
| 0 | ccs | long_k10 | range | 0.0302 | 0.0002 |
| 0 | ccs | long_k10 | weighted_sd | 0.0068 | 0.0001 |
| 0 | ccs | ccs_group | range | 0.0546 | 0.0002 |
| 0 | ccs | ccs_group | weighted_sd | 0.0102 | 0.0001 |
| 0 | k10 | long_k10 | range | 0.0237 | 0.0089 |
| 0 | k10 | long_k10 | weighted_sd | 0.0054 | 0.0079 |
| 0 | k10 | ccs_group | range | 0.0623 | 0.0001 |
| 0 | k10 | ccs_group | weighted_sd | 0.0119 | 0.0001 |
| 0 | global | long_k10 | range | 0.0320 | 0.0002 |
| 0 | global | long_k10 | weighted_sd | 0.0073 | 0.0001 |
| 0 | global | ccs_group | range | 0.0648 | 0.0001 |
| 0 | global | ccs_group | weighted_sd | 0.0124 | 0.0001 |
| 1 | baseline | long_k10 | range | 0.0344 | 0.0001 |
| 1 | baseline | long_k10 | weighted_sd | 0.0078 | 0.0001 |
| 1 | baseline | ccs_group | range | 0.0623 | 0.0001 |
| 1 | baseline | ccs_group | weighted_sd | 0.0120 | 0.0001 |
| 1 | ccs | long_k10 | range | 0.0327 | 0.0001 |
| 1 | ccs | long_k10 | weighted_sd | 0.0075 | 0.0001 |
| 1 | ccs | ccs_group | range | 0.0572 | 0.0002 |
| 1 | ccs | ccs_group | weighted_sd | 0.0106 | 0.0001 |
| 1 | k10 | long_k10 | range | 0.0277 | 0.0012 |
| 1 | k10 | long_k10 | weighted_sd | 0.0065 | 0.0003 |
| 1 | k10 | ccs_group | range | 0.0628 | 0.0001 |
| 1 | k10 | ccs_group | weighted_sd | 0.0117 | 0.0001 |
| 1 | global | long_k10 | range | 0.0340 | 0.0001 |
| 1 | global | long_k10 | weighted_sd | 0.0078 | 0.0001 |
| 1 | global | ccs_group | range | 0.0633 | 0.0001 |
| 1 | global | ccs_group | weighted_sd | 0.0121 | 0.0001 |
| 2 | baseline | long_k10 | range | 0.0270 | 0.0015 |
| 2 | baseline | long_k10 | weighted_sd | 0.0056 | 0.0040 |
| 2 | baseline | ccs_group | range | 0.0542 | 0.0007 |
| 2 | baseline | ccs_group | weighted_sd | 0.0106 | 0.0001 |
| 2 | ccs | long_k10 | range | 0.0332 | 0.0002 |
| 2 | ccs | long_k10 | weighted_sd | 0.0074 | 0.0001 |
| 2 | ccs | ccs_group | range | 0.0534 | 0.0006 |
| 2 | ccs | ccs_group | weighted_sd | 0.0099 | 0.0001 |
| 2 | k10 | long_k10 | range | 0.0267 | 0.0018 |
| 2 | k10 | long_k10 | weighted_sd | 0.0058 | 0.0027 |
| 2 | k10 | ccs_group | range | 0.0620 | 0.0002 |
| 2 | k10 | ccs_group | weighted_sd | 0.0117 | 0.0001 |
| 2 | global | long_k10 | range | 0.0372 | 0.0001 |
| 2 | global | long_k10 | weighted_sd | 0.0086 | 0.0001 |
| 2 | global | ccs_group | range | 0.0670 | 0.0001 |
| 2 | global | ccs_group | weighted_sd | 0.0129 | 0.0001 |
| 3 | baseline | long_k10 | range | 0.0314 | 0.0002 |
| 3 | baseline | long_k10 | weighted_sd | 0.0070 | 0.0001 |
| 3 | baseline | ccs_group | range | 0.0641 | 0.0001 |
| 3 | baseline | ccs_group | weighted_sd | 0.0120 | 0.0001 |
| 3 | ccs | long_k10 | range | 0.0297 | 0.0003 |
| 3 | ccs | long_k10 | weighted_sd | 0.0068 | 0.0001 |
| 3 | ccs | ccs_group | range | 0.0660 | 0.0001 |
| 3 | ccs | ccs_group | weighted_sd | 0.0107 | 0.0001 |
| 3 | k10 | long_k10 | range | 0.0240 | 0.0076 |
| 3 | k10 | long_k10 | weighted_sd | 0.0054 | 0.0099 |
| 3 | k10 | ccs_group | range | 0.0622 | 0.0001 |
| 3 | k10 | ccs_group | weighted_sd | 0.0118 | 0.0001 |
| 3 | global | long_k10 | range | 0.0353 | 0.0001 |
| 3 | global | long_k10 | weighted_sd | 0.0079 | 0.0001 |
| 3 | global | ccs_group | range | 0.0646 | 0.0001 |
| 3 | global | ccs_group | weighted_sd | 0.0126 | 0.0001 |
| mean | baseline | ccs_group | range | 0.0589 | 0.0003 |
| mean | baseline | ccs_group | weighted_sd | 0.0114 | 0.0001 |
| mean | baseline | long_k10 | range | 0.0301 | 0.0007 |
| mean | baseline | long_k10 | weighted_sd | 0.0066 | 0.0011 |
| mean | ccs | ccs_group | range | 0.0578 | 0.0003 |
| mean | ccs | ccs_group | weighted_sd | 0.0104 | 0.0001 |
| mean | ccs | long_k10 | range | 0.0314 | 0.0002 |
| mean | ccs | long_k10 | weighted_sd | 0.0071 | 0.0001 |
| mean | global | ccs_group | range | 0.0649 | 0.0001 |
| mean | global | ccs_group | weighted_sd | 0.0125 | 0.0001 |
| mean | global | long_k10 | range | 0.0346 | 0.0001 |
| mean | global | long_k10 | weighted_sd | 0.0079 | 0.0001 |
| mean | k10 | ccs_group | range | 0.0623 | 0.0001 |
| mean | k10 | ccs_group | weighted_sd | 0.0118 | 0.0001 |
| mean | k10 | long_k10 | range | 0.0255 | 0.0049 |
| mean | k10 | long_k10 | weighted_sd | 0.0058 | 0.0052 |

## 4. 격차(gap) 통계: Jaccard 범위 (long_k10, raw/조정)

| seed | variant | source | observed | p_bonferroni |
|---|---|---|---|---|
| 0 | baseline | raw | 0.0681 | 0.0035 |
| 0 | baseline | residual | 0.0520 | 0.0440 |
| 0 | ccs | raw | 0.0680 | 0.0045 |
| 0 | ccs | residual | 0.0536 | 0.0160 |
| 0 | k10 | raw | 0.0700 | 0.0030 |
| 0 | k10 | residual | 0.0551 | 0.0145 |
| 0 | global | raw | 0.0699 | 0.0030 |
| 0 | global | residual | 0.0503 | 0.0410 |
| 1 | baseline | raw | 0.0819 | 0.0005 |
| 1 | baseline | residual | 0.0553 | 0.0155 |
| 1 | ccs | raw | 0.0797 | 0.0005 |
| 1 | ccs | residual | 0.0594 | 0.0050 |
| 1 | k10 | raw | 0.0800 | 0.0005 |
| 1 | k10 | residual | 0.0599 | 0.0050 |
| 1 | global | raw | 0.0789 | 0.0005 |
| 1 | global | residual | 0.0584 | 0.0060 |
| 2 | baseline | raw | 0.0818 | 0.0005 |
| 2 | baseline | residual | 0.0683 | 0.0015 |
| 2 | ccs | raw | 0.0726 | 0.0020 |
| 2 | ccs | residual | 0.0622 | 0.0020 |
| 2 | k10 | raw | 0.0760 | 0.0015 |
| 2 | k10 | residual | 0.0656 | 0.0015 |
| 2 | global | raw | 0.0696 | 0.0030 |
| 2 | global | residual | 0.0598 | 0.0035 |
| 3 | baseline | raw | 0.0769 | 0.0015 |
| 3 | baseline | residual | 0.0672 | 0.0025 |
| 3 | ccs | raw | 0.0775 | 0.0015 |
| 3 | ccs | residual | 0.0670 | 0.0015 |
| 3 | k10 | raw | 0.0767 | 0.0015 |
| 3 | k10 | residual | 0.0657 | 0.0015 |
| 3 | global | raw | 0.0749 | 0.0015 |
| 3 | global | residual | 0.0615 | 0.0040 |
| mean | baseline | raw | 0.0772 | 0.0015 |
| mean | baseline | residual | 0.0607 | 0.0159 |
| mean | ccs | raw | 0.0744 | 0.0021 |
| mean | ccs | residual | 0.0605 | 0.0061 |
| mean | global | raw | 0.0733 | 0.0020 |
| mean | global | residual | 0.0575 | 0.0136 |
| mean | k10 | raw | 0.0757 | 0.0016 |
| mean | k10 | residual | 0.0616 | 0.0056 |

## 5. 군집별 표 (시드 평균)

| variant | axis | group | n_seeds | mean_ddi_pred | mean_ddi_true | mean_excess_ddi | min_excess_ddi | max_excess_ddi | adj_jaccard |
|---|---|---|---|---|---|---|---|---|---|
| baseline | ccs_group | 2ndary malig | 4 | 0.0617 | 0.0719 | -0.0103 | -0.0128 | -0.0088 | 0.5399 |
| baseline | ccs_group | Ac renl fail | 4 | 0.0649 | 0.0626 | 0.0023 | 0.0002 | 0.0057 | 0.4967 |
| baseline | ccs_group | Acute CVD | 4 | 0.0617 | 0.0591 | 0.0026 | -0.0002 | 0.0065 | 0.5228 |
| baseline | ccs_group | Acute MI | 4 | 0.0663 | 0.1205 | -0.0542 | -0.0601 | -0.0477 | 0.5210 |
| baseline | ccs_group | Adlt resp fl | 4 | 0.0687 | 0.1090 | -0.0403 | -0.0419 | -0.0380 | 0.4977 |
| baseline | ccs_group | Alcohol-related disorders | 4 | 0.0653 | 0.0747 | -0.0094 | -0.0208 | -0.0029 | 0.5482 |
| baseline | ccs_group | Aneurysm | 4 | 0.0614 | 0.0727 | -0.0112 | -0.0138 | -0.0093 | 0.5515 |
| baseline | ccs_group | Asp pneumon | 4 | 0.0691 | 0.0960 | -0.0269 | -0.0283 | -0.0259 | 0.5045 |
| baseline | ccs_group | Complic devi | 4 | 0.0610 | 0.0787 | -0.0177 | -0.0200 | -0.0152 | 0.5042 |
| baseline | ccs_group | Complic proc | 4 | 0.0629 | 0.0786 | -0.0157 | -0.0161 | -0.0147 | 0.4951 |
| baseline | ccs_group | Coron athero | 4 | 0.0585 | 0.0810 | -0.0225 | -0.0273 | -0.0185 | 0.6036 |
| baseline | ccs_group | DiabMel w/cm | 4 | 0.0631 | 0.0626 | 0.0005 | -0.0010 | 0.0034 | 0.5131 |
| baseline | ccs_group | Dysrhythmia | 4 | 0.0706 | 0.1118 | -0.0412 | -0.0457 | -0.0356 | 0.4808 |
| baseline | ccs_group | GI hemorrhag | 4 | 0.0644 | 0.0737 | -0.0093 | -0.0123 | -0.0071 | 0.4864 |
| baseline | ccs_group | Hrt valve dx | 4 | 0.0586 | 0.0699 | -0.0113 | -0.0124 | -0.0097 | 0.5776 |
| baseline | ccs_group | Intracrn inj | 4 | 0.0615 | 0.0613 | 0.0002 | -0.0029 | 0.0040 | 0.5151 |
| baseline | ccs_group | Oth liver dx | 4 | 0.0445 | 0.0420 | 0.0024 | -0.0017 | 0.0059 | 0.4913 |
| baseline | ccs_group | Pneumonia | 4 | 0.0683 | 0.0804 | -0.0121 | -0.0156 | -0.0097 | 0.5023 |
| baseline | ccs_group | Septicemia | 4 | 0.0603 | 0.0680 | -0.0077 | -0.0089 | -0.0066 | 0.5357 |
| baseline | ccs_group | chf;nonhp | 4 | 0.0661 | 0.0977 | -0.0316 | -0.0338 | -0.0271 | 0.4928 |
| baseline | ccs_group | 기타(소규모) | 4 | 0.0639 | 0.0771 | -0.0132 | -0.0144 | -0.0124 | 0.5012 |
| baseline | ccs_group | 미지정/매핑불가 | 4 | 0.0616 | 0.0874 | -0.0259 | -0.0289 | -0.0234 | 0.5088 |
| baseline | long_k10 | 0 | 4 | 0.0690 | 0.1037 | -0.0347 | -0.0378 | -0.0300 | 0.5480 |
| baseline | long_k10 | 1 | 4 | 0.0629 | 0.0796 | -0.0167 | -0.0200 | -0.0150 | 0.4876 |
| baseline | long_k10 | 2 | 4 | 0.0663 | 0.0862 | -0.0199 | -0.0212 | -0.0176 | 0.4972 |
| baseline | long_k10 | 3 | 4 | 0.0669 | 0.0804 | -0.0135 | -0.0150 | -0.0111 | 0.5118 |
| baseline | long_k10 | 4 | 4 | 0.0625 | 0.0756 | -0.0131 | -0.0142 | -0.0109 | 0.4943 |
| baseline | long_k10 | 5 | 4 | 0.0621 | 0.0742 | -0.0120 | -0.0123 | -0.0118 | 0.5068 |
| baseline | long_k10 | 6 | 4 | 0.0538 | 0.0648 | -0.0110 | -0.0126 | -0.0069 | 0.5031 |
| baseline | long_k10 | 7 | 4 | 0.0582 | 0.0628 | -0.0046 | -0.0068 | -0.0030 | 0.5543 |
| baseline | long_k10 | 8 | 4 | 0.0611 | 0.0708 | -0.0097 | -0.0130 | -0.0076 | 0.5393 |
| baseline | long_k10 | 9 | 4 | 0.0626 | 0.0771 | -0.0145 | -0.0155 | -0.0132 | 0.5323 |
| ccs | ccs_group | 2ndary malig | 4 | 0.0544 | 0.0719 | -0.0175 | -0.0183 | -0.0166 | 0.5367 |
| ccs | ccs_group | Ac renl fail | 4 | 0.0595 | 0.0626 | -0.0030 | -0.0051 | -0.0010 | 0.4962 |
| ccs | ccs_group | Acute CVD | 4 | 0.0574 | 0.0591 | -0.0017 | -0.0038 | 0.0013 | 0.5230 |
| ccs | ccs_group | Acute MI | 4 | 0.0660 | 0.1205 | -0.0545 | -0.0606 | -0.0500 | 0.5264 |
| ccs | ccs_group | Adlt resp fl | 4 | 0.0704 | 0.1090 | -0.0386 | -0.0410 | -0.0377 | 0.4998 |
| ccs | ccs_group | Alcohol-related disorders | 4 | 0.0518 | 0.0747 | -0.0229 | -0.0264 | -0.0205 | 0.5626 |
| ccs | ccs_group | Aneurysm | 4 | 0.0588 | 0.0727 | -0.0138 | -0.0159 | -0.0118 | 0.5485 |
| ccs | ccs_group | Asp pneumon | 4 | 0.0671 | 0.0960 | -0.0289 | -0.0305 | -0.0267 | 0.5046 |
| ccs | ccs_group | Complic devi | 4 | 0.0568 | 0.0787 | -0.0219 | -0.0230 | -0.0209 | 0.5047 |
| ccs | ccs_group | Complic proc | 4 | 0.0580 | 0.0786 | -0.0206 | -0.0212 | -0.0199 | 0.4976 |
| ccs | ccs_group | Coron athero | 4 | 0.0588 | 0.0810 | -0.0222 | -0.0265 | -0.0181 | 0.6130 |
| ccs | ccs_group | DiabMel w/cm | 4 | 0.0640 | 0.0626 | 0.0014 | -0.0002 | 0.0053 | 0.5185 |
| ccs | ccs_group | Dysrhythmia | 4 | 0.0681 | 0.1118 | -0.0437 | -0.0482 | -0.0415 | 0.4849 |
| ccs | ccs_group | GI hemorrhag | 4 | 0.0583 | 0.0737 | -0.0154 | -0.0165 | -0.0137 | 0.4910 |
| ccs | ccs_group | Hrt valve dx | 4 | 0.0534 | 0.0699 | -0.0165 | -0.0185 | -0.0142 | 0.5830 |
| ccs | ccs_group | Intracrn inj | 4 | 0.0514 | 0.0613 | -0.0099 | -0.0107 | -0.0093 | 0.5179 |
| ccs | ccs_group | Oth liver dx | 4 | 0.0439 | 0.0420 | 0.0018 | -0.0005 | 0.0037 | 0.4938 |
| ccs | ccs_group | Pneumonia | 4 | 0.0649 | 0.0804 | -0.0155 | -0.0181 | -0.0144 | 0.5127 |
| ccs | ccs_group | Septicemia | 4 | 0.0561 | 0.0680 | -0.0119 | -0.0124 | -0.0108 | 0.5326 |
| ccs | ccs_group | chf;nonhp | 4 | 0.0633 | 0.0977 | -0.0344 | -0.0364 | -0.0305 | 0.4970 |
| ccs | ccs_group | 기타(소규모) | 4 | 0.0600 | 0.0771 | -0.0171 | -0.0177 | -0.0166 | 0.5023 |
| ccs | ccs_group | 미지정/매핑불가 | 4 | 0.0612 | 0.0874 | -0.0263 | -0.0285 | -0.0237 | 0.5017 |
| ccs | long_k10 | 0 | 4 | 0.0632 | 0.1037 | -0.0406 | -0.0429 | -0.0384 | 0.5504 |
| ccs | long_k10 | 1 | 4 | 0.0594 | 0.0796 | -0.0202 | -0.0220 | -0.0193 | 0.4903 |
| ccs | long_k10 | 2 | 4 | 0.0625 | 0.0862 | -0.0237 | -0.0244 | -0.0232 | 0.5007 |
| ccs | long_k10 | 3 | 4 | 0.0612 | 0.0804 | -0.0192 | -0.0216 | -0.0182 | 0.5129 |
| ccs | long_k10 | 4 | 4 | 0.0605 | 0.0756 | -0.0151 | -0.0161 | -0.0138 | 0.4967 |
| ccs | long_k10 | 5 | 4 | 0.0589 | 0.0742 | -0.0152 | -0.0162 | -0.0143 | 0.5075 |
| ccs | long_k10 | 6 | 4 | 0.0492 | 0.0648 | -0.0156 | -0.0169 | -0.0150 | 0.5111 |
| ccs | long_k10 | 7 | 4 | 0.0537 | 0.0628 | -0.0091 | -0.0106 | -0.0075 | 0.5556 |
| ccs | long_k10 | 8 | 4 | 0.0561 | 0.0708 | -0.0146 | -0.0155 | -0.0139 | 0.5415 |
| ccs | long_k10 | 9 | 4 | 0.0592 | 0.0771 | -0.0178 | -0.0196 | -0.0163 | 0.5311 |
| global | ccs_group | 2ndary malig | 4 | 0.0606 | 0.0719 | -0.0113 | -0.0135 | -0.0087 | 0.5393 |
| global | ccs_group | Ac renl fail | 4 | 0.0618 | 0.0626 | -0.0007 | -0.0040 | 0.0011 | 0.4984 |
| global | ccs_group | Acute CVD | 4 | 0.0600 | 0.0591 | 0.0008 | -0.0016 | 0.0026 | 0.5228 |
| global | ccs_group | Acute MI | 4 | 0.0616 | 0.1205 | -0.0589 | -0.0610 | -0.0574 | 0.5202 |
| global | ccs_group | Adlt resp fl | 4 | 0.0652 | 0.1090 | -0.0437 | -0.0447 | -0.0432 | 0.4953 |
| global | ccs_group | Alcohol-related disorders | 4 | 0.0604 | 0.0747 | -0.0143 | -0.0182 | -0.0105 | 0.5627 |
| global | ccs_group | Aneurysm | 4 | 0.0607 | 0.0727 | -0.0119 | -0.0135 | -0.0101 | 0.5478 |
| global | ccs_group | Asp pneumon | 4 | 0.0651 | 0.0960 | -0.0309 | -0.0314 | -0.0301 | 0.5036 |
| global | ccs_group | Complic devi | 4 | 0.0588 | 0.0787 | -0.0199 | -0.0209 | -0.0193 | 0.5070 |
| global | ccs_group | Complic proc | 4 | 0.0587 | 0.0786 | -0.0198 | -0.0204 | -0.0191 | 0.4982 |
| global | ccs_group | Coron athero | 4 | 0.0601 | 0.0810 | -0.0209 | -0.0250 | -0.0181 | 0.6118 |
| global | ccs_group | DiabMel w/cm | 4 | 0.0636 | 0.0626 | 0.0010 | -0.0002 | 0.0036 | 0.5177 |
| global | ccs_group | Dysrhythmia | 4 | 0.0589 | 0.1118 | -0.0529 | -0.0594 | -0.0461 | 0.4794 |
| global | ccs_group | GI hemorrhag | 4 | 0.0549 | 0.0737 | -0.0188 | -0.0206 | -0.0174 | 0.4892 |
| global | ccs_group | Hrt valve dx | 4 | 0.0593 | 0.0699 | -0.0106 | -0.0134 | -0.0079 | 0.5824 |
| global | ccs_group | Intracrn inj | 4 | 0.0591 | 0.0613 | -0.0022 | -0.0056 | 0.0003 | 0.5192 |
| global | ccs_group | Oth liver dx | 4 | 0.0473 | 0.0420 | 0.0052 | 0.0015 | 0.0075 | 0.4963 |
| global | ccs_group | Pneumonia | 4 | 0.0638 | 0.0804 | -0.0166 | -0.0184 | -0.0160 | 0.5146 |
| global | ccs_group | Septicemia | 4 | 0.0588 | 0.0680 | -0.0092 | -0.0100 | -0.0072 | 0.5333 |
| global | ccs_group | chf;nonhp | 4 | 0.0608 | 0.0977 | -0.0369 | -0.0384 | -0.0350 | 0.4965 |
| global | ccs_group | 기타(소규모) | 4 | 0.0598 | 0.0771 | -0.0174 | -0.0179 | -0.0166 | 0.5027 |
| global | ccs_group | 미지정/매핑불가 | 4 | 0.0590 | 0.0874 | -0.0284 | -0.0312 | -0.0270 | 0.5030 |
| global | long_k10 | 0 | 4 | 0.0616 | 0.1037 | -0.0421 | -0.0440 | -0.0392 | 0.5471 |
| global | long_k10 | 1 | 4 | 0.0602 | 0.0796 | -0.0195 | -0.0204 | -0.0181 | 0.4902 |
| global | long_k10 | 2 | 4 | 0.0618 | 0.0862 | -0.0244 | -0.0253 | -0.0235 | 0.5003 |
| global | long_k10 | 3 | 4 | 0.0614 | 0.0804 | -0.0190 | -0.0220 | -0.0176 | 0.5135 |
| global | long_k10 | 4 | 4 | 0.0610 | 0.0756 | -0.0146 | -0.0160 | -0.0134 | 0.4972 |
| global | long_k10 | 5 | 4 | 0.0594 | 0.0742 | -0.0148 | -0.0155 | -0.0141 | 0.5078 |
| global | long_k10 | 6 | 4 | 0.0522 | 0.0648 | -0.0126 | -0.0143 | -0.0115 | 0.5114 |
| global | long_k10 | 7 | 4 | 0.0554 | 0.0628 | -0.0075 | -0.0093 | -0.0066 | 0.5555 |
| global | long_k10 | 8 | 4 | 0.0578 | 0.0708 | -0.0130 | -0.0148 | -0.0109 | 0.5426 |
| global | long_k10 | 9 | 4 | 0.0596 | 0.0771 | -0.0174 | -0.0193 | -0.0161 | 0.5311 |
| k10 | ccs_group | 2ndary malig | 4 | 0.0569 | 0.0719 | -0.0150 | -0.0160 | -0.0145 | 0.5404 |
| k10 | ccs_group | Ac renl fail | 4 | 0.0612 | 0.0626 | -0.0014 | -0.0030 | 0.0001 | 0.4963 |
| k10 | ccs_group | Acute CVD | 4 | 0.0599 | 0.0591 | 0.0007 | -0.0017 | 0.0019 | 0.5237 |
| k10 | ccs_group | Acute MI | 4 | 0.0636 | 0.1205 | -0.0568 | -0.0590 | -0.0548 | 0.5228 |
| k10 | ccs_group | Adlt resp fl | 4 | 0.0651 | 0.1090 | -0.0438 | -0.0444 | -0.0426 | 0.4958 |
| k10 | ccs_group | Alcohol-related disorders | 4 | 0.0545 | 0.0747 | -0.0203 | -0.0236 | -0.0165 | 0.5618 |
| k10 | ccs_group | Aneurysm | 4 | 0.0595 | 0.0727 | -0.0131 | -0.0155 | -0.0101 | 0.5454 |
| k10 | ccs_group | Asp pneumon | 4 | 0.0644 | 0.0960 | -0.0316 | -0.0327 | -0.0299 | 0.5042 |
| k10 | ccs_group | Complic devi | 4 | 0.0583 | 0.0787 | -0.0204 | -0.0210 | -0.0197 | 0.5059 |
| k10 | ccs_group | Complic proc | 4 | 0.0581 | 0.0786 | -0.0205 | -0.0216 | -0.0196 | 0.4979 |
| k10 | ccs_group | Coron athero | 4 | 0.0603 | 0.0810 | -0.0207 | -0.0250 | -0.0181 | 0.6131 |
| k10 | ccs_group | DiabMel w/cm | 4 | 0.0634 | 0.0626 | 0.0008 | -0.0013 | 0.0033 | 0.5192 |
| k10 | ccs_group | Dysrhythmia | 4 | 0.0638 | 0.1118 | -0.0480 | -0.0507 | -0.0457 | 0.4841 |
| k10 | ccs_group | GI hemorrhag | 4 | 0.0559 | 0.0737 | -0.0178 | -0.0192 | -0.0168 | 0.4897 |
| k10 | ccs_group | Hrt valve dx | 4 | 0.0589 | 0.0699 | -0.0111 | -0.0146 | -0.0087 | 0.5830 |
| k10 | ccs_group | Intracrn inj | 4 | 0.0571 | 0.0613 | -0.0042 | -0.0067 | -0.0017 | 0.5196 |
| k10 | ccs_group | Oth liver dx | 4 | 0.0471 | 0.0420 | 0.0051 | 0.0015 | 0.0072 | 0.4963 |
| k10 | ccs_group | Pneumonia | 4 | 0.0636 | 0.0804 | -0.0168 | -0.0171 | -0.0159 | 0.5132 |
| k10 | ccs_group | Septicemia | 4 | 0.0587 | 0.0680 | -0.0093 | -0.0101 | -0.0076 | 0.5334 |
| k10 | ccs_group | chf;nonhp | 4 | 0.0619 | 0.0977 | -0.0359 | -0.0380 | -0.0339 | 0.4966 |
| k10 | ccs_group | 기타(소규모) | 4 | 0.0593 | 0.0771 | -0.0178 | -0.0183 | -0.0173 | 0.5028 |
| k10 | ccs_group | 미지정/매핑불가 | 4 | 0.0598 | 0.0874 | -0.0276 | -0.0317 | -0.0256 | 0.5013 |
| k10 | long_k10 | 0 | 4 | 0.0670 | 0.1037 | -0.0367 | -0.0400 | -0.0350 | 0.5511 |
| k10 | long_k10 | 1 | 4 | 0.0595 | 0.0796 | -0.0201 | -0.0210 | -0.0190 | 0.4899 |
| k10 | long_k10 | 2 | 4 | 0.0645 | 0.0862 | -0.0217 | -0.0233 | -0.0203 | 0.5013 |
| k10 | long_k10 | 3 | 4 | 0.0659 | 0.0804 | -0.0146 | -0.0161 | -0.0132 | 0.5145 |
| k10 | long_k10 | 4 | 4 | 0.0597 | 0.0756 | -0.0160 | -0.0169 | -0.0149 | 0.4965 |
| k10 | long_k10 | 5 | 4 | 0.0574 | 0.0742 | -0.0168 | -0.0177 | -0.0161 | 0.5069 |
| k10 | long_k10 | 6 | 4 | 0.0470 | 0.0648 | -0.0179 | -0.0188 | -0.0168 | 0.5119 |
| k10 | long_k10 | 7 | 4 | 0.0516 | 0.0628 | -0.0112 | -0.0124 | -0.0101 | 0.5563 |
| k10 | long_k10 | 8 | 4 | 0.0531 | 0.0708 | -0.0177 | -0.0186 | -0.0165 | 0.5437 |
| k10 | long_k10 | 9 | 4 | 0.0584 | 0.0771 | -0.0187 | -0.0203 | -0.0176 | 0.5305 |

## 6. 해석

- long_k10 축, ccs 변형: excess-DDI 범위가 baseline 대비 좁아진 시드 수 2/4.
- long_k10 축, k10 변형: excess-DDI 범위가 baseline 대비 좁아진 시드 수 4/4.
- ccs_group 축, ccs 변형: excess-DDI 범위가 baseline 대비 좁아진 시드 수 3/4.
- ccs_group 축, k10 변형: excess-DDI 범위가 baseline 대비 좁아진 시드 수 1/4.

- global 대조군과 ccs/k10의 격차 축소 정도를 비교하면, 예산의 크기 자체(하나의 숫자)가 아니라 그룹별로 예산을 다르게 준 것이 격차 축소에 기여했는지를 판단할 수 있다.
- 위 표는 test 지표, excess-DDI 및 Jaccard 격차 통계, 군집별 시드 평균을 보여준다. 해석은 이 표들이 보여주는 범위를 넘지 않는다.