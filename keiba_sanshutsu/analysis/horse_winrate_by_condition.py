import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pymysql
from keiba_sanshutsu.config.db_config import DB_CONFIG
import numpy as np
import math
import re

# 日本語フォント設定（Macの場合）
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# race.csvを読み込み
race_csv = 'keiba_sanshutsu/data_preprocessing/race.csv'
df_race = pd.read_csv(race_csv)

# race_shosaiからKYORIとレース詳細をJOIN
conn = pymysql.connect(**DB_CONFIG)
df_shosai = pd.read_sql("SELECT RACE_CODE, KAISAI_NEN, KAISAI_GAPPI, KEIBAJO_CODE, KAISAI_KAI, KAISAI_NICHIME, RACE_BANGO, KYORI FROM race_shosai", conn)
df_keibajo = pd.read_sql("SELECT CODE, JOMEI FROM keibajo_code", conn)
conn.close()
df_race = pd.merge(df_race, df_shosai, on='RACE_CODE', how='left')
df_race['KYORI'] = pd.to_numeric(df_race['KYORI'], errors='coerce')
df_race['KYORI_CAT'] = pd.cut(df_race['KYORI'], bins=[0, 1200, 2000, 4000], labels=['短距離', '中距離', '長距離'])

# RACE_CODEごとの出走頭数を集計
# race_counts = df_race.groupby('RACE_CODE')['BAMEI'].count().sort_values(ascending=False)
# target_race_code = race_counts.index[0]
# 2025年3月1日（KAISAI_NEN=2025, KAISAI_GAPPI=301）の最初のRACE_CODEを選択
march1_df = df_race[(df_race['KAISAI_NEN_x'] == 2025) & (df_race['KAISAI_GAPPI_x'] == 301)]
target_race_code = march1_df['RACE_CODE'].iloc[0]
target_df = df_race[df_race['RACE_CODE'] == target_race_code]
print('target_df columns:', target_df.columns.tolist())

# レース詳細情報をshosaiから直接取得
# 型を揃える
race_code_type = df_shosai['RACE_CODE'].dtype
if race_code_type == 'O':
    target_race_code_cast = str(target_race_code)
else:
    target_race_code_cast = int(target_race_code)
shosai_row = df_shosai[df_shosai['RACE_CODE'] == target_race_code_cast]
if shosai_row.empty:
    print(f"[ERROR] df_shosaiにRACE_CODE={target_race_code_cast}が存在しません")
    print(f"df_shosai['RACE_CODE']の型: {df_shosai['RACE_CODE'].dtype}, target_race_codeの型: {type(target_race_code_cast)}")
    print(f"df_shosai['RACE_CODE']のユニーク例: {df_shosai['RACE_CODE'].unique()[:10]}")
    sys.exit(1)
race_info_row = shosai_row.iloc[0]
kaisai_nen = str(race_info_row['KAISAI_NEN'])
kaisai_gappi = str(race_info_row['KAISAI_GAPPI'])
kaisai_gappi_fmt = f"{int(kaisai_gappi[:2])}月{int(kaisai_gappi[2:])}日" if len(kaisai_gappi) == 4 else kaisai_gappi
keibajo_code = str(race_info_row['KEIBAJO_CODE']).zfill(2)
keibajo_name = df_keibajo[df_keibajo['CODE'] == keibajo_code]['JOMEI'].values[0] if keibajo_code in df_keibajo['CODE'].values else keibajo_code
kai = f"{int(race_info_row['KAISAI_KAI'])}回"
nichime = f"{int(race_info_row['KAISAI_NICHIME'])}日目"
race_no = f"{int(race_info_row['RACE_BANGO'])}R"

title_str = f"{kaisai_nen}年{kaisai_gappi_fmt} {keibajo_name} {kai}{nichime} {race_no}\nRACE_CODE: {target_race_code} 出走馬ごとの過去2年複勝率（3位以内）"

# 当該レースの馬名と着順の辞書を作成
chaku_dict = dict(zip(target_df['BAMEI'], target_df['KAKUTEI_CHAKUJUN']))

# 各馬の過去2年分の全レースから複勝率（3位以内率）を算出
result = []
for name in target_df['BAMEI']:
    horse_df = df_race[df_race['BAMEI'] == name]
    total_race = len(horse_df)
    if total_race == 0:
        fukusho_rate = 0
    else:
        fukusho = (horse_df['KAKUTEI_CHAKUJUN'] <= 3).sum()
        fukusho_rate = fukusho / total_race
    chaku = chaku_dict.get(name, None)
    # 取消判定
    if pd.isna(chaku) or chaku == 0:
        label = f"{name}（取消）"
    else:
        label = f"{name}（{int(chaku)}）"
    result.append({'BAMEI': name, '複勝率': fukusho_rate, '出走数': total_race, 'LABEL': label})

df_result = pd.DataFrame(result)
# 複勝率で降順ソート
plot_df = df_result.sort_values('複勝率', ascending=False).reset_index(drop=True)
print(plot_df)

# 棒グラフで可視化
plt.figure(figsize=(12, 6))
plt.bar(plot_df['LABEL'], plot_df['複勝率'], color='skyblue')
plt.title(title_str)
plt.ylabel('複勝率')
plt.xlabel('馬名（当該レース着順/取消）')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(plot_df['複勝率']):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

# 保存用ディレクトリとファイル名生成
save_dir = os.path.join(os.path.dirname(__file__), '../result_images')
os.makedirs(save_dir, exist_ok=True)
# レース名（例: 2024年5月12日東京1回6日目11R）を生成
race_name = f"{kaisai_nen}年{kaisai_gappi_fmt}{keibajo_name}{kai}{nichime}{race_no}"
# ファイル名に使えない文字を除去
race_name_safe = re.sub(r'[\\/:*?"<>|\s]', '', race_name)
content_name = '複勝3位以内'
filename = f"{race_name_safe}_{content_name}.png"
save_path = os.path.join(save_dir, filename)

plt.tight_layout()
plt.savefig(save_path)
plt.show()
print(f"グラフ画像を保存しました: {save_path}") 