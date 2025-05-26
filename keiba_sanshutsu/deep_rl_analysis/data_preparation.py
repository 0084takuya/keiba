import pandas as pd
import pymysql
from keiba_sanshutsu.config.db_config import DB_CONFIG
from datetime import datetime, timedelta

# 5年分のデータをDBから取得
N_YEARS = 5
today = datetime.today()
start_date = today - timedelta(days=365*N_YEARS)
start_date_str = start_date.strftime('%Y%m%d')

conn = pymysql.connect(**DB_CONFIG)

# レース詳細・馬・騎手・調教師・血統などをJOIN
query = f'''
SELECT r.*, s.KAISAI_NEN, s.KAISAI_GAPPI, s.KEIBAJO_CODE, s.KAISAI_KAI, s.KAISAI_NICHIME, s.RACE_BANGO, s.KYORI,
       h.BAMEI, h.SEIBETSU_CODE, h.CHOKYOSHI_CODE as UMA_CHOKYOSHI_CODE, h.BANUSHI_CODE,
       j.KISHUMEI_RYAKUSHO, t.CHOKYOSHIMEI_RYAKUSHO, p1.KEITO_MEI as KETTO1_KEITO_MEI
FROM umagoto_race_joho r
LEFT JOIN race_shosai s ON r.RACE_CODE = s.RACE_CODE
LEFT JOIN kyosoba_master2 h ON r.KETTO_TOROKU_BANGO = h.KETTO_TOROKU_BANGO
LEFT JOIN kishu_master j ON r.KISHU_CODE = j.KISHU_CODE
LEFT JOIN chokyoshi_master t ON r.CHOKYOSHI_CODE = t.CHOKYOSHI_CODE
LEFT JOIN keito_joho2 p1 ON h.KETTO1_HANSHOKU_TOROKU_BANGO = p1.KEITO_ID
WHERE CONCAT(s.KAISAI_NEN, LPAD(s.KAISAI_GAPPI, 4, '0')) >= {start_date_str}
'''
df = pd.read_sql(query, conn)
conn.close()

# 前処理例（必要に応じて拡張）
# 欠損値処理
for col in ['BATAIJU', 'ZOGEN_SA']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
# 馬体重変化量
if 'BATAIJU' in df.columns and 'ZOGEN_SA' in df.columns:
    df['BATAIJU_DIFF'] = df['ZOGEN_SA']
# カテゴリ変換例
if 'KEIBAJO_CODE' in df.columns:
    df['KEIBAJO_CODE'] = df['KEIBAJO_CODE'].astype(str)
# ...（他の特徴量エンジニアリングもここに追加）

# --- 特徴量選択 ---
features = [
    'WAKUBAN',           # 枠番
    'BATAIJU',           # 馬体重
    'BATAIJU_DIFF',      # 馬体重変化量
    'KYORI',             # 距離
    'KEIBAJO_CODE',      # 競馬場
    'SEIBETSU_CODE',     # 性別
    'BAREI',             # 馬齢
    'CHOKYOSHI_CODE',    # 調教師
    'KISHU_CODE',        # 騎手
    'KETTO1_KEITO_MEI',  # 血統名カテゴリ
    'BANUSHI_CODE',      # 馬主
    'RACE_BANGO',        # レース番号
    'KAISAI_KAI',        # 開催回
    'KAISAI_NICHIME',    # 日目
    # 必要に応じて追加（最大30項目まで）
]
# カラム名の前後空白・改行を除去し、ユニーク化
features = list(dict.fromkeys([col.strip() for col in features]))
df.columns = [col.strip() for col in df.columns]

# --- object型カラムを数値変換 ---
for col in features:
    print(f'col={col}, type={type(df[col])}')
    if col in df.columns and str(df[col].dtype) == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- カテゴリ変数のエンコーディング ---
for col in ['KEIBAJO_CODE', 'SEIBETSU_CODE', 'CHOKYOSHI_CODE', 'KISHU_CODE', 'BANUSHI_CODE', 'KETTO1_KEITO_MEI']:
    if col in df.columns and isinstance(df[col], pd.Series):
        df[col] = df[col].astype('category').cat.codes

# --- ラベル作成（3位以内=1, それ以外=0） ---
df['KAKUTEI_CHAKUJUN'] = pd.to_numeric(df['KAKUTEI_CHAKUJUN'], errors='coerce')
df['WIN3'] = (df['KAKUTEI_CHAKUJUN'] <= 3).astype(int)

# --- データセット作成 ---
X = df[features].fillna(0)
y = df['WIN3']

# --- 保存 ---
X.to_pickle('keiba_sanshutsu/deep_rl_analysis/X.pkl')
y.to_pickle('keiba_sanshutsu/deep_rl_analysis/y.pkl')

print('特徴量サンプル:')
print(X.head())
print('ラベル分布:')
print(y.value_counts())

print(df.head())
print(f"データ件数: {len(df)}") 