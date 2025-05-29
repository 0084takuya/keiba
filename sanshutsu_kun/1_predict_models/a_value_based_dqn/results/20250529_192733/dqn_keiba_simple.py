import os
import pymysql
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
from dotenv import load_dotenv
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import base64
from modules.plot_utils import plot_and_save_graphs, save_script_and_make_run_dir
from modules.gpt_utils import ask_gpt41
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, List
import torch.nn.functional as F
import lightgbm as lgb
from sklearn.model_selection import KFold
from matplotlib import font_manager
from modules.progress_utils import print_progress_bar
from imblearn.over_sampling import SMOTE
from modules.env_utils import load_env_and_set_api
from modules.db_utils import get_db_connection
import sys
from modules.cli_utils import validate_pythonpath, parse_days_arg

print('=== DQNモデル実行開始 ===')

# --- .envからAPIキーをロード ---
load_env_and_set_api()

# --- DB接続設定（必要に応じて修正） ---
# DB_HOST = os.getenv('KEIBA_DB_HOST', 'localhost')
# DB_USER = os.getenv('KEIBA_DB_USER', 'root')
# DB_PASS = os.getenv('KEIBA_DB_PASS', '')
# DB_NAME = os.getenv('KEIBA_DB_NAME', 'mykeibadb')
# DB_PORT = int(os.getenv('KEIBA_DB_PORT', 3306))

@dataclass
class Feature:
    name: str
    db_column: Optional[str] = None
    description: str = ""
    compute_func: Optional[Callable] = None
    is_categorical: bool = False
    depends_on: Optional[str] = None  # 参照する辞書やSQL名

# 特徴量定義
FEATURES: List[Feature] = [
    Feature("BAREI", "BAREI", "馬齢"),
    Feature("SEIBETSU_CODE", "SEIBETSU_CODE", "性別コード", is_categorical=True),
    Feature("BATAIJU", "BATAIJU", "馬体重"),
    Feature("BATAIJU_DIFF", None, "直近体重変化"),
    Feature("FUTAN_JURYO", "FUTAN_JURYO", "負担重量"),
    Feature("WAKUBAN", "WAKUBAN", "枠順", is_categorical=True),
    Feature("KYORI", None, "距離", depends_on="race_shosai"),
    Feature("TRACK_CODE", None, "芝/ダート（トラックコード）", is_categorical=True, depends_on="race_shosai"),
    Feature("SHUSSO_TOSU", None, "頭数", depends_on="race_shosai"),
    Feature("KISHU_3IN_RATE", None, "騎手3着以内率", depends_on="kishu_stats"),
    Feature("TENKO_CODE", None, "馬場状態", is_categorical=True, depends_on="race_tenko"),
    Feature("CHOKYOSHI_3IN_RATE", None, "調教師3着以内率", depends_on="chokyoshi_stats"),
    Feature("FATHER_LINEAGE", None, "父系統ID", is_categorical=True, depends_on="uma_father"),
    Feature("BATAIJU_KYORI", None, "馬体重×距離"),
    Feature("BAREI_FUTAN_JURYO", None, "馬齢×負担重量"),
    Feature("RACE_KANKAKU", None, "レース間隔（日数）"),
    Feature("TENKO_FINE", None, "馬場状態（良=1, それ以外=0"),
    Feature("ZENSHO_ICHIAKUMA_SA", None, "前走一着馬との秒数差"),
    Feature("ZENSHO_GYAKUJUN", None, "前走逆順順位"),
]

target_days = None  # コマンドラインで指定する

# --- 特徴量とラベルの抽出 ---
def fetch_data():
    from collections import defaultdict  # 必ずfetch_data内でimport
    print('=== fetch_data関数 実行開始 ===')
    start_time = time.time()
    conn = get_db_connection()
    cursor = conn.cursor()
    # 直近N年分の日付を計算
    today = datetime.today()
    date_min = (today - timedelta(days=target_days)).strftime('%Y%m%d')

    # 1. 騎手ごとの3着以内率を計算
    print('=== 騎手ごとの3着以内率を計算 ===')
    cursor.execute('''
        SELECT KISHU_CODE, COUNT(*), SUM(CASE WHEN KAKUTEI_CHAKUJUN IN ('1','2','3') THEN 1 ELSE 0 END)
        FROM umagoto_race_joho
        WHERE KISHU_CODE IS NOT NULL AND KAKUTEI_CHAKUJUN IS NOT NULL
        GROUP BY KISHU_CODE
    ''')
    kishu_stats = {row[0]: (row[2]/row[1] if row[1]>0 else 0) for row in cursor.fetchall()}

    # 2. 調教師ごとの3着以内率
    print('=== 調教師ごとの3着以内率を計算 ===')
    cursor.execute('''
        SELECT CHOKYOSHI_CODE, COUNT(*), SUM(CASE WHEN KAKUTEI_CHAKUJUN IN ('1','2','3') THEN 1 ELSE 0 END)
        FROM umagoto_race_joho
        WHERE CHOKYOSHI_CODE IS NOT NULL AND KAKUTEI_CHAKUJUN IS NOT NULL
        GROUP BY CHOKYOSHI_CODE
    ''')
    chokyoshi_stats = {row[0]: (row[2]/row[1] if row[1]>0 else 0) for row in cursor.fetchall()}

    # 3. 馬場状態（TENKO_CODE）をRACE_CODEで取得
    print('=== 馬場状態（TENKO_CODE）をRACE_CODEで取得 ===')
    cursor.execute('''
        SELECT RACE_CODE, TENKO_CODE FROM race_shosai WHERE TENKO_CODE IS NOT NULL
    ''')
    race_tenko = {row[0]: row[1] for row in cursor.fetchall()}
    tenko_labels = {code: i for i, code in enumerate(sorted(set(race_tenko.values())))}

    # 4. 父系統IDをUMAごとに取得
    print('=== 父系統IDをUMAごとに取得 ===')
    cursor.execute('''
        SELECT KETTO_TOROKU_BANGO, KETTO1_HANSHOKU_TOROKU_BANGO FROM kyosoba_master2
    ''')
    uma_father = {row[0]: row[1] for row in cursor.fetchall()}
    father_ids = list(set(uma_father.values()))
    father_label = {code: i for i, code in enumerate(father_ids)}

    # 5. メインデータ取得
    print('=== メインデータ取得 ===')
    db_columns = [f.db_column for f in FEATURES if f.db_column]
    # KAKUTEI_CHAKUJUNを必ず含める
    if 'KAKUTEI_CHAKUJUN' not in db_columns:
        db_columns.append('KAKUTEI_CHAKUJUN')
    query = f'''
        SELECT {', '.join(db_columns)}, KISHU_CODE, RACE_CODE, KETTO_TOROKU_BANGO, CHOKYOSHI_CODE, KEIBAJO_CODE, KAISAI_NEN, KAISAI_GAPPI
        FROM umagoto_race_joho
        WHERE CONCAT(KAISAI_NEN, LPAD(KAISAI_GAPPI, 4, '0')) >= %s
        AND {' AND '.join([f'{col} IS NOT NULL' for col in db_columns if col != 'KAKUTEI_CHAKUJUN'])}
        AND KISHU_CODE IS NOT NULL AND RACE_CODE IS NOT NULL AND KETTO_TOROKU_BANGO IS NOT NULL AND CHOKYOSHI_CODE IS NOT NULL AND KEIBAJO_CODE IS NOT NULL
    '''
    cursor.execute(query, (date_min,))
    rows = cursor.fetchall()
    print(f'抽出件数: {len(rows)}')
    # race_shosaiからKYORI, TRACK_CODE, SHUSSO_TOSUを全件取得（ここで定義）
    print('=== race_shosaiからKYORI, TRACK_CODE, SHUSSO_TOSUを全件取得 ===')
    cursor.execute('SELECT RACE_CODE, KYORI, TRACK_CODE, SHUSSO_TOSU FROM race_shosai')
    race_shosai_df = pd.DataFrame(cursor.fetchall(), columns=['RACE_CODE','KYORI','TRACK_CODE','SHUSSO_TOSU'])
    for col in ['KYORI','TRACK_CODE','SHUSSO_TOSU']:
        race_shosai_df[col] = pd.to_numeric(race_shosai_df[col], errors='coerce').fillna(0)
    df = pd.DataFrame(rows, columns=db_columns + ['KISHU_CODE','RACE_CODE','KETTO_TOROKU_BANGO','CHOKYOSHI_CODE','KEIBAJO_CODE','KAISAI_NEN','KAISAI_GAPPI'])
    # レース日付列を作成
    df['RACE_DATE'] = df['KAISAI_NEN'].astype(str) + df['KAISAI_GAPPI'].astype(str).str.zfill(4)
    df['RACE_DATE'] = pd.to_datetime(df['RACE_DATE'], format='%Y%m%d', errors='coerce')
    # 馬ごとに日付ソートし、前走との日数差分を計算
    df = df.sort_values(['KETTO_TOROKU_BANGO', 'RACE_DATE'])
    df['RACE_KANKAKU'] = df.groupby('KETTO_TOROKU_BANGO')['RACE_DATE'].diff().dt.days.fillna(0)
    # 6. 馬の過去5走3着以内率を計算
    print('=== 馬の過去5走3着以内率を計算 ===')
    uma_race = defaultdict(list)
    uma_course = defaultdict(list)
    uma_course_5r = defaultdict(list)
    cursor.execute('''
        SELECT KETTO_TOROKU_BANGO, RACE_CODE, KAKUTEI_CHAKUJUN
        FROM umagoto_race_joho
        WHERE KETTO_TOROKU_BANGO IS NOT NULL AND KAKUTEI_CHAKUJUN IS NOT NULL
        ORDER BY KETTO_TOROKU_BANGO, RACE_CODE
    ''')
    for row in cursor.fetchall():
        uma_race[row[0]].append((row[1], row[2]))

    # 7. コース適性（KEIBAJO_CODE, 距離帯ごとの3着以内率）
    print('=== コース適性（KEIBAJO_CODE, 距離帯ごとの3着以内率）を計算 ===')
    for row in cursor.fetchall():
        try:
            kyori = int(row[2])
            if kyori < 1400:
                kyori_bin = 0
            elif kyori < 2000:
                kyori_bin = 1
            else:
                kyori_bin = 2
        except:
            kyori_bin = 1
        uma_course[(row[0], row[1], kyori_bin)].append(int(row[3]))

    # 8. コース×馬過去5走3着以内率を計算
    print('=== コース×馬過去5走3着以内率を計算 ===')
    cursor.execute('''
        SELECT u.KETTO_TOROKU_BANGO, u.KEIBAJO_CODE, s.KYORI, u.KAKUTEI_CHAKUJUN
        FROM umagoto_race_joho u
        JOIN race_shosai s ON u.RACE_CODE = s.RACE_CODE
        WHERE u.KETTO_TOROKU_BANGO IS NOT NULL AND u.KEIBAJO_CODE IS NOT NULL AND s.KYORI IS NOT NULL AND u.KAKUTEI_CHAKUJUN IS NOT NULL
    ''')
    for row in cursor.fetchall():
        try:
            kyori = int(row[2])
            if kyori < 1400:
                kyori_bin = 0
            elif kyori < 2000:
                kyori_bin = 1
            else:
                kyori_bin = 2
        except:
            kyori_bin = 1
        uma_course_5r[(row[0], row[1], kyori_bin)].append(int(row[3]))

    # TE用辞書の初期化と計算
    print('=== TE用辞書初期化、計算 ===')
    father_te = {}
    track_te = {}
    father_count = {}
    track_count = {}
    kishu_chokyoshi_te = {}
    kishu_chokyoshi_count = {}
    uma_course_te, uma_course_count = {}, {}
    for row in rows:
        idx = 0
        for f in FEATURES:
            if f.db_column:
                idx += 1
        kishu_code = row[idx]
        race_code = row[idx+1]
        ketto_toroku_bango = row[idx+2]
        chokyoshi_code = row[idx+3]
        keibajo_code = row[idx+4]
        father_id = uma_father.get(ketto_toroku_bango, None)
        race_shosai_row = race_shosai_df.loc[race_shosai_df['RACE_CODE'] == race_code, ['KYORI', 'TRACK_CODE', 'SHUSSO_TOSU']].iloc[0].to_dict()
        track_code = race_shosai_row['TRACK_CODE'] if track_code in race_shosai_row else 0
        chaku = int(row.iloc[4]) if len(row) > 4 else 0
        label = 1 if chaku in [1,2,3] else 0
        # FATHER_LINEAGE TE
        if father_id not in father_te:
            father_te[father_id] = 0
            father_count[father_id] = 0
        father_te[father_id] += label
        father_count[father_id] += 1
        # TRACK_CODE TE
        if track_code not in track_te:
            track_te[track_code] = 0
            track_count[track_code] = 0
        track_te[track_code] += label
        track_count[track_code] += 1
        # 騎手×調教師 TE
        kc_key = (kishu_code, chokyoshi_code)
        if kc_key not in kishu_chokyoshi_te:
            kishu_chokyoshi_te[kc_key] = 0
            kishu_chokyoshi_count[kc_key] = 0
        kishu_chokyoshi_te[kc_key] += label
        kishu_chokyoshi_count[kc_key] += 1
        # 馬×コース TE
        uma_course_key = (ketto_toroku_bango, keibajo_code)
        if uma_course_key not in uma_course_te:
            uma_course_te[uma_course_key] = 0
            uma_course_count[uma_course_key] = 0
        uma_course_te[uma_course_key] += label
        uma_course_count[uma_course_key] += 1
    for k in father_te:
        father_te[k] = father_te[k] / father_count[k] if father_count[k] > 0 else 0
    for k in track_te:
        track_te[k] = track_te[k] / track_count[k] if track_count[k] > 0 else 0
    for k in kishu_chokyoshi_te:
        kishu_chokyoshi_te[k] = kishu_chokyoshi_te[k] / kishu_chokyoshi_count[k] if kishu_chokyoshi_count[k] > 0 else 0
    for k in uma_course_te:
        uma_course_te[k] = uma_course_te[k] / uma_course_count[k] if uma_course_count[k] > 0 else 0

    # --- データ作成 ---
    print('=== データ作成 ===')
    X, y = [], []
    for i, row in df.iterrows():
        try:
            feature_dict = {}
            for f in FEATURES:
                if f.db_column:
                    val = row[f.db_column]
                    feature_dict[f.name] = float(val) if val is not None and str(val).replace('.', '', 1).isdigit() else 0
            kishu_code = row['KISHU_CODE']
            race_code = row['RACE_CODE']
            ketto_toroku_bango = row['KETTO_TOROKU_BANGO']
            chokyoshi_code = row['CHOKYOSHI_CODE']
            keibajo_code = row['KEIBAJO_CODE']
            race_shosai_row = race_shosai_df.loc[race_shosai_df['RACE_CODE'] == race_code, ['KYORI', 'TRACK_CODE', 'SHUSSO_TOSU']].iloc[0].to_dict()
            feature_dict.update(race_shosai_row)
            feature_dict['KISHU_3IN_RATE'] = kishu_stats.get(kishu_code, 0)
            tenko_code = race_tenko.get(race_code, None)
            feature_dict['TENKO_CODE'] = tenko_labels[tenko_code] if tenko_code in tenko_labels else 0
            feature_dict['CHOKYOSHI_3IN_RATE'] = chokyoshi_stats.get(chokyoshi_code, 0)
            father_id = uma_father.get(ketto_toroku_bango, None)
            feature_dict['FATHER_LINEAGE'] = father_label[father_id] if father_id in father_label else 0
            past = uma_race.get(ketto_toroku_bango, [])
            past5 = [int(x[1]) for x in past[-6:-1]] if len(past) >= 6 else [int(x[1]) for x in past[:-1]]
            feature_dict['UMA_5R_3IN_RATE'] = sum([1 for c in past5 if c in [1,2,3]]) / len(past5) if past5 else 0
            try:
                kyori = float(feature_dict['KYORI'])
                if kyori < 1400:
                    kyori_bin = 0
                elif kyori < 2000:
                    kyori_bin = 1
                else:
                    kyori_bin = 2
            except:
                kyori_bin = 1
            course_key = (ketto_toroku_bango, keibajo_code, kyori_bin)
            course_results = uma_course.get(course_key, [])
            feature_dict['COURSE_3IN_RATE'] = sum([1 for c in course_results if c in [1,2,3]]) / len(course_results) if course_results else 0
            course_uma5r = uma_course_5r.get(course_key, [])
            course_uma5r5 = course_uma5r[-5:] if len(course_uma5r) >= 5 else course_uma5r
            feature_dict['COURSE_UMA_5R_3IN_RATE'] = sum([1 for c in course_uma5r5 if c in [1,2,3]]) / len(course_uma5r5) if course_uma5r5 else 0
            # --- 追加特徴量 ---
            feature_dict['FATHER_LINEAGE_TE'] = father_te.get(father_id, 0)
            track_code = feature_dict['TRACK_CODE']
            feature_dict['TRACK_CODE_TE'] = track_te.get(track_code, 0)
            feature_dict['BATAIJU_KYORI'] = feature_dict['BATAIJU'] * feature_dict['KYORI']
            feature_dict['BAREI_FUTAN_JURYO'] = feature_dict['BAREI'] * feature_dict['FUTAN_JURYO']
            kc_key = (kishu_code, chokyoshi_code)
            feature_dict['KISHU_CHOKYOSHI_TE'] = kishu_chokyoshi_te.get(kc_key, 0)
            uma_course_key = (ketto_toroku_bango, keibajo_code)
            feature_dict['UMA_COURSE_TE'] = uma_course_te.get(uma_course_key, 0)
            # RACE_KANKAKU（日数ギャップ）
            feature_dict['RACE_KANKAKU'] = row['RACE_KANKAKU'] if not pd.isnull(row['RACE_KANKAKU']) else 0
            # 直近体重変化
            if i > 0 and row['KETTO_TOROKU_BANGO'] == df.iloc[i-1]['KETTO_TOROKU_BANGO']:
                bataiju_diff = row['BATAIJU'] - df.iloc[i-1]['BATAIJU']
            else:
                bataiju_diff = 0
            feature_dict['BATAIJU_DIFF'] = bataiju_diff
            # --- 追加: 前走一着馬との秒数差・前走逆順順位 ---
            # 前走レースを特定
            uma_hist = ref[ref['KETTO_TOROKU_BANGO'] == ketto_toroku_bango].sort_values('RACE_DATE')
            if len(uma_hist) > 0:
                zensho = uma_hist.iloc[-1]
                zensho_race_code = zensho['RACE_CODE']
                # 前走レースの全馬データ
                zensho_race = ref[ref['RACE_CODE'] == zensho_race_code]
                # 秒数差
                try:
                    my_time = float(zensho['RACE_TIME']) if 'RACE_TIME' in zensho and zensho['RACE_TIME'] not in [None, '', 'nan'] else None
                    ichaku_time = zensho_race[zensho_race['KAKUTEI_CHAKUJUN'] == '1']['RACE_TIME']
                    ichaku_time = float(ichaku_time.iloc[0]) if len(ichaku_time) > 0 else None
                    feature_dict['ZENSHO_ICHIAKUMA_SA'] = my_time - ichaku_time if my_time is not None and ichaku_time is not None else 0
                except:
                    feature_dict['ZENSHO_ICHIAKUMA_SA'] = 0
                # 逆順順位
                try:
                    n_tousu = len(zensho_race)
                    chaku = int(zensho['KAKUTEI_CHAKUJUN']) if str(zensho['KAKUTEI_CHAKUJUN']).isdigit() else None
                    feature_dict['ZENSHO_GYAKUJUN'] = n_tousu - chaku + 1 if chaku is not None else 0
                except:
                    feature_dict['ZENSHO_GYAKUJUN'] = 0
            else:
                feature_dict['ZENSHO_ICHIAKUMA_SA'] = 0
                feature_dict['ZENSHO_GYAKUJUN'] = 0
            X.append([feature_dict[f.name] for f in FEATURES])
            chaku = int(row.iloc[4]) if len(row) > 4 else 0
            y.append(1 if chaku in [1,2,3] else 0)
        except (ValueError, TypeError, KeyError):
            continue
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    # 標準化（連続値のみ）
    if len(X) > 0:
        scaler = StandardScaler()
        num_idx = [i for i, f in enumerate(FEATURES) if not f.is_categorical]
        X[:, num_idx] = scaler.fit_transform(X[:, num_idx])
    fetch_time = time.time() - start_time
    print(f'【データ取得・前処理所要時間】{fetch_time:.2f}秒')
    return X, y

# --- PyTorch Dataset ---
class KeibaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- DQNネットワーク（Embedding対応） ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_info):
        super().__init__()
        # embedding_info: [(num_embeddings, embedding_dim), ...]
        self.embeddings = nn.ModuleList([
            nn.Embedding(num, dim) for num, dim in embedding_info
        ])
        emb_total_dim = sum([dim for _, dim in embedding_info])
        self.net = nn.Sequential(
            nn.Linear(input_dim + emb_total_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, output_dim)
        )
    def forward(self, x_cont, x_cat):
        # x_cont: [batch, input_dim], x_cat: list of [batch]
        emb = [emb_layer(x_cat[:,i]) for i, emb_layer in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1) if emb else torch.zeros((x_cont.size(0),0), device=x_cont.device)
        x = torch.cat([x_cont, emb], dim=1)
        return self.net(x)

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# --- 時系列分割で未来情報を遮断したデータ抽出 ---
def fetch_data_time_split(test_ratio=0.2):
    print('=== fetch_data_time_split関数 実行開始 ===')
    start_time = time.time()
    conn = get_db_connection()
    cursor = conn.cursor()
    today = datetime.today()
    date_min = (today - timedelta(days=target_days)).strftime('%Y%m%d')
    db_columns = [f.db_column for f in FEATURES if f.db_column]
    # KAKUTEI_CHAKUJUNを必ず含める
    if 'KAKUTEI_CHAKUJUN' not in db_columns:
        db_columns.append('KAKUTEI_CHAKUJUN')
    print('=== データ抽出開始 ===')
    select_columns = db_columns + ['KISHU_CODE','RACE_CODE','KETTO_TOROKU_BANGO','CHOKYOSHI_CODE','KEIBAJO_CODE','KAISAI_NEN','KAISAI_GAPPI']
    query = f'''
        SELECT {', '.join(select_columns)}
        FROM umagoto_race_joho
        WHERE CONCAT(KAISAI_NEN, LPAD(KAISAI_GAPPI, 4, '0')) >= %s
        AND {' AND '.join([f'{col} IS NOT NULL' for col in db_columns if col != 'KAKUTEI_CHAKUJUN'])}
        AND KISHU_CODE IS NOT NULL AND RACE_CODE IS NOT NULL AND KETTO_TOROKU_BANGO IS NOT NULL AND CHOKYOSHI_CODE IS NOT NULL AND KEIBAJO_CODE IS NOT NULL
    '''
    cursor.execute(query, (date_min,))
    rows = cursor.fetchall()
    columns = select_columns
    df_raw = pd.DataFrame(rows, columns=columns)
    assert 'KAKUTEI_CHAKUJUN' in df_raw.columns, 'KAKUTEI_CHAKUJUNカラムが存在しません'
    # 正規化: 全角→半角、空白除去、先頭ゼロ除去
    df_raw['KAKUTEI_CHAKUJUN'] = df_raw['KAKUTEI_CHAKUJUN'].astype(str).str.strip().replace('　','').replace(' ','').str.translate(str.maketrans('０１２３４５６７８９', '0123456789')).str.lstrip('0')
    df_raw['RACE_DATE'] = df_raw['KAISAI_NEN'].astype(str) + df_raw['KAISAI_GAPPI'].astype(str).str.zfill(4)
    df_raw['RACE_DATE'] = pd.to_datetime(df_raw['RACE_DATE'], format='%Y%m%d', errors='coerce')
    df_raw = df_raw.sort_values('RACE_DATE').reset_index(drop=True)
    # race_shosaiをmerge
    cursor.execute('SELECT RACE_CODE, KYORI, TRACK_CODE, SHUSSO_TOSU FROM race_shosai')
    race_shosai_df = pd.DataFrame(cursor.fetchall(), columns=['RACE_CODE','KYORI','TRACK_CODE','SHUSSO_TOSU'])
    for col in ['KYORI','TRACK_CODE','SHUSSO_TOSU']:
        race_shosai_df[col] = pd.to_numeric(race_shosai_df[col], errors='coerce').fillna(0)
    df_raw = pd.merge(df_raw, race_shosai_df, on='RACE_CODE', how='left')
    df_raw['KYORI'] = df_raw['KYORI'].fillna(0)
    df_raw['TRACK_CODE'] = df_raw['TRACK_CODE'].fillna(0)
    df_raw['SHUSSO_TOSU'] = df_raw['SHUSSO_TOSU'].fillna(0)
    # train/test分割
    df_raw = df_raw.sort_values('RACE_DATE').reset_index(drop=True)
    split_idx = int(len(df_raw) * (1 - test_ratio))
    train_df = df_raw.iloc[:split_idx].copy()
    test_df = df_raw.iloc[split_idx:].copy()
    # --- 特徴量生成 ---
    print('=== make_features関数 実行開始 ===')
    def make_features(df, ref_df):
        X, y = [], []
        for idx, (_, row) in enumerate(df.iterrows()):
            print_progress_bar(idx+1, len(df), bar_length=100, prefix='進捗', suffix='')
            race_date = row['RACE_DATE']
            ref = ref_df[ref_df['RACE_DATE'] < race_date]  # 未来情報遮断
            feature_dict = {}
            for f in FEATURES:
                if f.db_column:
                    val = row[f.db_column]
                    feature_dict[f.name] = float(val) if val is not None and str(val).replace('.', '', 1).isdigit() else 0
            # --- 騎手3着率 ---
            kishu_code = row['KISHU_CODE']
            if len(ref) > 0 and 'KAKUTEI_CHAKUJUN' in ref.columns:
                kishu_3in = ref[ref['KISHU_CODE'] == kishu_code]
                feature_dict['KISHU_3IN_RATE'] = kishu_3in['KAKUTEI_CHAKUJUN'].isin(['1','2','3']).mean() if 'KAKUTEI_CHAKUJUN' in kishu_3in else 0
            else:
                feature_dict['KISHU_3IN_RATE'] = 0
            # --- 調教師3着率 ---
            chokyoshi_code = row['CHOKYOSHI_CODE']
            if len(ref) > 0 and 'KAKUTEI_CHAKUJUN' in ref.columns:
                chokyoshi_3in = ref[ref['CHOKYOSHI_CODE'] == chokyoshi_code]
                feature_dict['CHOKYOSHI_3IN_RATE'] = chokyoshi_3in['KAKUTEI_CHAKUJUN'].isin(['1','2','3']).mean() if 'KAKUTEI_CHAKUJUN' in chokyoshi_3in else 0
            else:
                feature_dict['CHOKYOSHI_3IN_RATE'] = 0
            # --- 父系統ID ---
            ketto_toroku_bango = row['KETTO_TOROKU_BANGO']
            if 'FATHER_LINEAGE' in feature_dict:
                pass  # 既にdb_columnから取得
            else:
                feature_dict['FATHER_LINEAGE'] = 0
            # --- 父系統TE ---
            if len(ref) > 0 and 'KAKUTEI_CHAKUJUN' in ref.columns:
                father_id = ketto_toroku_bango  # 仮: 本来は父ID列が必要
                if 'FATHER_LINEAGE' in row:
                    father_id = row['FATHER_LINEAGE']
                father_ref = ref[ref['KETTO_TOROKU_BANGO'] == father_id]
                feature_dict['FATHER_LINEAGE_TE'] = father_ref['KAKUTEI_CHAKUJUN'].isin(['1','2','3']).mean() if len(father_ref) > 0 else 0
            else:
                feature_dict['FATHER_LINEAGE_TE'] = 0
            # --- トラックコードTE ---
            if len(ref) > 0 and 'KAKUTEI_CHAKUJUN' in ref.columns and 'TRACK_CODE' in ref.columns:
                track_code = row['TRACK_CODE']
                track_ref = ref[ref['TRACK_CODE'] == track_code]
                feature_dict['TRACK_CODE_TE'] = track_ref['KAKUTEI_CHAKUJUN'].isin(['1','2','3']).mean() if len(track_ref) > 0 else 0
            else:
                feature_dict['TRACK_CODE_TE'] = 0
            # --- 騎手×調教師TE ---
            if len(ref) > 0 and 'KAKUTEI_CHAKUJUN' in ref.columns:
                kc_ref = ref[(ref['KISHU_CODE'] == kishu_code) & (ref['CHOKYOSHI_CODE'] == chokyoshi_code)]
                feature_dict['KISHU_CHOKYOSHI_TE'] = kc_ref['KAKUTEI_CHAKUJUN'].isin(['1','2','3']).mean() if len(kc_ref) > 0 else 0
            else:
                feature_dict['KISHU_CHOKYOSHI_TE'] = 0
            # --- 馬×コースTE ---
            keibajo_code = row['KEIBAJO_CODE'] if 'KEIBAJO_CODE' in row else 0
            if len(ref) > 0 and 'KAKUTEI_CHAKUJUN' in ref.columns:
                uma_course_ref = ref[(ref['KETTO_TOROKU_BANGO'] == ketto_toroku_bango) & (ref['KEIBAJO_CODE'] == keibajo_code)]
                feature_dict['UMA_COURSE_TE'] = uma_course_ref['KAKUTEI_CHAKUJUN'].isin(['1','2','3']).mean() if len(uma_course_ref) > 0 else 0
            else:
                feature_dict['UMA_COURSE_TE'] = 0
            # --- 馬過去5走3着以内率 ---
            if len(ref) > 0 and 'KAKUTEI_CHAKUJUN' in ref.columns:
                uma_hist = ref[ref['KETTO_TOROKU_BANGO'] == ketto_toroku_bango].sort_values('RACE_DATE')
                past5 = uma_hist.tail(5)['KAKUTEI_CHAKUJUN'] if len(uma_hist) > 0 else []
                feature_dict['UMA_5R_3IN_RATE'] = past5.isin(['1','2','3']).mean() if len(past5) > 0 else 0
            else:
                feature_dict['UMA_5R_3IN_RATE'] = 0
            # --- コース適性3着以内率 ---
            if len(ref) > 0 and 'KYORI' in ref.columns and 'KEIBAJO_CODE' in ref.columns and 'KAKUTEI_CHAKUJUN' in ref.columns:
                kyori = row['KYORI'] if 'KYORI' in row else 0
                course_ref = ref[(ref['KEIBAJO_CODE'] == keibajo_code) & (ref['KYORI'] == kyori)]
                feature_dict['COURSE_3IN_RATE'] = course_ref['KAKUTEI_CHAKUJUN'].isin(['1','2','3']).mean() if len(course_ref) > 0 else 0
            else:
                feature_dict['COURSE_3IN_RATE'] = 0
            # --- コース×馬過去5走3着以内率 ---
            if len(ref) > 0 and 'KYORI' in ref.columns and 'KEIBAJO_CODE' in ref.columns and 'KAKUTEI_CHAKUJUN' in ref.columns:
                course_uma_hist = ref[(ref['KETTO_TOROKU_BANGO'] == ketto_toroku_bango) & (ref['KEIBAJO_CODE'] == keibajo_code)].sort_values('RACE_DATE')
                course_uma5 = course_uma_hist.tail(5)['KAKUTEI_CHAKUJUN'] if len(course_uma_hist) > 0 else []
                feature_dict['COURSE_UMA_5R_3IN_RATE'] = course_uma5.isin(['1','2','3']).mean() if len(course_uma5) > 0 else 0
            else:
                feature_dict['COURSE_UMA_5R_3IN_RATE'] = 0
            # --- その他の特徴量（例: 直近体重変化, 順位, 組み合わせ特徴量等）は未来情報を使わない形で既存通り ---
            chaku = int(row.iloc[4]) if len(row) > 4 else 0
            y.append(1 if chaku in [1,2,3] else 0)
            X.append([feature_dict.get(f.name, 0) for f in FEATURES])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
    X_train, y_train = make_features(train_df, train_df)
    X_test, y_test = make_features(test_df, train_df)  # testもtrainまでの情報のみで特徴量生成
    fetch_time = time.time() - start_time
    print(f'【データ取得・前処理所要時間】{fetch_time:.2f}秒')
    return X_train, y_train, X_test, y_test, train_df, test_df

# --- データ前処理（embedding用index化） ---
def preprocess_for_embedding(X_raw, df_raw):
    # embedding対象: is_categoricalな全特徴量
    cat_features = [f for f in FEATURES if f.is_categorical]
    cat_lists = [sorted(df_raw[f.db_column].unique()) if f.db_column in df_raw else [0] for f in cat_features]
    cat2idxs = [{k: i for i, k in enumerate(lst)} for lst in cat_lists]
    X_cat = np.stack([
        df_raw[f.db_column].map(cat2idxs[i]).values if f.db_column in df_raw else np.zeros(len(df_raw), dtype=int)
        for i, f in enumerate(cat_features)
    ], axis=1)
    X_cont = X_raw
    embedding_info = [(len(lst), min(8, max(2, len(lst)//2))) for lst in cat_lists]
    return X_cont, X_cat, embedding_info

# --- Datasetクラス修正 ---
class KeibaDataset(Dataset):
    def __init__(self, X_cont, X_cat, y):
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X_cont)
    def __getitem__(self, idx):
        return self.X_cont[idx], self.X_cat[idx], self.y[idx]

# --- train関数の修正 ---
def train():
    run_dir = save_script_and_make_run_dir('sanshutsu_kun/1_predict_models/a_value_based_dqn/results')
    print('=== train関数 実行開始 ===')
    start_time = time.time()
    X_train, y_train, X_test, y_test, train_df, test_df = fetch_data_time_split(test_ratio=0.2)
    # embedding用index化（SMOTE前）
    X_train_cont, X_train_cat, embedding_info = preprocess_for_embedding(X_train, train_df)
    X_test_cont, X_test_cat, _ = preprocess_for_embedding(X_test, test_df)
    # NaN埋め
    X_train_cont = np.nan_to_num(X_train_cont, nan=0)
    X_test_cont = np.nan_to_num(X_test_cont, nan=0)
    # --- SMOTEでオーバーサンプリング（連続値のみ）---
    smote = SMOTE(random_state=42)
    X_train_cont_res, y_train_res = smote.fit_resample(X_train_cont, y_train)
    if hasattr(smote, 'sample_indices_'):
        valid_indices = smote.sample_indices_[smote.sample_indices_ < len(X_train_cat)]
        if len(valid_indices) == len(X_train_cont_res):
            X_train_cat_res = X_train_cat[valid_indices]
        else:
            reps = len(X_train_cont_res) // len(X_train_cat) + 1
            X_train_cat_res = np.tile(X_train_cat, (reps, 1))[:len(X_train_cont_res)]
    else:
        n_res = len(X_train_cont_res)
        n_orig = len(X_train_cat)
        if n_res > n_orig:
            reps = n_res // n_orig + 1
            X_train_cat_res = np.tile(X_train_cat, (reps, 1))[:n_res]
        else:
            X_train_cat_res = X_train_cat
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_res), y=y_train_res)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print('DQN学習開始')
    dataset = KeibaDataset(X_train_cont_res, X_train_cat_res, y_train_res)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = DQN(input_dim=X_train_cont_res.shape[1], output_dim=2, embedding_info=embedding_info)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = FocalLoss(alpha=class_weights[1].item(), gamma=2)
    train_losses = []
    train_accuracies = []
    best_val_loss = float('inf')
    patience = 1
    patience_counter = 0
    for epoch in range(10):
        print_progress_bar(epoch+1, 10, bar_length=100, prefix='進捗', suffix='')
        epoch_loss = 0
        correct = 0
        total = 0
        model.train()
        for batch_X_cont, batch_X_cat, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_X_cont, batch_X_cat)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X_cont.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        train_losses.append(epoch_loss / total)
        train_accuracies.append(correct / total)
        # EarlyStopping: val_lossで判定
        model.eval()
        with torch.no_grad():
            val_out = model(torch.tensor(X_test_cont, dtype=torch.float32), torch.tensor(X_test_cat, dtype=torch.long))
            val_loss = criterion(val_out, torch.tensor(y_test, dtype=torch.long)).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'EarlyStopping発動（{epoch+1}エポックで停止）')
                break
    # --- DQN単体で予測・評価 ---
    print('DQN単体で予測・評価開始')
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth')))
    model.eval()
    with torch.no_grad():
        X_tensor_cont = torch.tensor(X_test_cont, dtype=torch.float32)
        X_tensor_cat = torch.tensor(X_test_cat, dtype=torch.long)
        probs = torch.softmax(model(X_tensor_cont, X_tensor_cat), dim=1)[:,1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f'DQN Accuracy: {acc:.4f}, F1: {f1:.4f}')
    torch.save(model.state_dict(), os.path.join(run_dir, 'dqn_keiba_simple.pth'))
    print('モデル保存済み')
    evaluate(model, X_test_cont, X_test_cat, y_test, train_losses, train_accuracies, run_dir, test_df)

def summarize_top_features(df_raw, y_test, run_dir):
    print('=== summarize_top_features関数 実行開始 ===')
    import pandas as pd
    import numpy as np
    import pymysql
    # y_test: 3着以内=1, それ以外=0
    df = df_raw.copy()
    df['target'] = y_test
    # --- 馬ID→馬名、騎手ID→騎手名の変換辞書をDBから取得 ---
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, db=DB_NAME, port=DB_PORT, charset='utf8mb4')
    cursor = conn.cursor()
    cursor.execute('SELECT KETTO_TOROKU_BANGO, BAMEI FROM kyosoba_master2')
    umaid2name = {str(row[0]).strip().zfill(10): (row[1] if row[1] else '(不明)') for row in cursor.fetchall()}
    cursor.execute('SELECT KISHU_CODE, KISHUMEI FROM kishu_master')
    kishu2name = {str(row[0]).strip().zfill(5): (row[1] if row[1] else '(不明)') for row in cursor.fetchall()}
    conn.close()
    print(f'馬名変換例: {list(umaid2name.items())[:3]}')
    print(f'騎手名変換例: {list(kishu2name.items())[:3]}')
    feature_summaries = []
    for f in FEATURES:
        name = f.name
        desc = f.description
        if name not in df.columns:
            continue
        col = df[name]
        # 連続値はビン分割（数値型のみ）
        if not f.is_categorical and col.nunique() > 10 and np.issubdtype(col.dropna().dtype, np.number):
            bins = np.linspace(col.min(), col.max(), 6)
            df[f'{name}_bin'] = pd.cut(col, bins, include_lowest=True)
            group_col = f'{name}_bin'
        else:
            group_col = name
        grp = df.groupby(group_col)['target'].agg(['mean','count']).reset_index()
        for _, row in grp.iterrows():
            # 該当する馬・騎手の組み合わせ（上位3件、名称で）
            mask = (df[group_col] == row[group_col])
            horses = df.loc[mask, 'KETTO_TOROKU_BANGO'].astype(str).str.strip().str.zfill(10).unique()[:3]
            horses_name = [umaid2name.get(h, f'{h}(不明)') for h in horses]
            jockeys = df.loc[mask, 'KISHU_CODE'].astype(str).str.strip().str.zfill(5).unique()[:3] if 'KISHU_CODE' in df.columns else []
            jockeys_name = [kishu2name.get(k, f'{k}(不明)') for k in jockeys]
            feature_summaries.append({
                'feature': desc,
                'feature_name': name,
                'value': str(row[group_col]),
                'rate': row['mean'],
                'count': int(row['count']),
                'horses': ','.join(horses_name),
                'jockeys': ','.join(jockeys_name)
            })
    # 件数10以上のみ、3着率降順で上位15
    feature_summaries = [d for d in feature_summaries if d['count'] >= 10]
    feature_summaries = sorted(feature_summaries, key=lambda x: x['rate'], reverse=True)[:15]
    # テキスト化
    lines = []
    for i, d in enumerate(feature_summaries, 1):
        lines.append(f"{i}. {d['feature']}（{d['feature_name']}）: {d['value']} → 3着以内率={d['rate']:.2%}（{d['count']}件）")
        if d['horses']:
            lines.append(f"   馬例: {d['horses']}")
        if d['jockeys']:
            lines.append(f"   騎手例: {d['jockeys']}")
    txt = '\n'.join(lines)
    with open(os.path.join(run_dir, 'top15_feature_ranking.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)
    return txt

def evaluate(model, X_test_cont, X_test_cat, y_test, train_losses, train_accuracies, run_dir, df_raw):
    print('=== evaluate関数 実行開始 ===')
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        X_tensor_cont = torch.tensor(X_test_cont, dtype=torch.float32)
        X_tensor_cat = torch.tensor(X_test_cat, dtype=torch.long)
        outputs = model(X_tensor_cont, X_tensor_cat)
        probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds)
    print('--- 評価結果 ---')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(report)
    # PRカーブからF1最大化・precision重視・recall重視の複数閾値で自動評価
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = f1s.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    # precision重視（0.8以上）
    mask_p = precisions >= 0.8
    best_idx_p = (f1s * mask_p).argmax() if mask_p.sum() > 0 else best_idx
    best_threshold_p = thresholds[best_idx_p] if best_idx_p < len(thresholds) else 0.5
    # recall重視（0.8以上）
    mask_r = recalls >= 0.8
    best_idx_r = (f1s * mask_r).argmax() if mask_r.sum() > 0 else best_idx
    best_threshold_r = thresholds[best_idx_r] if best_idx_r < len(thresholds) else 0.5
    print(f'--- PRカーブ最適閾値: F1最大={best_threshold:.3f}, precision重視={best_threshold_p:.3f}, recall重視={best_threshold_r:.3f} ---')
    # 各閾値で再評価
    for th, label in zip([best_threshold, best_threshold_p, best_threshold_r], ['F1最大','precision重視','recall重視']):
        best_preds = (probs >= th).astype(int)
        best_acc = accuracy_score(y_test, best_preds)
        best_f1 = f1_score(y_test, best_preds)
        best_report = classification_report(y_test, best_preds)
        print(f'--- {label}での再評価（閾値={th:.3f}）---')
        print(f'Accuracy: {best_acc:.4f}')
        print(f'F1 Score: {best_f1:.4f}')
        print(best_report)
    # グラフ出力
    graph_paths = plot_and_save_graphs(y_test, preds, probs, train_losses, train_accuracies, run_dir)
    # 画像base64化
    images_b64 = []
    for path in graph_paths:
        with open(path, 'rb') as imgf:
            b64img = base64.b64encode(imgf.read()).decode('utf-8')
            images_b64.append((os.path.basename(path), b64img))
    # プロンプト生成部分
    feature_desc = get_feature_description()
    prompt = f"""
競馬の3着以内に入る馬の特徴をDQNで学習したモデルの評価結果です。

【モデルで利用している特徴量（入力項目）】
{feature_desc}

精度: {acc:.4f}
F1スコア: {f1:.4f}
詳細:
{report}

PRカーブからF1最大化となる最適閾値: {best_threshold:.3f}
最適閾値での再評価:
精度: {best_acc:.4f}
F1スコア: {best_f1:.4f}
詳細:
{best_report}

以下の4つのグラフ画像（base64エンコード済み）も参考にして、モデル改善のためのアドバイスを日本語で簡潔に出力してください。
"""
    gpt_advice = ask_gpt41(prompt, images_b64)
    # 結果をファイル保存
    with open(os.path.join(run_dir, 'eval_and_gpt_advice.txt'), 'w', encoding='utf-8') as f:
        # 追加: 対象年数・特徴量リスト
        f.write('【評価対象データ】\n')
        f.write(f'・対象期間: 直近{target_days}日分\n')
        f.write('・使用特徴量:\n')
        for feat in FEATURES:
            f.write(f'  - {feat.description}（{feat.name}）\n')
        f.write('\n')
        f.write('--- 評価結果 ---\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(report + '\n')
        f.write(f'--- PRカーブからF1最大化となる最適閾値: {best_threshold:.3f} ---\n')
        f.write('--- 最適閾値での再評価 ---\n')
        f.write(f'Accuracy: {best_acc:.4f}\n')
        f.write(f'F1 Score: {best_f1:.4f}\n')
        f.write(best_report + '\n')
        f.write('\n--- GPT-4.1からの改善アドバイス ---\n')
        f.write(gpt_advice + '\n')
    eval_time = time.time() - start_time
    print(f'【推論・評価所要時間】{eval_time:.2f}秒')
    evaluate_feature_importance(model, X_test_cont, X_test_cat, y_test, run_dir)
    # --- 追加: 特徴量ごとの値別3着以内率ランキングを出力 ---
    summarize_top_features(df_raw, y_test, run_dir)

def evaluate_feature_importance(model, X_test_cont, X_test_cat, y_test, run_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score
    # 日本語フォント設定
    try:
        font_path = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf'
        if not os.path.exists(font_path):
            font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
        if not os.path.exists(font_path):
            font_path = '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
    except Exception as e:
        pass
    base_pred = torch.argmax(model(torch.tensor(X_test_cont, dtype=torch.float32), torch.tensor(X_test_cat, dtype=torch.long)), dim=1).cpu().numpy()
    base_f1 = f1_score(y_test, base_pred)
    importances = []
    feature_names = [f.name for f in FEATURES]
    for i in range(X_test_cont.shape[1]):
        X_perm = X_test_cont.copy()
        np.random.shuffle(X_perm[:,i])
        perm_pred = torch.argmax(model(torch.tensor(X_perm, dtype=torch.float32), torch.tensor(X_test_cat, dtype=torch.long)), dim=1).cpu().numpy()
        perm_f1 = f1_score(y_test, perm_pred)
        importances.append(base_f1 - perm_f1)
    # 棒グラフ
    plt.figure(figsize=(10,6))
    idx = np.argsort(importances)[::-1]
    plt.bar(np.array(feature_names)[idx], np.array(importances)[idx])
    plt.xticks(rotation=90)
    plt.title('Permutation Importance (F1低下度)')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'feature_importance.png'))
    plt.close()

# --- 特徴量説明文の自動生成 ---
def get_feature_description():
    return '\n'.join([f'- {f.description}（{f.name}）' for f in FEATURES])

# 既存のmain部
if __name__ == '__main__':
    validate_pythonpath('dqn_keiba_simple.py')
    target_days = parse_days_arg(sys.argv, 'dqn_keiba_simple.py')
    start_time = time.time()
    print('=== DQNモデル 実行開始 ===')
    run_dir = save_script_and_make_run_dir('sanshutsu_kun/1_predict_models/a_value_based_dqn/results')
    params = {
        'epochs': 10,
        'batch_size': 64,
        'lr': 0.001,
        'gamma': 0.99,
        'hidden_dim': 64,
        'test_ratio': 0.2,
    }
    train()
    end_time = time.time()
    print('=== DQNモデル 実行終了 ===')
    print(f'【実行所要時間】{end_time - start_time:.2f}秒')