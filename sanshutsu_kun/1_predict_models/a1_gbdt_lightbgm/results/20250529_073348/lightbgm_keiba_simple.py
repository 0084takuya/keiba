import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import pymysql
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix, recall_score, average_precision_score
from dotenv import load_dotenv
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import base64
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GroupKFold
from matplotlib import font_manager
import time
from modules.progress_utils import print_progress_bar
from modules.gpt_utils import ask_gpt41
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from modules.plot_utils import save_script_and_make_run_dir, plot_and_save_graphs

# --- .envからAPIキーをロード ---
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# --- DB接続設定 ---
DB_HOST = os.getenv('KEIBA_DB_HOST', 'localhost')
DB_USER = os.getenv('KEIBA_DB_USER', 'root')
DB_PASS = os.getenv('KEIBA_DB_PASS', '')
DB_NAME = os.getenv('KEIBA_DB_NAME', 'mykeibadb')
DB_PORT = int(os.getenv('KEIBA_DB_PORT', 3306))

# データ抽出期間（日数）
target_days = 120  # 例: 直近90日分

# --- 日本語フォントグローバル設定 ---
import matplotlib
font_candidates = [
    'IPAexGothic', 'Noto Sans CJK JP', 'Yu Gothic', 'Meiryo', 'TakaoGothic', 'MS Gothic', 'VL Gothic', 'Osaka', 'Arial Unicode MS'
]
found_font = None
font_path = None
for font_name in font_candidates:
    matches = [f for f in font_manager.fontManager.ttflist if font_name in f.name]
    if matches:
        matplotlib.rcParams['font.family'] = font_name
        found_font = font_name
        font_path = matches[0].fname
        print(f'日本語フォント設定: {font_name}')
        break
if not found_font:
    print('警告: 日本語フォントが見つかりません。デフォルトフォントで描画します。')
from matplotlib.font_manager import FontProperties
jp_font = FontProperties(fname=font_path) if font_path else None

def fetch_data_time_split(test_ratio=0.2):
    print('=== fetch_data_time_split関数 実行開始 ===')
    start_time = time.time()
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, db=DB_NAME, port=DB_PORT, charset='utf8mb4')
    cursor = conn.cursor()
    today = datetime.today()
    date_min = (today - timedelta(days=target_days)).strftime('%Y%m%d')
    # --- メインデータ ---
    print('メインデータ取得開始')
    query = '''SELECT BAREI, SEIBETSU_CODE, BATAIJU, FUTAN_JURYO, TANSHO_NINKIJUN, KISHU_CODE, CHOKYOSHI_CODE, KETTO_TOROKU_BANGO, RACE_CODE, KAISAI_NEN, KAISAI_GAPPI, KAKUTEI_CHAKUJUN, WAKUBAN, UMABAN FROM umagoto_race_joho WHERE CONCAT(KAISAI_NEN, LPAD(KAISAI_GAPPI, 4, '0')) >= %s AND KISHU_CODE IS NOT NULL AND RACE_CODE IS NOT NULL AND KETTO_TOROKU_BANGO IS NOT NULL AND CHOKYOSHI_CODE IS NOT NULL AND KAISAI_NEN IS NOT NULL AND KAISAI_GAPPI IS NOT NULL AND WAKUBAN IS NOT NULL AND UMABAN IS NOT NULL'''
    cursor.execute(query, (date_min,))
    rows = cursor.fetchall()
    columns = ['BAREI','SEIBETSU_CODE','BATAIJU','FUTAN_JURYO','TANSHO_NINKIJUN','KISHU_CODE','CHOKYOSHI_CODE','KETTO_TOROKU_BANGO','RACE_CODE','KAISAI_NEN','KAISAI_GAPPI','KAKUTEI_CHAKUJUN','WAKUBAN','UMABAN']
    df = pd.DataFrame(rows, columns=columns)
    # --- レース日付 ---
    df['RACE_DATE'] = df['KAISAI_NEN'].astype(str) + df['KAISAI_GAPPI'].astype(str).str.zfill(4)
    df['RACE_DATE'] = pd.to_datetime(df['RACE_DATE'], format='%Y%m%d', errors='coerce')
    df = df.sort_values(['RACE_DATE', 'RACE_CODE', 'KETTO_TOROKU_BANGO']).reset_index(drop=True)
    # --- ラベル ---
    df['KAKUTEI_CHAKUJUN_norm'] = df['KAKUTEI_CHAKUJUN'].apply(lambda x: str(x).strip().replace('　','').replace(' ','').translate(str.maketrans('０１２３４５６７８９', '0123456789')).lstrip('0'))
    df['target'] = df['KAKUTEI_CHAKUJUN_norm'].isin(['1','2','3']).astype(int)
    # --- odds1_tansho ---
    print('odds1_tansho取得開始')
    cursor.execute('SELECT RACE_CODE, UMABAN, ODDS, NINKI FROM odds1_tansho')
    odds_tansho = {}
    for row in cursor.fetchall():
        try:
            odds = float(row[2])
        except:
            odds = 0
        try:
            ninki = int(row[3])
        except:
            ninki = 0
        odds_tansho[(row[0], row[1])] = (odds, ninki)
    # --- odds1_fukusho ---
    print('odds1_fukusho取得開始')
    cursor.execute('SELECT RACE_CODE, UMABAN, ODDS_SAITEI, ODDS_SAIKOU, NINKI FROM odds1_fukusho')
    odds_fukusho = {}
    for row in cursor.fetchall():
        try:
            saitei = float(row[2])
        except:
            saitei = 0
        try:
            saikou = float(row[3])
        except:
            saikou = 0
        try:
            ninki = int(row[4])
        except:
            ninki = 0
        odds_fukusho[(row[0], row[1])] = (saitei, saikou, ninki)
    # --- keito_joho2（血統） ---
    print('keito_joho2取得開始')
    cursor.execute('SELECT HANSHOKU_TOROKU_BANGO, KEITO_ID, KEITO_MEI FROM keito_joho2')
    keito_dict = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    # --- bataiju（直近馬体重） ---
    print('bataiju取得開始')
    cursor.execute('SELECT RACE_CODE, UMABAN1, BATAIJU1 FROM bataiju')
    bataiju_dict = {}
    for row in cursor.fetchall():
        try:
            b = float(row[2])
        except:
            b = 0
        bataiju_dict[(row[0], row[1])] = b
    # --- kishu_master（騎手成績） ---
    print('kishu_master取得開始')
    cursor.execute('SELECT KISHU_CODE, HEICHI_1CHAKU_HONNEN, HEICHI_2CHAKU_HONNEN, HEICHI_3CHAKU_HONNEN, HEICHI_CHAKUGAI_HONNEN FROM kishu_master')
    kishu_seiseki = {}
    for row in cursor.fetchall():
        try:
            ichi = int(row[1]) if row[1] else 0
            ni = int(row[2]) if row[2] else 0
            san = int(row[3]) if row[3] else 0
            gai = int(row[4]) if row[4] else 0
            total = ichi + ni + san + gai
            win = ichi / total if total > 0 else 0
            ren = (ichi + ni) / total if total > 0 else 0
            fuku = (ichi + ni + san) / total if total > 0 else 0
        except:
            win, ren, fuku = 0, 0, 0
        kishu_seiseki[row[0]] = (win, ren, fuku)
    # --- race_shosai（馬場・コース・距離・天候） ---
    print('race_shosai取得開始')
    cursor.execute('SELECT RACE_CODE, KYORI, TRACK_CODE, SHIBA_BABAJOTAI_CODE, DIRT_BABAJOTAI_CODE, TENKO_CODE, COURSE_KUBUN FROM race_shosai')
    race_shosai_dict = {}
    for row in cursor.fetchall():
        race_shosai_dict[row[0]] = {
            'KYORI': float(row[1]) if row[1] else 0,
            'TRACK_CODE': row[2] or '',
            'SHIBA_BABAJOTAI_CODE': row[3] or '',
            'DIRT_BABAJOTAI_CODE': row[4] or '',
            'TENKO_CODE': row[5] or '',
            'COURSE_KUBUN': row[6] or ''
        }
    # --- 前走情報（自己JOIN） ---
    print('前走情報JOIN開始')
    df['prev_race_date'] = pd.NaT
    df['prev_chaku'] = 0
    df['prev_ninki'] = 0
    df['prev_bataiju'] = 0
    df['prev_kishu'] = 0
    df['prev_course'] = 0
    df['prev_days'] = 0
    df['syussou_count'] = 0
    for uma, group in df.groupby('KETTO_TOROKU_BANGO'):
        group = group.sort_values('RACE_DATE').reset_index()
        prev_race_date = None
        syussou_count = 0
        for i, row in group.iterrows():
            idx = row['index']
            syussou_count += 1
            if prev_race_date is not None:
                days = (row['RACE_DATE'] - prev_race_date).days
            else:
                days = 9999
            df.loc[idx, 'syussou_count'] = syussou_count
            df.loc[idx, 'prev_days'] = days
            if i > 0:
                prev_row = group.iloc[i-1]
                df.loc[idx, 'prev_race_date'] = prev_row['RACE_DATE']
                df.loc[idx, 'prev_chaku'] = prev_row['KAKUTEI_CHAKUJUN_norm'] if 'KAKUTEI_CHAKUJUN_norm' in prev_row else 0
                df.loc[idx, 'prev_ninki'] = prev_row['TANSHO_NINKIJUN'] if 'TANSHO_NINKIJUN' in prev_row else 0
                df.loc[idx, 'prev_bataiju'] = prev_row['BATAIJU'] if 'BATAIJU' in prev_row else 0
                df.loc[idx, 'prev_kishu'] = prev_row['KISHU_CODE'] if 'KISHU_CODE' in prev_row else 0
                df.loc[idx, 'prev_course'] = prev_row['RACE_CODE'] if 'RACE_CODE' in prev_row else 0
            prev_race_date = row['RACE_DATE']
    # --- 直近3走の着順平均・人気平均 ---
    print('直近3走の着順平均・人気平均計算')
    df['chaku_list'] = [[] for _ in range(len(df))]
    df['ninki_list'] = [[] for _ in range(len(df))]
    for uma, group in df.groupby('KETTO_TOROKU_BANGO'):
        group = group.sort_values('RACE_DATE').reset_index()
        chaku_hist = []
        ninki_hist = []
        for i, row in group.iterrows():
            idx = row['index']
            # 直近3走の履歴
            if i > 0:
                chaku_hist.append(int(row['KAKUTEI_CHAKUJUN_norm']) if str(row['KAKUTEI_CHAKUJUN_norm']).isdigit() else 99)
                ninki_hist.append(int(row['TANSHO_NINKIJUN']) if str(row['TANSHO_NINKIJUN']).isdigit() else 99)
            # 直近3走の平均
            df.loc[idx, 'chaku_list'] = chaku_hist[-3:]
            df.loc[idx, 'ninki_list'] = ninki_hist[-3:]
            df.loc[idx, 'chaku3mean'] = np.mean(chaku_hist[-3:]) if chaku_hist[-3:] else 99
            df.loc[idx, 'ninki3mean'] = np.mean(ninki_hist[-3:]) if ninki_hist[-3:] else 99
    # --- 特徴量生成 ---
    feature_rows = []
    print('特徴量生成開始')
    for idx, row in df.iterrows():
        print_progress_bar(idx+1, len(df), bar_length=100, prefix='進捗', suffix='')
        uma = row['KETTO_TOROKU_BANGO']
        race_code = row['RACE_CODE']
        umaban = row['UMABAN']
        odds_t = odds_tansho.get((race_code, umaban), (0, 0))
        odds_f = odds_fukusho.get((race_code, umaban), (0, 0, 0))
        keito = keito_dict.get(uma, ('', ''))
        bataiju = bataiju_dict.get((race_code, umaban), 0)
        kishu_win, kishu_ren, kishu_fuku = kishu_seiseki.get(row['KISHU_CODE'], (0, 0, 0))
        race_shosai = race_shosai_dict.get(race_code, {'KYORI':0,'TRACK_CODE':'','SHIBA_BABAJOTAI_CODE':'','DIRT_BABAJOTAI_CODE':'','TENKO_CODE':'','COURSE_KUBUN':''})
        feats = {
            'BAREI': row['BAREI'],
            'SEIBETSU_CODE': row['SEIBETSU_CODE'],
            'BATAIJU': row['BATAIJU'],
            'FUTAN_JURYO': row['FUTAN_JURYO'],
            'KISHU_CODE': row['KISHU_CODE'],
            'CHOKYOSHI_CODE': row['CHOKYOSHI_CODE'],
            'WAKUBAN': row['WAKUBAN'],
            'TANSHO_NINKIJUN': row['TANSHO_NINKIJUN'],
            'TANSHO_ODDS': odds_t[0],
            'TANSHO_ODDS_NINKI': odds_t[1],
            'FUKUSHO_ODDS_SAITEI': odds_f[0],
            'FUKUSHO_ODDS_SAIKOU': odds_f[1],
            'FUKUSHO_ODDS_NINKI': odds_f[2],
            'KEITO_ID': keito[0],
            'KEITO_MEI': keito[1],
            'BATAIJU_LATEST': bataiju,
            'KISHU_WIN_RATE': kishu_win,
            'KISHU_RENTAI_RATE': kishu_ren,
            'KISHU_FUKUSHO_RATE': kishu_fuku,
            'KYORI': race_shosai['KYORI'],
            'TRACK_CODE': race_shosai['TRACK_CODE'],
            'SHIBA_BABAJOTAI_CODE': race_shosai['SHIBA_BABAJOTAI_CODE'],
            'DIRT_BABAJOTAI_CODE': race_shosai['DIRT_BABAJOTAI_CODE'],
            'TENKO_CODE': race_shosai['TENKO_CODE'],
            'COURSE_KUBUN': race_shosai['COURSE_KUBUN'],
            'prev_chaku': row['prev_chaku'],
            'prev_ninki': row['prev_ninki'],
            'prev_bataiju': row['prev_bataiju'],
            'prev_kishu': row['prev_kishu'],
            'prev_course': row['prev_course'],
            'prev_days': row['prev_days'],
            'syussou_count': row['syussou_count'],
            'chaku3mean': row['chaku3mean'],
            'ninki3mean': row['ninki3mean'],
        }
        feature_rows.append(feats)
    feature_df = pd.DataFrame(feature_rows)
    # カテゴリ変数をLabelEncoder
    for col in ['SEIBETSU_CODE','KISHU_CODE','CHOKYOSHI_CODE','WAKUBAN','KEITO_ID','KEITO_MEI','TRACK_CODE','SHIBA_BABAJOTAI_CODE','DIRT_BABAJOTAI_CODE','TENKO_CODE','COURSE_KUBUN','prev_kishu','prev_course']:
        if col in feature_df.columns:
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col].astype(str))
    feature_df = feature_df.fillna(0)
    feature_df = feature_df.replace('', 0)
    float_cols = [col for col in feature_df.columns]
    feature_df[float_cols] = feature_df[float_cols].astype(float)
    n1 = (df['target'] == 1).sum()
    n0 = (df['target'] == 0).sum()
    print(f'target=1(3着以内): {n1}件, target=0: {n0}件')
    if n1 < 100:
        print('警告: 3着以内データが極端に少ないです。抽出期間や条件を見直してください。')
    feature_cols = list(feature_df.columns)
    X = feature_df[feature_cols]
    y = df['target']
    # --- 負例ダウンサンプリング ---
    pos_idx = y_train[y_train==1].index
    neg_idx = y_train[y_train==0].sample(n=len(pos_idx), random_state=42)
    idx = pos_idx.union(neg_idx)
    X_train = X_train.loc[idx].reset_index(drop=True)
    y_train = y_train.loc[idx].reset_index(drop=True)
    end_time = time.time()
    print(f'【fetch_data_time_split関数 実行所要時間】{end_time - start_time:.2f}秒')
    return X_train, X_test, y_train, y_test, n0, n1, df

def main():
    run_dir = save_script_and_make_run_dir('sanshutsu_kun/1_predict_models/a1_gbdt_lightbgm/results')
    print('=== LightGBM競馬3着以内予測 実行開始 ===')
    X_train, X_test, y_train, y_test, n0, n1, df = fetch_data_time_split()
    if len(X_train) == 0:
        print('データがありません')
        return
    # --- LightGBMパラメータ ---
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 7,
        'learning_rate': 0.03,
        'feature_fraction': 0.9,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'scale_pos_weight': 1.0,
        'class_weight': 'balanced',
        'boost_from_average': True,
        'device_type': 'cpu',
        'deterministic': True,
        'seed': 42,
        'extra_trees': True,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_child_samples': 10,
        'min_split_gain': 0.01,
        'min_child_weight': 1e-3,
        'max_bin': 255,
        'num_threads': 4,
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc', 'average_precision'],
        'focal_loss': True,
    }
    evals_result = {}
    lgb_model = lgb.train(
        lgb_params, lgb.Dataset(X_train, y_train), valid_sets=[lgb.Dataset(X_train, y_train), lgb.Dataset(X_test, y_test)],
        valid_names=['train','valid'], num_boost_round=200,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20), lgb.record_evaluation(evals_result)]
    )
    lgb_probs = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    # --- PRカーブで複数閾値自動評価 ---
    from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score, recall_score, average_precision_score, confusion_matrix, classification_report
    precisions, recalls, thresholds = precision_recall_curve(y_test, lgb_probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    pr_auc = average_precision_score(y_test, lgb_probs)
    best_idx_f1 = f1s.argmax()
    best_threshold_f1 = thresholds[best_idx_f1] if best_idx_f1 < len(thresholds) else 0.5
    for th, label in zip([0.5, best_threshold_f1], ['0.5','F1最大']):
        preds = (lgb_probs >= th).astype(int)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        recall_v = recall_score(y_test, preds)
        pr_auc_v = average_precision_score(y_test, lgb_probs)
        cm = confusion_matrix(y_test, preds)
        print(f'--- 評価結果（{label}, 閾値={th:.3f}） ---')
        print(f'Accuracy: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Recall: {recall_v:.4f}')
        print(f'PR-AUC: {pr_auc_v:.4f}')
        print(f'Confusion Matrix:\n{cm}')
        print(classification_report(y_test, preds))

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    print(f'【実行所要時間】{end_time - start_time:.2f}秒')