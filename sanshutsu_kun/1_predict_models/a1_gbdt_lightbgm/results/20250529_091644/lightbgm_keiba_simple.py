import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
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
from sklearn.model_selection import KFold

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
target_days = 360 * 3  # 例: 直近90日分

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
            df.loc[idx, 'syussou_count'] = int(syussou_count)
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
            df.loc[idx, 'chaku3mean'] = np.mean(chaku_hist[-3:]) if chaku_hist[-3:] else 99
            df.loc[idx, 'ninki3mean'] = np.mean(ninki_hist[-3:]) if ninki_hist[-3:] else 99
    # --- 連対率（2着以内率） ---
    df['rentai_count'] = 0
    df['race_count'] = 0
    for uma, group in df.groupby('KETTO_TOROKU_BANGO'):
        group = group.sort_values('RACE_DATE').reset_index()
        rentai_hist = []
        for i, row in group.iterrows():
            idx = row['index']
            if i > 0:
                chaku = int(row['KAKUTEI_CHAKUJUN_norm']) if str(row['KAKUTEI_CHAKUJUN_norm']).isdigit() else 99
                rentai_hist.append(1 if chaku <= 2 else 0)
            df.loc[idx, 'rentai_count'] = sum(rentai_hist[-3:])
            df.loc[idx, 'race_count'] = len(rentai_hist[-3:])
    df['rentai_rate3'] = df['rentai_count'] / df['race_count'].replace(0, 1)
    # --- 直近成績の標準偏差 ---
    df['chaku_std3'] = 99
    for uma, group in df.groupby('KETTO_TOROKU_BANGO'):
        group = group.sort_values('RACE_DATE').reset_index()
        chaku_hist = []
        for i, row in group.iterrows():
            idx = row['index']
            if i > 0:
                chaku = int(row['KAKUTEI_CHAKUJUN_norm']) if str(row['KAKUTEI_CHAKUJUN_norm']).isdigit() else 99
                chaku_hist.append(chaku)
            df.loc[idx, 'chaku_std3'] = np.std(chaku_hist[-3:]) if chaku_hist[-3:] else 99
    # --- 馬場状態×脚質の組み合わせ ---
    if 'BABA_CODE' in df.columns and 'KAKUSITU_CODE' in df.columns:
        df['BABA_KAKUSITU'] = df['BABA_CODE'].astype(str) + '_' + df['KAKUSITU_CODE'].astype(str)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['BABA_KAKUSITU'] = le.fit_transform(df['BABA_KAKUSITU'])
    # --- 馬場状態・脚質・馬体重変化 ---
    if 'BABA_CODE' in df.columns:
        df['BABA_CODE'] = df['BABA_CODE'].fillna('不明')
    if 'KAKUSITU_CODE' in df.columns:
        df['KAKUSITU_CODE'] = df['KAKUSITU_CODE'].fillna('不明')
    if 'BATAIJU' in df.columns:
        df['BATAIJU'] = pd.to_numeric(df['BATAIJU'], errors='coerce').fillna(0)
        df['BATAIJU_DIFF'] = df.groupby('KETTO_TOROKU_BANGO')['BATAIJU'].diff().fillna(0)
    # --- 騎手・厩舎の直近3走成績平均 ---
    for col in ['KISHU_CODE', 'CHOKYOSHI_CODE']:
        df[f'{col}_3chaku_mean'] = 99
        for code, group in df.groupby(col):
            group = group.sort_values('RACE_DATE').reset_index()
            chaku_hist = []
            for i, row in group.iterrows():
                idx = row['index']
                if i > 0:
                    chaku_hist.append(int(row['KAKUTEI_CHAKUJUN_norm']) if str(row['KAKUTEI_CHAKUJUN_norm']).isdigit() else 99)
                df.loc[idx, f'{col}_3chaku_mean'] = np.mean(chaku_hist[-3:]) if chaku_hist[-3:] else 99
    # --- 馬体重変化率・人気変化率 ---
    df['BATAIJU_DIFF_RATE'] = 0.0
    df['NINKI_DIFF_RATE'] = 0.0
    for uma, group in df.groupby('KETTO_TOROKU_BANGO'):
        group = group.sort_values('RACE_DATE').reset_index()
        prev_bataiju = None
        prev_ninki = None
        for i, row in group.iterrows():
            idx = row['index']
            cur_bataiju = float(row['BATAIJU']) if row['BATAIJU'] != '' else 0.0
            cur_ninki = float(row['TANSHO_NINKIJUN']) if row['TANSHO_NINKIJUN'] != '' else 0.0
            if i > 0:
                if prev_bataiju is not None and prev_bataiju != 0:
                    df.loc[idx, 'BATAIJU_DIFF_RATE'] = (cur_bataiju - prev_bataiju) / prev_bataiju
                if prev_ninki is not None and prev_ninki != 0:
                    df.loc[idx, 'NINKI_DIFF_RATE'] = (cur_ninki - prev_ninki) / prev_ninki
            prev_bataiju = cur_bataiju
            prev_ninki = cur_ninki
    # --- Target Encoding ---
    for col in ['KISHU_CODE', 'CHOKYOSHI_CODE', 'COURSE_CODE', 'WAKUBAN']:
        if col in df.columns:
            te_col = f'{col}_te'
            df[te_col] = 0.0
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(df):
                means = df.iloc[train_idx].groupby(col)['target'].mean()
                df.iloc[val_idx, df.columns.get_loc(te_col)] = df.iloc[val_idx][col].map(means).fillna(0)
    # --- 特徴量リスト ---
    features = [
        'BAREI', 'SEIBETSU_CODE', 'BATAIJU', 'BATAIJU_DIFF', 'BATAIJU_DIFF_RATE', 'FUTAN_JURYO', 'TANSHO_NINKIJUN', 'NINKI_DIFF_RATE',
        'BABA_CODE', 'KAKUSITU_CODE', 'BABA_KAKUSITU', 'chaku3mean', 'ninki3mean', 'chaku_std3', 'rentai_rate3',
        'KISHU_CODE_te', 'CHOKYOSHI_CODE_te', 'COURSE_CODE_te', 'WAKUBAN_te',
        'KISHU_CODE_3chaku_mean', 'CHOKYOSHI_CODE_3chaku_mean'
    ]
    features = [f for f in features if f in df.columns]
    # 型変換
    for col in ['BAREI', 'SEIBETSU_CODE', 'FUTAN_JURYO', 'TANSHO_NINKIJUN', 'BATAIJU', 'BATAIJU_DIFF', 'BATAIJU_DIFF_RATE', 'NINKI_DIFF_RATE']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # カテゴリ変数はLabelEncoder
    for col in ['BABA_CODE', 'KAKUSITU_CODE']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    X = df[features].copy().reset_index(drop=True)
    y = df['target'].reset_index(drop=True)
    # --- 開催日順で必ずソート ---
    if 'RACE_DATE' in df.columns:
        df = df.sort_values('RACE_DATE').reset_index(drop=True)
    # --- 時系列分割 ---
    n = len(X)
    split = int(n * (1 - test_ratio))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print('【分割後】y_train分布:', y_train.value_counts())
    print('【分割後】y_test分布:', y_test.value_counts())
    # --- データ抽出直後のtarget分布チェック ---
    print('【抽出直後】target値の分布:', df['target'].value_counts())
    if (df['target'] == 1).sum() == 0 or (df['target'] == 0).sum() == 0:
        print('警告: target=1または0が存在しません。データ抽出条件を見直してください。')
        return None, None, None, None, 0, 0, df
    # --- 特徴量生成後のtarget分布チェック ---
    print('【特徴量生成後】target値の分布:', df['target'].value_counts())
    if (df['target'] == 1).sum() == 0 or (df['target'] == 0).sum() == 0:
        print('警告: target=1または0が存在しません。特徴量生成で消失しています。')
        return None, None, None, None, 0, 0, df
    return X_train, X_test, y_train, y_test, (y==0).sum(), (y==1).sum(), df

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
        'scale_pos_weight': (y_train == 0).sum() / max((y_train == 1).sum(), 1),
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
    # --- LightGBM学習（1回目） ---
    lgb_model = lgb.train(
        lgb_params,
        lgb.Dataset(X_train, y_train),
        valid_sets=[lgb.Dataset(X_train, y_train), lgb.Dataset(X_test, y_test)],
        valid_names=['train', 'valid'],
        num_boost_round=200,
        callbacks=[
            lgb.early_stopping(20),
            lgb.log_evaluation(20),
            lgb.record_evaluation(evals_result)
        ]
    )
    # --- 特徴量重要度で下位30%を除外 ---
    importances = lgb_model.feature_importance(importance_type='gain')
    feature_names = X_train.columns
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False)
    n_remove = int(len(imp_df) * 0.3)
    if n_remove > 0:
        remove_features = imp_df.tail(n_remove)['feature'].tolist()
        print(f'重要度下位30%の特徴量を除外: {remove_features}')
        X_train2 = X_train.drop(columns=remove_features)
        X_test2 = X_test.drop(columns=remove_features)
        # --- LightGBM再学習 ---
        lgb_model2 = lgb.train(
            lgb_params,
            lgb.Dataset(X_train2, y_train),
            valid_sets=[lgb.Dataset(X_train2, y_train), lgb.Dataset(X_test2, y_test)],
            valid_names=['train', 'valid'],
            num_boost_round=200,
            callbacks=[
                lgb.early_stopping(20),
                lgb.log_evaluation(20),
                lgb.record_evaluation(evals_result)
            ]
        )
        lgb_probs2 = lgb_model2.predict(X_test2, num_iteration=lgb_model2.best_iteration)
        # --- PRカーブでRecall重視閾値自動探索 ---
        precisions, recalls, thresholds = precision_recall_curve(y_test, lgb_probs2)
        recall_target = 0.85
        recall_idx = np.where(recalls >= recall_target)[0]
        if len(recall_idx) > 0:
            best_idx_recall = recall_idx[-1]
            best_threshold_recall = thresholds[best_idx_recall] if best_idx_recall < len(thresholds) else 0.5
            print(f'[再学習] Recall>={recall_target}となる最小閾値: {best_threshold_recall:.3f}')
            preds_recall = (lgb_probs2 >= best_threshold_recall).astype(int)
            acc = accuracy_score(y_test, preds_recall)
            f1 = f1_score(y_test, preds_recall)
            recall_v = recall_score(y_test, preds_recall)
            pr_auc = average_precision_score(y_test, lgb_probs2)
            print(f'[再学習] --- Recall重視評価（閾値={best_threshold_recall:.3f}） ---')
            print(f'Accuracy: {acc:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print(f'Recall: {recall_v:.4f}')
            print(f'PR-AUC: {pr_auc:.4f}')
            print('Confusion Matrix:')
            print(confusion_matrix(y_test, preds_recall))
            print(classification_report(y_test, preds_recall))
            # --- グラフ・GPTアドバイス・保存も同様に実施（省略: 既存処理を流用）
    # --- 学習曲線データ抽出 ---
    train_losses = evals_result['train']['binary_logloss'] if 'binary_logloss' in evals_result['train'] else []
    valid_losses = evals_result['valid']['binary_logloss'] if 'binary_logloss' in evals_result['valid'] else []
    train_accuracies = []
    valid_accuracies = []
    if 'train' in evals_result and 'valid' in evals_result:
        for i in range(len(train_losses)):
            train_pred = lgb_model.predict(X_train, num_iteration=i+1)
            valid_pred = lgb_model.predict(X_test, num_iteration=i+1)
            train_accuracies.append(accuracy_score(y_train, (train_pred>=0.5).astype(int)))
            valid_accuracies.append(accuracy_score(y_test, (valid_pred>=0.5).astype(int)))
    lgb_probs = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    # --- XGBoostアンサンブル ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=20)
    xgb_probs = xgb_model.predict_proba(X_test)[:,1]
    # --- LightGBM+XGBoost平均予測 ---
    avg_probs = (lgb_probs + xgb_probs) / 2
    # --- PRカーブでRecall重視閾値自動探索 ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, avg_probs)
    recall_target = 0.85
    recall_idx = np.where(recalls >= recall_target)[0]
    if len(recall_idx) > 0:
        best_idx_recall = recall_idx[-1]
        best_threshold_recall = thresholds[best_idx_recall] if best_idx_recall < len(thresholds) else 0.5
        print(f'[アンサンブル] Recall>={recall_target}となる最小閾値: {best_threshold_recall:.3f}')
        preds_recall = (avg_probs >= best_threshold_recall).astype(int)
        acc = accuracy_score(y_test, preds_recall)
        f1 = f1_score(y_test, preds_recall)
        recall_v = recall_score(y_test, preds_recall)
        pr_auc = average_precision_score(y_test, avg_probs)
        print(f'[アンサンブル] --- Recall重視評価（閾値={best_threshold_recall:.3f}） ---')
        print(f'Accuracy: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Recall: {recall_v:.4f}')
        print(f'PR-AUC: {pr_auc:.4f}')
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, preds_recall))
        print(classification_report(y_test, preds_recall))
        # --- グラフ・GPTアドバイス・保存も同様に実施（省略: 既存処理を流用）
        graph_paths = plot_and_save_graphs(y_test, preds_recall, avg_probs, train_losses, train_accuracies, run_dir)
        # 画像base64化
        images_b64 = []
        for path in graph_paths:
            if path and os.path.exists(path):
                with open(path, 'rb') as imgf:
                    b64img = base64.b64encode(imgf.read()).decode('utf-8')
                    images_b64.append((os.path.basename(path), b64img))
        prompt = f"""
競馬の3着以内に入る馬の特徴をLightGBMで学習したモデルの評価結果です。\n\nRecall重視閾値: {best_threshold_recall:.4f}\n精度: {acc:.4f}\nF1スコア: {f1:.4f}\n詳細:\n{classification_report(y_test, preds_recall)}\n\n"""
        if images_b64:
            prompt += "以下のグラフ画像（base64エンコード済み）も参考にして、モデル改善のためのアドバイスを日本語で簡潔に出力してください。\n"
        else:
            prompt += "グラフ画像は出力されませんでした。モデル改善のためのアドバイスを日本語で簡潔に出力してください。\n"
        gpt_advice = ask_gpt41(prompt, images_b64)
        # --- 結果保存 ---
        with open(os.path.join(run_dir, f'eval_and_gpt_advice_Recall.txt'), 'w', encoding='utf-8') as f:
            f.write('--- 評価結果 ---\n')
            f.write(f'Accuracy: {acc:.4f}\n')
            f.write(f'F1 Score: {f1:.4f}\n')
            f.write(classification_report(y_test, preds_recall) + '\n')
            f.write('\n--- GPT-4.1からの改善アドバイス ---\n')
            f.write(gpt_advice + '\n')
    # --- モデル保存 ---
    lgb_model.save_model(os.path.join(run_dir, 'lgb_model.txt'))

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    print(f'【実行所要時間】{end_time - start_time:.2f}秒')