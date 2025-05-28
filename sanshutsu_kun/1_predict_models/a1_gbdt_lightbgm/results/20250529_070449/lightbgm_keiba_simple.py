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

def fetch_data():
    print('=== fetch_data関数 実行開始 ===')
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
        group = group.sort_values('RACE_DATE')
        prev_race_date = None
        syussou_count = 0
        for idx, row in group.iterrows():
            syussou_count += 1
            if prev_race_date is not None:
                days = (row['RACE_DATE'] - prev_race_date).days
            else:
                days = 9999
            df.loc[idx, 'syussou_count'] = syussou_count
            df.loc[idx, 'prev_days'] = days
            if idx > group.index.min():
                prev_row = group.loc[idx-1]
                df.loc[idx, 'prev_race_date'] = prev_row['RACE_DATE']
                df.loc[idx, 'prev_chaku'] = prev_row['KAKUTEI_CHAKUJUN_norm'] if 'KAKUTEI_CHAKUJUN_norm' in prev_row else 0
                df.loc[idx, 'prev_ninki'] = prev_row['TANSHO_NINKIJUN'] if 'TANSHO_NINKIJUN' in prev_row else 0
                df.loc[idx, 'prev_bataiju'] = prev_row['BATAIJU'] if 'BATAIJU' in prev_row else 0
                df.loc[idx, 'prev_kishu'] = prev_row['KISHU_CODE'] if 'KISHU_CODE' in prev_row else 0
                df.loc[idx, 'prev_course'] = prev_row['RACE_CODE'] if 'RACE_CODE' in prev_row else 0
            prev_race_date = row['RACE_DATE']
    # --- 特徴量生成 ---
    feature_rows = []
    print('特徴量生成開始')
    for idx, row in df.iterrows():
        print_progress_bar(idx+1, len(df), bar_length=100, prefix='進捗', suffix='')
        uma = row['KETTO_TOROKU_BANGO']
        race_code = row['RACE_CODE']
        umaban = row['UMABAN']
        # オッズ
        odds_t = odds_tansho.get((race_code, umaban), (0, 0))
        odds_f = odds_fukusho.get((race_code, umaban), (0, 0, 0))
        # 血統
        keito = keito_dict.get(uma, ('', ''))
        # 馬体重
        bataiju = bataiju_dict.get((race_code, umaban), 0)
        # 騎手成績
        kishu_win, kishu_ren, kishu_fuku = kishu_seiseki.get(row['KISHU_CODE'], (0, 0, 0))
        # 馬場・コース・距離・天候
        race_shosai = race_shosai_dict.get(race_code, {'KYORI':0,'TRACK_CODE':'','SHIBA_BABAJOTAI_CODE':'','DIRT_BABAJOTAI_CODE':'','TENKO_CODE':'','COURSE_KUBUN':''})
        # 既存特徴量＋追加
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
            'target': row['target'],
            'RACE_CODE': row['RACE_CODE']
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
    float_cols = [col for col in feature_df.columns if col not in ['RACE_CODE']]
    feature_df[float_cols] = feature_df[float_cols].astype(float)
    n1 = (feature_df['target'] == 1).sum()
    n0 = (feature_df['target'] == 0).sum()
    print(f'target=1(3着以内): {n1}件, target=0: {n0}件')
    if n1 < 100:
        print('警告: 3着以内データが極端に少ないです。抽出期間や条件を見直してください。')
    feature_cols = [
        'BAREI','SEIBETSU_CODE','BATAIJU','FUTAN_JURYO','KISHU_CODE','CHOKYOSHI_CODE','WAKUBAN','TANSHO_NINKIJUN',
        'TANSHO_ODDS','TANSHO_ODDS_NINKI','FUKUSHO_ODDS_SAITEI','FUKUSHO_ODDS_SAIKOU','FUKUSHO_ODDS_NINKI',
        'KEITO_ID','KEITO_MEI','BATAIJU_LATEST','KISHU_WIN_RATE','KISHU_RENTAI_RATE','KISHU_FUKUSHO_RATE',
        'KYORI','TRACK_CODE','SHIBA_BABAJOTAI_CODE','DIRT_BABAJOTAI_CODE','TENKO_CODE','COURSE_KUBUN',
        'prev_chaku','prev_ninki','prev_bataiju','prev_kishu','prev_course','prev_days','syussou_count'
    ]
    feature_cols = [col for col in feature_cols if col in feature_df.columns]
    X = feature_df[feature_cols]
    y = feature_df['target']
    # --- 負例ダウンサンプリング ---
    pos_idx = y[y==1].index
    neg_idx = y[y==0].sample(n=len(pos_idx), random_state=42)
    idx = pos_idx.union(neg_idx)
    X = X.loc[idx].reset_index(drop=True)
    y = y.loc[idx].reset_index(drop=True)
    end_time = time.time()
    print(f'【fetch_data関数 実行所要時間】{end_time - start_time:.2f}秒')
    return X, y, feature_df, n0, n1

def main():
    # run_dir作成＋スクリプト保存
    run_dir = save_script_and_make_run_dir('sanshutsu_kun/1_predict_models/a1_gbdt_lightbgm/results')
    print('=== LightGBM/XGBoost競馬3着以内予測 実行開始 ===')
    X, y, df, n0, n1 = fetch_data()
    if len(X) == 0:
        print('データがありません')
        return
    # GroupKFoldでRACE_CODE単位で分割
    groups = df['RACE_CODE']
    gkf = GroupKFold(n_splits=5)
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        break  # 1fold目のみで実行
    # --- SMOTEでオーバーサンプリング（比率強化） ---
    smote = SMOTE(sampling_strategy=1.0, random_state=42)  # 正例が全体の50%になるまで水増し
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f'SMOTE後: {np.bincount(y_train_res)}')
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
        'scale_pos_weight': n0/n1*2 if n1>0 else 1.0,
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
        'focal_loss': True,  # focal loss有効化（LightGBM>=4.0で有効）
    }
    # --- LightGBM学習（学習曲線記録用） ---
    evals_result = {}
    lgb_model = lgb.train(
        lgb_params, lgb.Dataset(X_train_res, y_train_res), valid_sets=[lgb.Dataset(X_train_res, y_train_res), lgb.Dataset(X_test, y_test)],
        valid_names=['train','valid'], num_boost_round=200,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20), lgb.record_evaluation(evals_result)]
    )
    # --- 学習曲線の描画・保存 ---
    plt.figure(figsize=(8,5))
    plt.plot(evals_result['train']['binary_logloss'], label='train')
    plt.plot(evals_result['valid']['binary_logloss'], label='valid')
    plt.xlabel('Iteration')
    plt.ylabel('Binary Logloss')
    plt.title('Learning Curve (LightGBM)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'learning_curve.png'))
    plt.close()
    lgb_probs = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    # --- XGBoost学習 ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.03, 
        reg_alpha=1.0, reg_lambda=1.0, use_label_encoder=False, eval_metric='logloss', verbosity=0,
        tree_method='hist',
        objective='binary:logistic',
        booster='gbtree',
        n_jobs=4,
        random_state=42,
        enable_categorical=True,
        importance_type='gain',
        grow_policy='lossguide',
        max_bin=255,
        subsample=0.8,
        colsample_bytree=0.9,
        min_child_weight=1e-3,
        max_leaves=63,
        gamma=0.01,
        scale_pos_weight=n0/n1*2 if n1>0 else 1.0
    )
    xgb_model.fit(X_train_res, y_train_res)
    xgb_probs = xgb_model.predict_proba(X_test)[:,1]
    # --- CatBoost学習 ---
    cat_model = CatBoostClassifier(iterations=200, depth=7, learning_rate=0.03, l2_leaf_reg=1.0, verbose=0, random_seed=42, class_weights=[1, n0/n1*2 if n1>0 else 1.0])
    cat_model.fit(X_train_res, y_train_res)
    cat_probs = cat_model.predict_proba(X_test)[:,1]
    # --- アンサンブル（平均） ---
    probs = (lgb_probs + xgb_probs + cat_probs) / 3
    # --- PRカーブで複数閾値自動評価 ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    pr_auc = average_precision_score(y_test, probs)
    best_idx_f1 = f1s.argmax()
    best_threshold_f1 = thresholds[best_idx_f1] if best_idx_f1 < len(thresholds) else 0.5
    # recall重視（0.5以上）
    mask_r = recalls >= 0.5
    best_idx_r = (f1s * mask_r).argmax() if mask_r.sum() > 0 else best_idx_f1
    best_threshold_r = thresholds[best_idx_r] if best_idx_r < len(thresholds) else 0.5
    # precision重視（0.8以上）
    mask_p = precisions >= 0.8
    best_idx_p = (f1s * mask_p).argmax() if mask_p.sum() > 0 else best_idx_f1
    best_threshold_p = thresholds[best_idx_p] if best_idx_p < len(thresholds) else 0.5
    # --- 複数閾値で自動評価 ---
    thresholds_to_eval = [0.5, best_threshold_f1, best_threshold_r, best_threshold_p]
    for th, label in zip(thresholds_to_eval, ['0.5','F1最大','recall重視','precision重視']):
        preds = (probs >= th).astype(int)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        recall_v = recall_score(y_test, preds)
        pr_auc_v = average_precision_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        print(f'--- 評価結果（{label}, 閾値={th:.3f}） ---')
        print(f'Accuracy: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Recall: {recall_v:.4f}')
        print(f'PR-AUC: {pr_auc_v:.4f}')
        print(f'Confusion Matrix:\n{cm}')
        print(classification_report(y_test, preds))
    # --- SHAP値で特徴量重要度グラフ ---
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False, plot_type='bar')
    shap_path = os.path.join(run_dir, 'shap_feature_importance.png')
    plt.savefig(shap_path, bbox_inches='tight')
    plt.close()
    # グラフ
    graph_paths = plot_and_save_graphs(y_test, preds, probs, [], [], run_dir)
    # 画像base64化
    images_b64 = []
    for path in graph_paths:
        with open(path, 'rb') as imgf:
            b64img = base64.b64encode(imgf.read()).decode('utf-8')
            images_b64.append((os.path.basename(path), b64img))
    # GPT評価
    prompt = f"""
競馬の3着以内に入る馬の特徴をLightGBM+XGBoost+CatBoostアンサンブルで学習したモデルの評価結果です。\n\n最適閾値: {best_threshold_f1:.4f}\n精度: {acc:.4f}\nF1スコア: {f1:.4f}\n詳細:\n{classification_report(y_test, preds)}\n\n以下のグラフ画像（base64エンコード済み）も参考にして、モデル改善のためのアドバイスを日本語で簡潔に出力してください。\n"""
    gpt_advice = ask_gpt41(prompt, images_b64)
    # 結果保存
    with open(os.path.join(run_dir, 'eval_and_gpt_advice.txt'), 'w', encoding='utf-8') as f:
        f.write('--- 評価結果 ---\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(classification_report(y_test, preds) + '\n')
        f.write('\n--- GPT-4.1からの改善アドバイス ---\n')
        f.write(gpt_advice + '\n')
    # モデル保存
    lgb_model.save_model(os.path.join(run_dir, 'lgb_model.txt'))
    xgb_model.save_model(os.path.join(run_dir, 'xgb_model.json'))
    cat_model.save_model(os.path.join(run_dir, 'cat_model.cbm'))
    # --- SHAP値で寄与の低い特徴量を自動除外し再学習 ---
    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    threshold = np.percentile(shap_abs_mean, 20)  # 下位20%を除外
    selected_features = [col for col, val in zip(X_test.columns, shap_abs_mean) if val > threshold]
    print(f'SHAP値で選択された特徴量数: {len(selected_features)} / {len(X_test.columns)}')
    # 再学習
    X_train_sel = X_train_res[selected_features]
    X_test_sel = X_test[selected_features]
    lgb_model_sel = lgb.train(
        lgb_params, lgb.Dataset(X_train_sel, y_train_res), valid_sets=[lgb.Dataset(X_test_sel, y_test)], num_boost_round=200,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)]
    )
    lgb_probs_sel = lgb_model_sel.predict(X_test_sel, num_iteration=lgb_model_sel.best_iteration)
    # アンサンブルも再計算
    xgb_model_sel = xgb.XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.03, 
        reg_alpha=1.0, reg_lambda=1.0, use_label_encoder=False, eval_metric='logloss', verbosity=0,
        tree_method='hist',
        objective='binary:logistic',
        booster='gbtree',
        n_jobs=4,
        random_state=42,
        enable_categorical=True,
        importance_type='gain',
        grow_policy='lossguide',
        max_bin=255,
        subsample=0.8,
        colsample_bytree=0.9,
        min_child_weight=1e-3,
        max_leaves=63,
        gamma=0.01,
        scale_pos_weight=n0/n1*2 if n1>0 else 1.0
    )
    xgb_model_sel.fit(X_train_sel, y_train_res)
    xgb_probs_sel = xgb_model_sel.predict_proba(X_test_sel)[:,1]
    cat_model_sel = CatBoostClassifier(iterations=200, depth=7, learning_rate=0.03, l2_leaf_reg=1.0, verbose=0, random_seed=42, class_weights=[1, n0/n1*2 if n1>0 else 1.0])
    cat_model_sel.fit(X_train_sel, y_train_res)
    cat_probs_sel = cat_model_sel.predict_proba(X_test_sel)[:,1]
    probs_sel = (lgb_probs_sel + xgb_probs_sel + cat_probs_sel) / 3
    # 再評価
    precisions_sel, recalls_sel, thresholds_sel = precision_recall_curve(y_test, probs_sel)
    f1s_sel = 2 * precisions_sel * recalls_sel / (precisions_sel + recalls_sel + 1e-8)
    mask_sel = precisions_sel >= 0.8
    if mask_sel.sum() > 0:
        best_idx_sel = (f1s_sel * mask_sel).argmax()
    else:
        best_idx_sel = f1s_sel.argmax()
    best_threshold_sel = thresholds_sel[best_idx_sel] if best_idx_sel < len(thresholds_sel) else 0.5
    preds_sel = (probs_sel >= best_threshold_sel).astype(int)
    acc_sel = accuracy_score(y_test, preds_sel)
    f1_sel = f1_score(y_test, preds_sel)
    report_sel = classification_report(y_test, preds_sel)
    print(f'--- SHAP特徴量選択後の評価（閾値={best_threshold_sel:.2f}）---')
    print(f'Accuracy: {acc_sel:.4f}')
    print(f'F1 Score: {f1_sel:.4f}')
    print(report_sel)
    # 保存
    with open(os.path.join(run_dir, "eval_and_gpt_advice_shap_selected.txt"), 'w', encoding='utf-8') as f:
        f.write('--- SHAP特徴量選択後の評価 ---\n')
        f.write(f'Accuracy: {acc_sel:.4f}\n')
        f.write(f'F1 Score: {f1_sel:.4f}\n')
        f.write(report_sel + '\n')
    # --- Stackingアンサンブル（ロジスティック回帰/MLP） ---
    # 各モデルの予測値を特徴量化
    stack_X_train = np.vstack([
        lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration),
        xgb_model.predict_proba(X_train)[:,1],
        cat_model.predict_proba(X_train)[:,1]
    ]).T
    stack_X_test = np.vstack([lgb_probs, xgb_probs, cat_probs]).T
    # ロジスティック回帰
    stack_lr = LogisticRegression(max_iter=200)
    stack_lr.fit(stack_X_train, y_train)
    stack_probs_lr = stack_lr.predict_proba(stack_X_test)[:,1]
    # MLP
    stack_mlp = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=200, random_state=42)
    stack_mlp.fit(stack_X_train, y_train)
    stack_probs_mlp = stack_mlp.predict_proba(stack_X_test)[:,1]
    # 既存アンサンブルとstackingの平均
    probs_stacked = (probs + stack_probs_lr + stack_probs_mlp) / 3
    # --- PRカーブから最適閾値再探索 ---
    precisions_s, recalls_s, thresholds_s = precision_recall_curve(y_test, probs_stacked)
    f1s_s = 2 * precisions_s * recalls_s / (precisions_s + recalls_s + 1e-8)
    mask_s = precisions_s >= 0.8
    if mask_s.sum() > 0:
        best_idx_s = (f1s_s * mask_s).argmax()
    else:
        best_idx_s = f1s_s.argmax()
    best_threshold_s = thresholds_s[best_idx_s] if best_idx_s < len(thresholds_s) else 0.5
    preds_s = (probs_stacked >= best_threshold_s).astype(int)
    acc_s = accuracy_score(y_test, preds_s)
    f1_s = f1_score(y_test, preds_s)
    report_s = classification_report(y_test, preds_s)
    print(f'--- Stackingアンサンブル評価（閾値={best_threshold_s:.2f}）---')
    print(f'Accuracy: {acc_s:.4f}')
    print(f'F1 Score: {f1_s:.4f}')
    print(report_s)
    # 保存
    with open(os.path.join(run_dir, "eval_and_gpt_advice_stacking.txt"), 'w', encoding='utf-8') as f:
        f.write('--- Stackingアンサンブル評価 ---\n')
        f.write(f'Accuracy: {acc_s:.4f}\n')
        f.write(f'F1 Score: {f1_s:.4f}\n')
        f.write(report_s + '\n')
    # --- 誤分類サンプルの特徴量分布可視化 ---
    # FN（False Negative: 3着以内を外した）
    fn_idx = (y_test == 1) & (preds == 0)
    fp_idx = (y_test == 0) & (preds == 1)
    fn_df = X_test[fn_idx]
    fp_df = X_test[fp_idx]
    # 主要特徴量（SHAP値上位10個）
    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    top10_idx = np.argsort(shap_abs_mean)[::-1][:10]
    top10_features = X_test.columns[top10_idx]
    for feat in top10_features:
        plt.figure(figsize=(8,4))
        sns.kdeplot(X_test[feat], label='all', color='gray')
        if len(fn_df)>0:
            sns.kdeplot(fn_df[feat], label='FN', color='red')
        if len(fp_df)>0:
            sns.kdeplot(fp_df[feat], label='FP', color='blue')
        if jp_font:
            plt.title(f'特徴量[{feat}]の分布（全体/FN/FP）')
        else:
            plt.title(f'特徴量[{feat}]の分布（全体/FN/FP）')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f'feature_dist_{feat}.png'), bbox_inches='tight', dpi=120)
        plt.close()

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    print(f'【実行所要時間】{end_time - start_time:.2f}秒')