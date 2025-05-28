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
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
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
from sanshutsu_kun.modules.plot_utils import plot_and_save_graphs
from sanshutsu_kun.modules.gpt_utils import ask_gpt41
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid

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
    three_years_ago = today - timedelta(days=365*3)
    date_min = three_years_ago.strftime('%Y%m%d')
    # --- メインデータ ---
    print('メインデータ取得開始')
    query = '''SELECT BAREI, SEIBETSU_CODE, BATAIJU, FUTAN_JURYO, TANSHO_NINKIJUN, KISHU_CODE, CHOKYOSHI_CODE, KETTO_TOROKU_BANGO, RACE_CODE, KAISAI_NEN, KAISAI_GAPPI, KAKUTEI_CHAKUJUN FROM umagoto_race_joho WHERE CONCAT(KAISAI_NEN, LPAD(KAISAI_GAPPI, 4, '0')) >= %s AND KISHU_CODE IS NOT NULL AND RACE_CODE IS NOT NULL AND KETTO_TOROKU_BANGO IS NOT NULL AND CHOKYOSHI_CODE IS NOT NULL AND KAISAI_NEN IS NOT NULL AND KAISAI_GAPPI IS NOT NULL'''
    cursor.execute(query, (date_min,))
    rows = cursor.fetchall()
    columns = ['BAREI','SEIBETSU_CODE','BATAIJU','FUTAN_JURYO','TANSHO_NINKIJUN','KISHU_CODE','CHOKYOSHI_CODE','KETTO_TOROKU_BANGO','RACE_CODE','KAISAI_NEN','KAISAI_GAPPI','KAKUTEI_CHAKUJUN']
    df = pd.DataFrame(rows, columns=columns)
    # --- レース日付 ---
    df['RACE_DATE'] = df['KAISAI_NEN'].astype(str) + df['KAISAI_GAPPI'].astype(str).str.zfill(4)
    df['RACE_DATE'] = pd.to_datetime(df['RACE_DATE'], format='%Y%m%d', errors='coerce')
    df = df.sort_values(['RACE_DATE', 'RACE_CODE', 'KETTO_TOROKU_BANGO']).reset_index(drop=True)
    # --- ラベル ---
    df['KAKUTEI_CHAKUJUN_norm'] = df['KAKUTEI_CHAKUJUN'].apply(lambda x: str(x).strip().replace('　','').replace(' ','').translate(str.maketrans('０１２３４５６７８９', '0123456789')).lstrip('0'))
    df['target'] = df['KAKUTEI_CHAKUJUN_norm'].isin(['1','2','3']).astype(int)
    # --- race_shosai ---
    print('race_shosai取得開始')
    cursor.execute('SELECT RACE_CODE, KYORI, TRACK_CODE, SHUSSO_TOSU FROM race_shosai')
    race_shosai = {row[0]: (float(row[1]) if row[1] else 0, int(row[2]) if row[2] else 0, int(row[3]) if row[3] else 0) for row in cursor.fetchall()}
    # --- tenko_baba_jotai ---
    print('tenko_baba_jotai取得開始')
    cursor.execute('SELECT RACE_CODE, TENKO_JOTAI_GENZAI, BABA_JOTAI_SHIBA_GENZAI, BABA_JOTAI_DIRT_GENZAI FROM tenko_baba_jotai')
    tenko_baba = {row[0]: (row[1], row[2], row[3]) for row in cursor.fetchall()}
    # --- 特徴量生成（未来情報ゼロ） ---
    feature_rows = []
    print('特徴量生成開始')
    grouped = df.groupby('KETTO_TOROKU_BANGO')
    for idx, row in df.iterrows():
        uma = row['KETTO_TOROKU_BANGO']
        race_date = row['RACE_DATE']
        race_code = row['RACE_CODE']
        # 馬の過去データ
        print(f'馬の過去データ取得開始: {uma}')
        uma_hist = grouped.get_group(uma)
        uma_hist = uma_hist[uma_hist['RACE_DATE'] < race_date]
        # 騎手の過去データ
        print(f'騎手の過去データ取得開始: {row["KISHU_CODE"]}')
        kishu_hist = df[(df['KISHU_CODE'] == row['KISHU_CODE']) & (df['RACE_DATE'] < race_date)]
        # 調教師の過去データ
        print(f'調教師の過去データ取得開始: {row["CHOKYOSHI_CODE"]}')
        chokyoshi_hist = df[(df['CHOKYOSHI_CODE'] == row['CHOKYOSHI_CODE']) & (df['RACE_DATE'] < race_date)]
        # コース・距離の過去データ
        print(f'コース・距離の過去データ取得開始: {race_code}')
        kyori = race_shosai.get(race_code, (0,0,0))[0]
        track_code = race_shosai.get(race_code, (0,0,0))[1]
        course_id = str(race_code)[:6]
        uma_course_hist = uma_hist[uma_hist['RACE_CODE'].astype(str).str[:6] == course_id]
        uma_kyori_hist = uma_hist[(uma_hist['RACE_CODE'].map(lambda x: race_shosai.get(x, (0,0,0))[0]) // 100 * 100) == (kyori // 100 * 100)]
        # --- 特徴量 ---
        feats = {
            'BAREI': row['BAREI'],
            'SEIBETSU_CODE': row['SEIBETSU_CODE'],
            'BATAIJU': row['BATAIJU'],
            'FUTAN_JURYO': row['FUTAN_JURYO'],
            'KISHU_CODE': row['KISHU_CODE'],
            'CHOKYOSHI_CODE': row['CHOKYOSHI_CODE'],
            'KYORI': kyori,
            'TRACK_CODE': track_code,
            'SHUSSO_TOSU': race_shosai.get(race_code, (0,0,0))[2],
            'RACE_KANKAKU': (row['RACE_DATE'] - uma_hist['RACE_DATE'].max()).days if not uma_hist.empty else 0,
            'SHUSSO_KAISU': len(uma_hist),
            'TENKO_JOTAI_GENZAI': tenko_baba.get(race_code, (None, None, None))[0],
            'BABA_JOTAI_SHIBA_GENZAI': tenko_baba.get(race_code, (None, None, None))[1],
            'BABA_JOTAI_DIRT_GENZAI': tenko_baba.get(race_code, (None, None, None))[2],
        }
        # 馬体重変動
        print(f'馬体重変動取得開始: {row["BATAIJU"]}')
        row_bataiju = float(row['BATAIJU']) if str(row['BATAIJU']).strip() not in [None, '', 'nan'] else 0
        uma_hist_bataiju = pd.to_numeric(uma_hist['BATAIJU'], errors='coerce').fillna(0) if not uma_hist.empty else pd.Series([0])
        feats['BATAIJU_DIFF'] = row_bataiju - uma_hist_bataiju.iloc[-1] if not uma_hist.empty else 0
        # 斤量変動
        print(f'斤量変動取得開始: {row["FUTAN_JURYO"]}')
        row_futan = float(row['FUTAN_JURYO']) if str(row['FUTAN_JURYO']).strip() not in [None, '', 'nan'] else 0
        uma_hist_futan = pd.to_numeric(uma_hist['FUTAN_JURYO'], errors='coerce').fillna(0) if not uma_hist.empty else pd.Series([0])
        feats['FUTAN_JURYO_DIFF'] = row_futan - uma_hist_futan.iloc[-1] if not uma_hist.empty else 0
        # 騎手3着率
        print(f'騎手3着率取得開始: {row["KISHU_CODE"]}')
        feats['KISHU_3IN_RATE'] = kishu_hist['target'].mean() if not kishu_hist.empty else 0
        # 調教師3着率
        print(f'調教師3着率取得開始: {row["CHOKYOSHI_CODE"]}')
        feats['CHOKYOSHI_3IN_RATE'] = chokyoshi_hist['target'].mean() if not chokyoshi_hist.empty else 0
        # 馬の過去5走3着率
        print(f'馬の過去5走3着率取得開始: {uma}')
        zen_5r_chakujun = pd.to_numeric(uma_hist['KAKUTEI_CHAKUJUN_norm'].tail(5), errors='coerce').fillna(0) if len(uma_hist) >= 1 else pd.Series([0])
        feats['UMA_5R_3IN_RATE'] = zen_5r_chakujun.mean() if not zen_5r_chakujun.empty else 0
        feats['ZEN_5R_CHAKUJUN_DIFF'] = zen_5r_chakujun.diff().mean() if len(zen_5r_chakujun) > 1 else 0
        # コース適性
        print(f'コース適性取得開始: {course_id}')
        course_chakujun = pd.to_numeric(uma_course_hist['KAKUTEI_CHAKUJUN_norm'], errors='coerce').fillna(0) if not uma_course_hist.empty else pd.Series([0])
        feats['COURSE_3IN_RATE'] = uma_course_hist['target'].mean() if not uma_course_hist.empty else 0
        feats['COURSE_CHAKUJUN_MEAN'] = course_chakujun.mean() if not course_chakujun.empty else 0
        # 距離適性
        print(f'距離適性取得開始: {kyori}')
        kyori_chakujun = pd.to_numeric(uma_kyori_hist['KAKUTEI_CHAKUJUN_norm'], errors='coerce').fillna(0) if not uma_kyori_hist.empty else pd.Series([0])
        feats['KYORI_3IN_RATE'] = uma_kyori_hist['target'].mean() if not uma_kyori_hist.empty else 0
        feats['KYORI_CHAKUJUN_MEAN'] = kyori_chakujun.mean() if not kyori_chakujun.empty else 0
        # 過去5走着順トレンド
        print(f'過去5走着順トレンド取得開始: {uma}')
        if len(zen_5r_chakujun) >= 2:
            x = np.arange(len(zen_5r_chakujun))
            y = zen_5r_chakujun.values
            A = np.vstack([x, np.ones(len(x))]).T
            m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            feats['ZEN_5R_CHAKUJUN_SLOPE'] = m
        else:
            feats['ZEN_5R_CHAKUJUN_SLOPE'] = 0
        # 過去5走馬体重トレンド
        print(f'過去5走馬体重トレンド取得開始: {uma}')
        zen_5r_bataiju = pd.to_numeric(uma_hist['BATAIJU'].tail(5), errors='coerce').fillna(0) if len(uma_hist) >= 1 else pd.Series([0])
        if len(zen_5r_bataiju) >= 2:
            x = np.arange(len(zen_5r_bataiju))
            y = zen_5r_bataiju.values
            A = np.vstack([x, np.ones(len(x))]).T
            m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            feats['ZEN_5R_BATAIJU_SLOPE'] = m
        else:
            feats['ZEN_5R_BATAIJU_SLOPE'] = 0
        # ラベル
        feats['target'] = row['target']
        feature_rows.append(feats)
    feature_df = pd.DataFrame(feature_rows)
    # カテゴリ変数をLabelEncoder
    for col in ['SEIBETSU_CODE','KISHU_CODE','CHOKYOSHI_CODE','TENKO_JOTAI_GENZAI','BABA_JOTAI_SHIBA_GENZAI','BABA_JOTAI_DIRT_GENZAI']:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))
    feature_df = feature_df.fillna(0)
    n1 = (feature_df['target'] == 1).sum()
    n0 = (feature_df['target'] == 0).sum()
    print(f'target=1(3着以内): {n1}件, target=0: {n0}件')
    if n1 < 100:
        print('警告: 3着以内データが極端に少ないです。抽出期間や条件を見直してください。')
    feature_cols = [
        'BAREI','SEIBETSU_CODE','BATAIJU','BATAIJU_DIFF','FUTAN_JURYO','FUTAN_JURYO_DIFF',
        'KISHU_CODE','CHOKYOSHI_CODE','KYORI','TRACK_CODE','SHUSSO_TOSU','KISHU_3IN_RATE','CHOKYOSHI_3IN_RATE','UMA_5R_3IN_RATE','RACE_KANKAKU','SHUSSO_KAISU','TENKO_JOTAI_GENZAI','BABA_JOTAI_SHIBA_GENZAI','BABA_JOTAI_DIRT_GENZAI',
        'COURSE_3IN_RATE','COURSE_CHAKUJUN_MEAN','KYORI_3IN_RATE','KYORI_CHAKUJUN_MEAN','ZEN_5R_CHAKUJUN_SLOPE','ZEN_5R_BATAIJU_SLOPE'
    ]
    X = feature_df[feature_cols]
    y = feature_df['target']
    end_time = time.time()
    print(f'【fetch_data関数 実行所要時間】{end_time - start_time:.2f}秒')
    return X, y, feature_df, n0, n1

def main():
    # ディレクトリ作成と実行ファイルのコピー
    run_dir = f'sanshutsu_kun/1_predict_models/a1_gbdt_lightbgm/results/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(run_dir, exist_ok=True)
    import sys
    script_path = os.path.abspath(sys.argv[0])
    import shutil, sys
    script_path = os.path.abspath(sys.argv[0])
    shutil.copyfile(script_path, os.path.join(run_dir, os.path.basename(script_path)))
    
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
    # --- SMOTEでオーバーサンプリング ---
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f'SMOTE後: {np.bincount(y_train_res)}')
    # --- GridSearchでscale_pos_weightと閾値を自動最適化 ---
    best_f1 = -1
    best_params = None
    best_threshold_gs = 0.5
    for params in ParameterGrid({'scale_pos_weight':[n0/n1 if n1>0 else 1.0, n0/n1*1.2 if n1>0 else 1.0, n0/n1*1.5 if n1>0 else 1.0], 'learning_rate':[0.03, 0.01]}):
        lgb_params_gs = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 7,
            'learning_rate': params['learning_rate'],
            'feature_fraction': 0.9,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'scale_pos_weight': params['scale_pos_weight']
        }
        lgb_model_gs = lgb.train(
            lgb_params_gs, lgb.Dataset(X_train_res, y_train_res), valid_sets=[lgb.Dataset(X_test, y_test)], num_boost_round=100,
            callbacks=[lgb.early_stopping(10)]
        )
        probs_gs = lgb_model_gs.predict(X_test, num_iteration=lgb_model_gs.best_iteration)
        precisions_gs, recalls_gs, thresholds_gs = precision_recall_curve(y_test, probs_gs)
        f1s_gs = 2 * precisions_gs * recalls_gs / (precisions_gs + recalls_gs + 1e-8)
        idx_gs = f1s_gs.argmax()
        if f1s_gs[idx_gs] > best_f1:
            best_f1 = f1s_gs[idx_gs]
            best_params = params
            best_threshold_gs = thresholds_gs[idx_gs] if idx_gs < len(thresholds_gs) else 0.5
    print(f'GridSearch最良パラメータ: {best_params}, 最良閾値: {best_threshold_gs:.4f}, F1: {best_f1:.4f}')
    # 最良パラメータで再学習
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 7,
        'learning_rate': best_params['learning_rate'],
        'feature_fraction': 0.9,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'scale_pos_weight': best_params['scale_pos_weight']
    }
    # --- LightGBM学習（学習曲線記録用） ---
    evals_result = {}
    lgb_model = lgb.train(
        lgb_params, lgb.Dataset(X_train_res, y_train_res), valid_sets=[lgb.Dataset(X_train_res, y_train_res), lgb.Dataset(X_test, y_test)],
        valid_names=['train','valid'], num_boost_round=200,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)],
        evals_result=evals_result
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
    # --- XGBoost学習（同じ特徴量） ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=best_params['learning_rate'], scale_pos_weight=best_params['scale_pos_weight'],
        reg_alpha=1.0, reg_lambda=1.0, use_label_encoder=False, eval_metric='logloss', verbosity=0
    )
    xgb_model.fit(X_train_res, y_train_res)
    xgb_probs = xgb_model.predict_proba(X_test)[:,1]
    # --- CatBoost学習（同じ特徴量） ---
    cat_model = CatBoostClassifier(iterations=200, depth=7, learning_rate=best_params['learning_rate'], scale_pos_weight=best_params['scale_pos_weight'], l2_leaf_reg=1.0, verbose=0, random_seed=42)
    cat_model.fit(X_train_res, y_train_res)
    cat_probs = cat_model.predict_proba(X_test)[:,1]
    # --- アンサンブル（平均） ---
    probs = (lgb_probs + xgb_probs + cat_probs) / 3
    # --- PRカーブからprecision重視（0.8以上）かつF1最大の閾値を探索 ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    mask = precisions >= 0.8
    if mask.sum() > 0:
        best_idx = (f1s * mask).argmax()
    else:
        best_idx = f1s.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f'最適閾値（precision>=0.8,F1最大）: {best_threshold:.4f}, F1={f1s[best_idx]:.4f}, precision={precisions[best_idx]:.4f}, recall={recalls[best_idx]:.4f}')
    # --- 複数閾値で自動評価 ---
    thresholds_to_eval = [0.5, best_threshold, 0.7, 0.8]
    for th in thresholds_to_eval:
        preds = (probs >= th).astype(int)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        report = classification_report(y_test, preds)
        print(f'--- 評価結果（閾値={th:.2f}） ---')
        print(f'Accuracy: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(report)
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
競馬の3着以内に入る馬の特徴をLightGBM+XGBoost+CatBoostアンサンブルで学習したモデルの評価結果です。\n\n最適閾値: {best_threshold:.4f}\n精度: {acc:.4f}\nF1スコア: {f1:.4f}\n詳細:\n{report}\n\n以下のグラフ画像（base64エンコード済み）も参考にして、モデル改善のためのアドバイスを日本語で簡潔に出力してください。\n"""
    gpt_advice = ask_gpt41(prompt, images_b64)
    # 結果保存
    with open(os.path.join(run_dir, 'eval_and_gpt_advice.txt'), 'w', encoding='utf-8') as f:
        f.write('--- 評価結果 ---\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(report + '\n')
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
        n_estimators=200, max_depth=7, learning_rate=best_params['learning_rate'], scale_pos_weight=best_params['scale_pos_weight'],
        reg_alpha=1.0, reg_lambda=1.0, use_label_encoder=False, eval_metric='logloss', verbosity=0
    )
    xgb_model_sel.fit(X_train_sel, y_train_res)
    xgb_probs_sel = xgb_model_sel.predict_proba(X_test_sel)[:,1]
    cat_model_sel = CatBoostClassifier(iterations=200, depth=7, learning_rate=best_params['learning_rate'], scale_pos_weight=best_params['scale_pos_weight'], l2_leaf_reg=1.0, verbose=0, random_seed=42)
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