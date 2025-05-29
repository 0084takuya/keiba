import os
import pymysql
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.metrics import recall_score
from dotenv import load_dotenv
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import base64
import pandas as pd
import time
from modules.plot_utils import plot_and_save_graphs, save_script_and_make_run_dir
from modules.gpt_utils import ask_gpt41
from modules.progress_utils import print_progress_bar
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

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

target_days = 360 * 3  # 直近3年分

# --- 特徴量定義（DQN/LightGBMと揃える） ---
FEATURES = [
    'BAREI', 'SEIBETSU_CODE', 'BATAIJU', 'BATAIJU_DIFF', 'FUTAN_JURYO', 'WAKUBAN',
    'KYORI', 'TRACK_CODE', 'SHUSSO_TOSU', 'KISHU_3IN_RATE', 'TENKO_CODE',
    'CHOKYOSHI_3IN_RATE', 'FATHER_LINEAGE', 'BATAIJU_KYORI', 'BAREI_FUTAN_JURYO',
    'RACE_KANKAKU', 'ZENSHO_ICHIAKUMA_SA', 'ZENSHO_GYAKUJUN'
]

# --- データ抽出・前処理（DQN/LightGBMのfetch_data_time_splitを流用・拡張） ---
def fetch_data_time_split(test_ratio=0.2):
    print('=== fetch_data_time_split関数 実行開始 ===')
    start_time = time.time()
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, db=DB_NAME, port=DB_PORT, charset='utf8mb4')
    cursor = conn.cursor()
    today = datetime.today()
    date_min = (today - timedelta(days=target_days)).strftime('%Y%m%d')
    # --- メインデータ ---
    db_columns = ['BAREI','SEIBETSU_CODE','BATAIJU','FUTAN_JURYO','WAKUBAN','KISHU_CODE','CHOKYOSHI_CODE','KETTO_TOROKU_BANGO','RACE_CODE','KAISAI_NEN','KAISAI_GAPPI','KAKUTEI_CHAKUJUN']
    cursor.execute(f'''SELECT {', '.join(db_columns)} FROM umagoto_race_joho WHERE CONCAT(KAISAI_NEN, LPAD(KAISAI_GAPPI, 4, '0')) >= %s''', (date_min,))
    rows = cursor.fetchall()
    df_raw = pd.DataFrame(rows, columns=db_columns)
    df_raw['RACE_DATE'] = df_raw['KAISAI_NEN'].astype(str) + df_raw['KAISAI_GAPPI'].astype(str).str.zfill(4)
    df_raw['RACE_DATE'] = pd.to_datetime(df_raw['RACE_DATE'], format='%Y%m%d', errors='coerce')
    df_raw = df_raw.sort_values('RACE_DATE').reset_index(drop=True)
    # --- race_shosaiをmerge ---
    cursor.execute('SELECT RACE_CODE, KYORI, TRACK_CODE, SHUSSO_TOSU FROM race_shosai')
    race_shosai_df = pd.DataFrame(cursor.fetchall(), columns=['RACE_CODE','KYORI','TRACK_CODE','SHUSSO_TOSU'])
    for col in ['KYORI','TRACK_CODE','SHUSSO_TOSU']:
        race_shosai_df[col] = pd.to_numeric(race_shosai_df[col], errors='coerce').fillna(0)
    df_raw = pd.merge(df_raw, race_shosai_df, on='RACE_CODE', how='left')
    df_raw['KYORI'] = df_raw['KYORI'].fillna(0)
    df_raw['TRACK_CODE'] = df_raw['TRACK_CODE'].fillna(0)
    df_raw['SHUSSO_TOSU'] = df_raw['SHUSSO_TOSU'].fillna(0)
    # --- ラベル ---
    df_raw['KAKUTEI_CHAKUJUN_norm'] = df_raw['KAKUTEI_CHAKUJUN'].astype(str).str.strip().replace('　','').replace(' ','').str.translate(str.maketrans('０１２３４５６７８９', '0123456789')).str.lstrip('0')
    df_raw['target'] = df_raw['KAKUTEI_CHAKUJUN_norm'].isin(['1','2','3']).astype(int)
    # --- 特徴量生成 ---
    def make_features(df, ref_df):
        X, y = [], []
        for idx, (_, row) in enumerate(df.iterrows()):
            print_progress_bar(idx+1, len(df), bar_length=100, prefix='進捗', suffix='')
            race_date = row['RACE_DATE']
            ref = ref_df[ref_df['RACE_DATE'] < race_date]
            feature_dict = {}
            for col in FEATURES:
                if col in row:
                    feature_dict[col] = pd.to_numeric(row[col], errors='coerce') if not pd.isnull(row[col]) else 0
            # 騎手3着率
            kishu_code = row['KISHU_CODE']
            if len(ref) > 0 and 'KAKUTEI_CHAKUJUN' in ref.columns:
                kishu_3in = ref[ref['KISHU_CODE'] == kishu_code]
                feature_dict['KISHU_3IN_RATE'] = kishu_3in['KAKUTEI_CHAKUJUN'].isin(['1','2','3']).mean() if 'KAKUTEI_CHAKUJUN' in kishu_3in else 0
            else:
                feature_dict['KISHU_3IN_RATE'] = 0
            # 調教師3着率
            chokyoshi_code = row['CHOKYOSHI_CODE']
            if len(ref) > 0 and 'KAKUTEI_CHAKUJUN' in ref.columns:
                chokyoshi_3in = ref[ref['CHOKYOSHI_CODE'] == chokyoshi_code]
                feature_dict['CHOKYOSHI_3IN_RATE'] = chokyoshi_3in['KAKUTEI_CHAKUJUN'].isin(['1','2','3']).mean() if 'KAKUTEI_CHAKUJUN' in chokyoshi_3in else 0
            else:
                feature_dict['CHOKYOSHI_3IN_RATE'] = 0
            # 馬体重×距離
            def safe_float(val):
                try:
                    if val is None or (isinstance(val, str) and val.strip() == ''):
                        return 0.0
                    return float(val)
                except Exception:
                    return 0.0
            bataiju = safe_float(row['BATAIJU'])
            kyori = safe_float(row['KYORI'])
            feature_dict['BATAIJU_KYORI'] = bataiju * kyori
            # 馬齢×負担重量
            barei = safe_float(row['BAREI'])
            futan = safe_float(row['FUTAN_JURYO'])
            feature_dict['BAREI_FUTAN_JURYO'] = barei * futan
            # 直近体重変化
            if idx > 0 and row['KETTO_TOROKU_BANGO'] == df.iloc[idx-1]['KETTO_TOROKU_BANGO']:
                feature_dict['BATAIJU_DIFF'] = safe_float(row['BATAIJU']) - safe_float(df.iloc[idx-1]['BATAIJU'])
            else:
                feature_dict['BATAIJU_DIFF'] = 0
            # レース間隔（日数）
            if idx > 0 and row['KETTO_TOROKU_BANGO'] == df.iloc[idx-1]['KETTO_TOROKU_BANGO']:
                feature_dict['RACE_KANKAKU'] = (row['RACE_DATE'] - df.iloc[idx-1]['RACE_DATE']).days
            else:
                feature_dict['RACE_KANKAKU'] = 0
            # ダミー: その他特徴量は0埋め
            for col in FEATURES:
                if col not in feature_dict:
                    feature_dict[col] = 0
            y.append(row['target'])
            X.append([feature_dict[f] for f in FEATURES])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
    split_idx = int(len(df_raw) * (1 - test_ratio))
    train_df = df_raw.iloc[:split_idx].copy()
    test_df = df_raw.iloc[split_idx:].copy()
    X_train, y_train = make_features(train_df, train_df)
    X_test, y_test = make_features(test_df, train_df)
    fetch_time = time.time() - start_time
    print(f'【データ取得・前処理所要時間】{fetch_time:.2f}秒')
    return X_train, y_train, X_test, y_test, train_df, test_df

# --- PyTorch Dataset ---
class KeibaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- A2Cネットワーク ---
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)  # NaNを0に
        out = self.net(x)
        probs = torch.softmax(out, dim=-1)
        probs = torch.clamp(probs, 1e-8, 1-1e-8)  # 0,1,NaN防止
        probs = probs / probs.sum(dim=-1, keepdim=True)  # 合計1に
        # NaNがあれば[0.5,0.5]で置換
        if torch.isnan(probs).any():
            probs = torch.where(torch.isnan(probs), torch.full_like(probs, 0.5), probs)
        return probs

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# --- A2C学習ループ ---
def train_a2c(params, run_dir):
    X_train, y_train, X_test, y_test, train_df, test_df = fetch_data_time_split(test_ratio=params['test_ratio'])
    dataset = KeibaDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    actor = Actor(input_dim=X_train.shape[1], output_dim=2, hidden_dim=params['hidden_dim'])
    critic = Critic(input_dim=X_train.shape[1], hidden_dim=params['hidden_dim'])
    actor_optim = optim.Adam(actor.parameters(), lr=params['lr'])
    critic_optim = optim.Adam(critic.parameters(), lr=params['lr'])
    gamma = params['gamma']
    # クラス重み計算（1:3着以内, 0:それ以外）
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    train_losses, train_accuracies = [], []
    for epoch in range(params['epochs']):
        print_progress_bar(epoch+1, params['epochs'], bar_length=50, prefix='進捗', suffix='')
        total_loss, correct, total = 0, 0, 0
        for Xb, yb in loader:
            # --- Actor ---
            probs = actor(Xb)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            # --- Critic ---
            values = critic(Xb)
            rewards = (actions == yb).float()  # 正解なら1, 不正解なら0
            next_values = values.detach()  # 単純化
            advantages = rewards + gamma * next_values - values
            # クラス重み適用
            sample_weights = class_weights[yb]
            actor_loss = -((log_probs * advantages.detach()) * sample_weights).mean()
            critic_loss = (advantages.pow(2) * sample_weights).mean()
            loss = actor_loss + critic_loss
            actor_optim.zero_grad()
            critic_optim.zero_grad()
            loss.backward()
            actor_optim.step()
            critic_optim.step()
            total_loss += loss.item() * Xb.size(0)
            correct += (actions == yb).sum().item()
            total += Xb.size(0)
        train_losses.append(total_loss / total)
        train_accuracies.append(correct / total)
    # --- 評価 ---
    actor.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        probs = actor(X_tensor).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    report = classification_report(y_test, preds, zero_division=0)
    print(f'A2C Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}')
    # --- グラフ保存 ---
    graph_paths = plot_and_save_graphs(y_test, preds, probs[:,1], train_losses, train_accuracies, run_dir)
    images_b64 = []
    for path in graph_paths:
        if path and os.path.exists(path):
            with open(path, 'rb') as imgf:
                b64img = base64.b64encode(imgf.read()).decode('utf-8')
                images_b64.append((os.path.basename(path), b64img))
    # --- GPT-4.1にアドバイスを問い合わせ ---
    prompt = f"""
競馬の3着以内に入る馬の特徴をA2Cで学習したモデルの評価結果です。\n\n精度: {acc:.4f}\nF1スコア: {f1:.4f}\nRecall: {recall:.4f}\n詳細:\n{report}\n\n以下のグラフ画像（base64エンコード済み）も参考にして、モデル改善のためのアドバイスを日本語で簡潔に出力してください。\n"""
    gpt_advice = ask_gpt41(prompt, images_b64)
    with open(os.path.join(run_dir, 'eval_and_gpt_advice.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Accuracy: {acc:.4f}\nF1 Score: {f1:.4f}\nRecall: {recall:.4f}\n')
        f.write(report + '\n')
        f.write('--- GPT-4.1からの改善アドバイス ---\n')
        f.write(gpt_advice + '\n')
    return acc, f1, recall, gpt_advice

# --- 自動改善ループ ---
def auto_improve():
    run_dir = save_script_and_make_run_dir('sanshutsu_kun/1_predict_models/b_policy_based_a2c/results')
    params = {
        'epochs': 10,
        'batch_size': 64,
        'lr': 0.001,
        'gamma': 0.99,
        'hidden_dim': 64,
        'test_ratio': 0.2,
    }
    best_f1 = 0
    for trial in range(5):
        print(f'==== 改善トライアル {trial+1} ===')
        acc, f1, recall, advice = train_a2c(params, run_dir)
        # --- アドバイスに基づきパラメータ自動調整（例: F1が低ければエポック数増やす等） ---
        if 'エポック' in advice or f1 < 0.7:
            params['epochs'] = min(params['epochs'] + 5, 50)
        if '学習率' in advice or f1 < 0.7:
            params['lr'] = max(params['lr'] * 0.7, 1e-5)
        if '層' in advice or f1 < 0.7:
            params['hidden_dim'] = min(params['hidden_dim'] + 32, 256)
        if f1 > best_f1:
            best_f1 = f1
        if f1 > 0.95 and recall > 0.95:
            print('十分な性能に到達したため自動改善を終了します')
            break

if __name__ == '__main__':
    start_time = time.time()
    auto_improve()
    end_time = time.time()
    print(f'【実行所要時間】{end_time - start_time:.2f}秒') 