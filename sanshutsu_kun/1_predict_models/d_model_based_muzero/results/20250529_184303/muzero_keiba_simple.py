import os
import pymysql
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score
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

target_days = 180  # 直近半年分

# --- 特徴量定義（DQN/LightGBMと揃える） ---
FEATURES = [
    'BAREI', 'SEIBETSU_CODE', 'BATAIJU', 'BATAIJU_DIFF', 'FUTAN_JURYO', 'WAKUBAN',
    'KYORI', 'TRACK_CODE', 'SHUSSO_TOSU', 'KISHU_3IN_RATE', 'TENKO_CODE',
    'CHOKYOSHI_3IN_RATE', 'FATHER_LINEAGE', 'BATAIJU_KYORI', 'BAREI_FUTAN_JURYO',
    'RACE_KANKAKU', 'ZENSHO_ICHIAKUMA_SA', 'ZENSHO_GYAKUJUN'
]

# --- データ抽出・前処理 ---
def fetch_data_time_split(test_ratio=0.2):
    print('=== fetch_data_time_split関数 実行開始 ===')
    start_time = time.time()
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, db=DB_NAME, port=DB_PORT, charset='utf8mb4')
    cursor = conn.cursor()
    today = datetime.today()
    date_min = (today - timedelta(days=target_days)).strftime('%Y%m%d')
    db_columns = ['BAREI','SEIBETSU_CODE','BATAIJU','FUTAN_JURYO','WAKUBAN','KISHU_CODE','CHOKYOSHI_CODE','KETTO_TOROKU_BANGO','RACE_CODE','KAISAI_NEN','KAISAI_GAPPI','KAKUTEI_CHAKUJUN']
    cursor.execute(f'''SELECT {', '.join(db_columns)} FROM umagoto_race_joho WHERE CONCAT(KAISAI_NEN, LPAD(KAISAI_GAPPI, 4, '0')) >= %s''', (date_min,))
    rows = cursor.fetchall()
    df_raw = pd.DataFrame(rows, columns=db_columns)
    df_raw['RACE_DATE'] = df_raw['KAISAI_NEN'].astype(str) + df_raw['KAISAI_GAPPI'].astype(str).str.zfill(4)
    df_raw['RACE_DATE'] = pd.to_datetime(df_raw['RACE_DATE'], format='%Y%m%d', errors='coerce')
    df_raw = df_raw.sort_values('RACE_DATE').reset_index(drop=True)
    cursor.execute('SELECT RACE_CODE, KYORI, TRACK_CODE, SHUSSO_TOSU FROM race_shosai')
    race_shosai_df = pd.DataFrame(cursor.fetchall(), columns=['RACE_CODE','KYORI','TRACK_CODE','SHUSSO_TOSU'])
    for col in ['KYORI','TRACK_CODE','SHUSSO_TOSU']:
        race_shosai_df[col] = pd.to_numeric(race_shosai_df[col], errors='coerce').fillna(0)
    df_raw = pd.merge(df_raw, race_shosai_df, on='RACE_CODE', how='left')
    df_raw['KYORI'] = df_raw['KYORI'].fillna(0)
    df_raw['TRACK_CODE'] = df_raw['TRACK_CODE'].fillna(0)
    df_raw['SHUSSO_TOSU'] = df_raw['SHUSSO_TOSU'].fillna(0)
    df_raw['KAKUTEI_CHAKUJUN_norm'] = df_raw['KAKUTEI_CHAKUJUN'].astype(str).str.strip().replace('　','').replace(' ','').str.translate(str.maketrans('０１２３４５６７８９', '0123456789')).str.lstrip('0')
    df_raw['target'] = df_raw['KAKUTEI_CHAKUJUN_norm'].isin(['1','2','3']).astype(int)
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

# --- MuZero主要ネットワーク ---
class RepresentationNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.fc(x))

class DynamicsNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(hidden_dim + 1, hidden_dim)  # 状態+行動
        self.relu = nn.ReLU()
    def forward(self, state, action):
        x = torch.cat([state, action.unsqueeze(-1)], dim=-1)
        return self.relu(self.fc(x))

class PredictionNet(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=2):
        super().__init__()
        self.policy = nn.Linear(hidden_dim, output_dim)
        self.value = nn.Linear(hidden_dim, 1)
    def forward(self, state):
        policy_logits = self.policy(state)
        value = self.value(state)
        return policy_logits, value

# --- 簡易ReplayBuffer ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
    def push(self, item):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(item)
    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in idxs]
    def __len__(self):
        return len(self.buffer)

# --- MuZero学習ループ（簡易版） ---
def train_muzero(params, run_dir):
    X_train, y_train, X_test, y_test, train_df, test_df = fetch_data_time_split(test_ratio=params['test_ratio'])
    dataset = KeibaDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    input_dim = X_train.shape[1]
    hidden_dim = params['hidden_dim']
    output_dim = 2
    representation = RepresentationNet(input_dim, hidden_dim)
    dynamics = DynamicsNet(hidden_dim)
    prediction = PredictionNet(hidden_dim, output_dim)
    optimizer = optim.Adam(list(representation.parameters()) + list(dynamics.parameters()) + list(prediction.parameters()), lr=params['lr'])
    buffer = ReplayBuffer(capacity=2000)
    train_losses, train_accuracies = [], []
    for epoch in range(params['epochs']):
        print_progress_bar(epoch+1, params['epochs'], bar_length=50, prefix='進捗', suffix='')
        total_loss, correct, total = 0, 0, 0
        for Xb, yb in loader:
            # --- 1step分のMuZero流れ（簡易） ---
            state = representation(Xb)
            policy_logits, value = prediction(state)
            policy = torch.softmax(policy_logits, dim=-1)
            # NaN/inf/負値対策
            if torch.isnan(policy).any() or torch.isinf(policy).any() or (policy < 0).any():
                policy = torch.full_like(policy, 1.0 / policy.shape[1])
            policy = torch.clamp(policy, 1e-8, 1-1e-8)
            policy = policy / policy.sum(dim=-1, keepdim=True)  # 合計1に
            if torch.isnan(policy).any() or torch.isinf(policy).any() or (policy < 0).any():
                policy = torch.full_like(policy, 1.0 / policy.shape[1])
            action = torch.multinomial(policy, 1).squeeze(-1)
            reward = (action == yb).float()
            next_state = dynamics(state, action.float())
            next_policy_logits, next_value = prediction(next_state)
            buffer.push((state.detach(), action.detach(), reward.detach(), next_state.detach(), yb.detach()))
            # --- バッファからサンプルして学習 ---
            if len(buffer) >= params['batch_size']:
                batch = buffer.sample(params['batch_size'])
                s, a, r, ns, y_true = zip(*batch)
                s = torch.stack(s)
                a = torch.stack(a)
                r = torch.stack(r)
                ns = torch.stack(ns)
                y_true = torch.stack(y_true)
                # 予測
                policy_logits, value = prediction(s)
                policy_loss = nn.CrossEntropyLoss()(policy_logits, y_true)
                value_loss = nn.MSELoss()(value.squeeze(-1), r)
                loss = policy_loss + value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * s.size(0)
                correct += (policy_logits.argmax(dim=1) == y_true).sum().item()
                total += s.size(0)
        if total > 0:
            train_losses.append(total_loss / total)
            train_accuracies.append(correct / total)
        else:
            train_losses.append(0)
            train_accuracies.append(0)
    # --- 評価 ---
    representation.eval()
    prediction.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        state = representation(X_tensor)
        policy_logits, value = prediction(state)
        probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    report = classification_report(y_test, preds, zero_division=0)
    print(f'MuZero Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}')
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
競馬の3着以内に入る馬の特徴をMuZeroで学習したモデルの評価結果です。\n\n精度: {acc:.4f}\nF1スコア: {f1:.4f}\nRecall: {recall:.4f}\n詳細:\n{report}\n\n以下のグラフ画像（base64エンコード済み）も参考にして、モデル改善のためのアドバイスを日本語で簡潔に出力してください。\n"""
    gpt_advice = ask_gpt41(prompt, images_b64)
    with open(os.path.join(run_dir, 'eval_and_gpt_advice.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Accuracy: {acc:.4f}\nF1 Score: {f1:.4f}\nRecall: {recall:.4f}\n')
        f.write(report + '\n')
        f.write('--- GPT-4.1からの改善アドバイス ---\n')
        f.write(gpt_advice + '\n')
    return acc, f1, recall, gpt_advice

if __name__ == '__main__':
    start_time = time.time()
    run_dir = save_script_and_make_run_dir('sanshutsu_kun/1_predict_models/d_model_based_muzero/results')
    params = {
        'epochs': 10,
        'batch_size': 64,
        'lr': 0.001,
        'hidden_dim': 64,
        'test_ratio': 0.2,
    }
    train_muzero(params, run_dir)
    end_time = time.time()
    print(f'【実行所要時間】{end_time - start_time:.2f}秒') 