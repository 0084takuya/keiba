import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import mysql.connector

class HorseRacingEnv(gym.Env):
    """
    深層強化学習用の競馬環境
    各馬エントリーをエージェントとして、Top3入りを予測する環境を定義します。
    """
    metadata = {'render.modes': []}

    def __init__(self, host, user, password, database, entry_table, result_table, port=3306, max_samples=None, lookback_number=1, lookback_unit='MONTH'):
        super(HorseRacingEnv, self).__init__()
        print("[Env __init__] DB接続を開始します...")
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )
        # pandas.read_sql用のSQLAlchemyエンジンを作成
        self.engine = create_engine(
            f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        )
        # 取得期間設定
        self.lookback_number = lookback_number
        self.lookback_unit = lookback_unit
        print("[Env __init__] DB接続が完了しました。")
        print("[Env __init__] データセットのロードを開始します。")
        self.entry_table = entry_table
        self.result_table = result_table
        self.max_samples = max_samples

        # データのロードと前処理
        self.data = self._load_dataset()
        print(f"[Env __init__] データセットのロード完了。サンプル数={len(self.data)}")
        # データセットが空の場合はエラーを出力して処理を中断します
        if len(self.data) == 0:
            raise ValueError(
                f"[Env __init__] データセットが空です。lookback_number={self.lookback_number}, lookback_unit={self.lookback_unit} を確認してください。"
            )
        if self.max_samples:
            self.data = self.data.iloc[:self.max_samples].reset_index(drop=True)
        self.num_samples = len(self.data)
        self.current_idx = 0

        # 特徴量とラベルの設定
        self.features = [c for c in self.data.columns if c not in ['horse_id', 'label']]
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32
        )
        self.action_space = Discrete(2)  # 0: 3位以内ではない, 1: 3位以内

    def _load_dataset(self):
        print("[Env _load_dataset] SQLクエリを実行中...")
        # メインデータをSQLで取得（直近1ヶ月分）
        query = f"""
        SELECT
            e.KETTO_TOROKU_BANGO AS horse_id,
            e.KAISAI_NEN AS kaisai_year,
            e.KAISAI_GAPPI AS kaisai_date,
            (CAST(e.KAISAI_NEN AS SIGNED) - CAST(SUBSTRING(s.SEINENGAPPI,1,4) AS SIGNED)) AS age,
            e.SEIBETSU_CODE AS gender,
            e.CHOKYOSHI_CODE AS trainer_code,
            e.UMAKIGO_CODE AS breed_code,
            e.HINSHU_CODE AS breed_type,
            e.MOSHOKU_CODE AS moshoku_code,
            CAST(e.BAREI AS SIGNED) AS horse_weight,
            CAST(e.BATAIJU AS SIGNED) AS body_weight,
            CAST(e.FUTAN_JURYO AS SIGNED) AS jockey_weight,
            CAST(e.NYUSEN_JUNI AS SIGNED) AS start_order,
            CAST(e.TANSHO_ODDS AS SIGNED) AS odds,
            CAST(e.TANSHO_NINKIJUN AS SIGNED) AS popularity,
            CAST(r.KYORI AS SIGNED) AS distance,
            r.TRACK_CODE AS track_code,
            CASE WHEN r.HOJIUMA1_KETTO_TOROKU_BANGO = e.KETTO_TOROKU_BANGO
                  OR r.HOJIUMA2_KETTO_TOROKU_BANGO = e.KETTO_TOROKU_BANGO
                  OR r.HOJIUMA3_KETTO_TOROKU_BANGO = e.KETTO_TOROKU_BANGO
                 THEN 1 ELSE 0 END AS label
        FROM {self.entry_table} e
        LEFT JOIN {self.result_table} r
          ON e.RACE_CODE = r.RACE_CODE
        LEFT JOIN sanku_master2 s
          ON e.KETTO_TOROKU_BANGO = s.KETTO_TOROKU_BANGO
        WHERE STR_TO_DATE(CONCAT(e.KAISAI_NEN, e.KAISAI_GAPPI), '%Y%m%d') >= DATE_SUB(CURDATE(), INTERVAL {self.lookback_number} {self.lookback_unit})
        """
        # SQLAlchemyエンジンを使ってDataFrameを取得
        df = pd.read_sql(query, self.engine)
        # 日付列に変換
        df['race_date'] = pd.to_datetime(df['kaisai_year'] + df['kaisai_date'], format='%Y%m%d')
        # ソートして時系列順に整列
        df = df.sort_values(['horse_id', 'race_date']).reset_index(drop=True)
        # 過去出走回数
        df['past_race_count'] = df.groupby('horse_id').cumcount()
        # 過去トップ3回数
        df['past_top3_count'] = df.groupby('horse_id')['label'].transform(lambda x: x.shift().fillna(0).cumsum())
        # 不要列を削除
        df = df.drop(columns=['kaisai_year', 'kaisai_date', 'race_date'])

        print(f"[Env _load_dataset] SQL実行完了。取得件数={len(df)}")

        # 数値変換
        print("[Env _load_dataset] 数値変換を実行中...")
        numeric_cols = ['age', 'distance', 'horse_weight', 'body_weight', 'jockey_weight', 'start_order', 'odds', 'popularity', 'past_race_count', 'past_top3_count']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        print("[Env _load_dataset] 数値変換完了")
        # カテゴリ変数のダミー化
        print("[Env _load_dataset] ダミー変数化を実行中...")
        df = pd.get_dummies(df, columns=['gender', 'trainer_code', 'track_code', 'breed_code', 'breed_type', 'moshoku_code'], drop_first=True)
        print(f"[Env _load_dataset] ダミー変数化完了。カラム数={df.shape[1]}")
        # 欠損値の除去
        print("[Env _load_dataset] 欠損値の除去を実行中...")
        df = df.dropna().reset_index(drop=True)
        print(f"[Env _load_dataset] 欠損値除去完了。取得件数={len(df)}")
        return df

    def reset(self, *args, **kwargs):
        """Gym/GymnasiumおよびStable-Baselines3互換: seed等を受け取ってもエラーなし"""
        seed = kwargs.get('seed', None)
        print(f"[Env reset] インデックス={self.current_idx}でリセット (seed={seed})")
        if self.current_idx >= self.num_samples:
            self.current_idx = 0
        obs = self.data.loc[self.current_idx, self.features].values.astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        print(f"[Env step] ステップ開始 index={self.current_idx}, action={action}")
        label = int(self.data.loc[self.current_idx, 'label'])
        reward = 1.0 if action == label else -1.0
        # Gymnasium互換: terminated（終了）とtruncated（打ち切り）を返します
        terminated = True
        truncated = False
        info = {}
        self.current_idx += 1
        obs = None
        if self.current_idx < self.num_samples:
            obs = self.data.loc[self.current_idx, self.features].values.astype(np.float32)
        print(f"[Env step] ステップ終了 next_index={self.current_idx}, reward={reward}")
        return obs, reward, terminated, truncated, info

    def close(self):
        print("[Env close] 環境をクローズします")
        self.conn.close() 