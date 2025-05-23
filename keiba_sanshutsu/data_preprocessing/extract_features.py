import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pandas as pd
import pymysql
from keiba_sanshutsu.config.db_config import DB_CONFIG
from datetime import datetime, timedelta

def get_connection():
    return pymysql.connect(**DB_CONFIG)

def extract_features():
    conn = get_connection()
    today = datetime.today()
    two_years_ago = today - timedelta(days=730)
    two_years_ago_str = two_years_ago.strftime('%Y%m%d')
    # race_shosaiから直近2年分のRACE_CODEを抽出
    df_race_shosai = pd.read_sql("SELECT RACE_CODE, KAISAI_NEN, KAISAI_GAPPI FROM race_shosai", conn)
    # 年月日を連結してint化
    df_race_shosai['DATE'] = df_race_shosai['KAISAI_NEN'].astype(str) + df_race_shosai['KAISAI_GAPPI'].astype(str)
    df_race_shosai['DATE'] = df_race_shosai['DATE'].astype(int)
    target_race_codes = df_race_shosai[df_race_shosai['DATE'] >= int(two_years_ago_str)]['RACE_CODE'].tolist()
    # レース成績（直近2年分）
    if target_race_codes:
        format_strings = ','.join(['%s'] * len(target_race_codes))
        df_race = pd.read_sql(f"SELECT * FROM umagoto_race_joho WHERE RACE_CODE IN ({format_strings})", conn, params=target_race_codes)
    else:
        df_race = pd.DataFrame()
    # 馬ID・騎手ID・調教師ID・血統IDを抽出
    horse_ids = df_race['KETTO_TOROKU_BANGO'].unique().tolist() if 'KETTO_TOROKU_BANGO' in df_race.columns else []
    jockey_ids = df_race['KISHU_CODE'].unique().tolist() if 'KISHU_CODE' in df_race.columns else []
    trainer_ids = df_race['CHOKYOSHI_CODE'].unique().tolist() if 'CHOKYOSHI_CODE' in df_race.columns else []
    # 馬の基本情報
    if horse_ids:
        format_strings = ','.join(['%s'] * len(horse_ids))
        df_horse = pd.read_sql(f"SELECT * FROM kyosoba_master2 WHERE KETTO_TOROKU_BANGO IN ({format_strings})", conn, params=horse_ids)
    else:
        df_horse = pd.DataFrame()
    # 騎手情報
    if jockey_ids:
        format_strings = ','.join(['%s'] * len(jockey_ids))
        df_jockey = pd.read_sql(f"SELECT * FROM kishu_master WHERE KISHU_CODE IN ({format_strings})", conn, params=jockey_ids)
    else:
        df_jockey = pd.DataFrame()
    # 調教師情報
    if trainer_ids:
        format_strings = ','.join(['%s'] * len(trainer_ids))
        df_trainer = pd.read_sql(f"SELECT * FROM chokyoshi_master WHERE CHOKYOSHI_CODE IN ({format_strings})", conn, params=trainer_ids)
    else:
        df_trainer = pd.DataFrame()
    # 血統情報（馬の血統IDを使う）
    pedigree_ids = df_horse['KEITO_ID'].unique().tolist() if 'KEITO_ID' in df_horse.columns else []
    if pedigree_ids:
        format_strings = ','.join(['%s'] * len(pedigree_ids))
        df_pedigree = pd.read_sql(f"SELECT * FROM keito_joho2 WHERE KEITO_ID IN ({format_strings})", conn, params=pedigree_ids)
    else:
        df_pedigree = pd.DataFrame()
    conn.close()
    print("データ抽出完了（直近2年分）")
    # CSVで保存
    df_horse.to_csv('keiba_sanshutsu/data_preprocessing/horse.csv', index=False)
    df_jockey.to_csv('keiba_sanshutsu/data_preprocessing/jockey.csv', index=False)
    df_trainer.to_csv('keiba_sanshutsu/data_preprocessing/trainer.csv', index=False)
    df_pedigree.to_csv('keiba_sanshutsu/data_preprocessing/pedigree.csv', index=False)
    df_race.to_csv('keiba_sanshutsu/data_preprocessing/race.csv', index=False)
    print("CSV保存完了")
    return df_horse, df_jockey, df_trainer, df_pedigree, df_race

if __name__ == "__main__":
    extract_features() 