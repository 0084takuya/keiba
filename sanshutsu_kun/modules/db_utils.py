import os
import pymysql

def get_db_connection():
    """
    .envの環境変数からDB接続情報を取得し、pymysqlのコネクションを返す
    """
    return pymysql.connect(
        host=os.getenv('KEIBA_DB_HOST', 'localhost'),
        user=os.getenv('KEIBA_DB_USER', 'root'),
        password=os.getenv('KEIBA_DB_PASS', ''),
        db=os.getenv('KEIBA_DB_NAME', 'mykeibadb'),
        port=int(os.getenv('KEIBA_DB_PORT', 3306)),
        charset='utf8mb4'
    ) 