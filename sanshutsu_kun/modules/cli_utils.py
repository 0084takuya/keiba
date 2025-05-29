import os
import sys

def validate_pythonpath(script_name):
    pythonpath = os.environ.get('PYTHONPATH', '')
    if not pythonpath.startswith('sanshutsu_kun'):
        print(f'エラー: PYTHONPATH=sanshutsu_kun を付けて実行してください。\n例: PYTHONPATH=sanshutsu_kun python {script_name} days=90')
        sys.exit(1)

def parse_days_arg(argv, script_name):
    days_arg = [arg for arg in argv if arg.startswith('days=')]
    if days_arg:
        try:
            return int(days_arg[0].split('=')[1])
        except Exception:
            print('daysパラメータの指定が不正です。例: days=90')
            sys.exit(1)
    else:
        print(f'エラー: 日数をdays=XXの形式で指定してください（例: python {script_name} days=90）')
        sys.exit(1) 