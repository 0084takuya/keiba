def print_progress_bar(current, total, bar_length=30, prefix='進捗', suffix=''):
    """
    current: 現在の進捗数
    total: 全体数
    bar_length: バーの長さ（デフォルト30）
    prefix: バーの前に表示する文字列
    suffix: バーの後ろに表示する文字列
    """
    percent = int(100 * current / total)
    hashes = '#' * int(bar_length * percent / 100)
    spaces = ' ' * (bar_length - len(hashes))
    bar = f'{prefix} [{hashes}{spaces}] {percent:3d}%{suffix}'
    print(f'\r{bar}', end='', flush=True)
    if current >= total:
        print()  # 完了時のみ改行 