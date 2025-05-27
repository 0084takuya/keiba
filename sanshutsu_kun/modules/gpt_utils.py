import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def ask_gpt41(prompt, images_b64=None, model="gpt-4.1", max_tokens=1000, temperature=0.7):
    """
    prompt: str, ユーザーから送るプロンプト
    images_b64: [(filename, base64str), ...] 画像をbase64で送る場合
    """
    if images_b64:
        prompt += '\n' + '\n'.join([
            f'【{fname}】\n{b64}' for fname, b64 in images_b64
        ])
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "あなたは機械学習エンジニアです。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        advice = response['choices'][0]['message']['content']
        print('--- GPT-4.1からの改善アドバイス ---')
        print(advice)
        return advice
    except Exception as e:
        print('GPT-4.1 API呼び出しでエラー:', e)
        return f'APIエラー: {e}' 