
import requests
import json
import traceback
import time
import sys
from typing import Callable, Any

def post_message(message: str, webhook_url: str ):
    """
    Slackにメッセージを送信する\\
    params:
    - message: 送信するメッセージ
    - webhook_url: SlackのWebhook URL
    """
    
    payload = {
        "text": message
    }
    response = requests.post(
        webhook_url, data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )
    if response.status_code == 200:
        print("Slack notification sent successfully")
    else:
        print(f"Failed to send Slack notification: {response.status_code}, {response.text}")
        
        
def send_notification(file: str,webhook_url: str, method: Callable[..., Any], *args, **kwargs):
    """
    メソッドの実行結果をSlackに通知する\\
    params:
    - file: 実行ファイルのパス
    - webhook_url: SlackのWebhook URL
    - method: 実行するメソッド
    - *args: methodの引数
    - **kwargs: methodのキーワード引数
    """
    try:
        start = time.time()
        method(*args, **kwargs)
    except Exception as e:
        error_message = traceback.format_exc()
        time_taken = time.time() - start
        print(error_message, file=sys.stderr)
        post_message(
            message=f"===\nPythonスクリプトの処理がエラー終了しました。\nError: {error_message}\ntime_taken:{time_taken} s \n===",
             webhook_url=webhook_url)
        sys.exit(1)
    else:
        time_taken = time.time() - start
        post_message(
            message=f'===\nPythonスクリプトの処理が完了しました。\n\tFile "{file}"\n\t\t{method.__name__}\ntime_taken:{time_taken} s\n===',
            webhook_url=webhook_url)