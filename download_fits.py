import os
import requests
from bs4 import BeautifulSoup
import tarfile

def download_and_extract_files(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    for link in soup.find_all('a'):
        file_url = link.get('href')
        if file_url.endswith('.tgz'):
            download_url = url + file_url
            response = requests.get(download_url)
            tgz_path = os.path.join(dest_folder, file_url)
            
            # ダウンロードしたtgzファイルを保存
            with open(tgz_path, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded: {tgz_path}')
            
            # tgzファイルを展開
            with tarfile.open(tgz_path, 'r:gz') as tar:
                tar.extractall(path=dest_folder)
            print(f'Extracted: {tgz_path}')

            # 展開後のtgzファイルを削除（オプション）
            os.remove(tgz_path)
            print(f'Deleted: {tgz_path}')

download_and_extract_files('https://data.darts.isas.jaxa.jp/pub/hayabusa2/onc_bundle/data/v03c/data_raw_l2a/', './Ryugu_FITS/')
