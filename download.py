import os
import urllib.request
import subprocess

# Sử dụng lệnh wget để tải về file
subprocess.run(['wget', 'https://public.vinai.io/PhoBERT_base_transformers.tar.gz'])

# Sử dụng lệnh tar để giải nén file
subprocess.run(['tar', '-xzvf', 'PhoBERT_base_transformers.tar.gz'])

# Tạo thư mục chứa các tệp cần thiết
os.makedirs('vncorenlp/models/wordsegmenter', exist_ok=True)

# Tải các tệp cần thiết
urllib.request.urlretrieve('https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar', 'vncorenlp/VnCoreNLP-1.1.1.jar')
urllib.request.urlretrieve('https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab', 'vncorenlp/models/wordsegmenter/vi-vocab')
urllib.request.urlretrieve('https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr', 'vncorenlp/models/wordsegmenter/wordsegmenter.rdr')
