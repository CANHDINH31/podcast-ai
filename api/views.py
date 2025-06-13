from rest_framework.decorators import api_view
from rest_framework.response import Response
from pyvi import ViTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import re
import os
from django.conf import settings

STOPWORDS_FILE = os.path.join(settings.BASE_DIR, 'api', 'data', 'vietnamese-stopwords.txt')

@api_view(['POST'])
def sumary(request):
    data = request.data.get('data')
    if not data or 'content' not in data:
        return Response({"message": "Thiếu nội dung cần tóm tắt."}, status=400)

    content = data['content']
    

    # Tiền xử lý văn bản
    contents_parsed = clean_text_vi(content)

    # Tách các câu trong văn bản, làm sạch và lọc câu ngắn
    sentences = [s for s in viet_sent_tokenize(contents_parsed) if len(s) > 5]


    if len(sentences) < 2:
        return Response({"message": "Không đủ dữ liệu để tóm tắt."}, status=400)

    # Tách từ trong câu
    sentences_tokenized = [remove_stop_words(ViTokenizer.tokenize(sentence)) for sentence in sentences]

    # Khởi tạo TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Chuyển các câu thành vector
    X = vectorizer.fit_transform(sentences_tokenized)

    # Chuyển thành mảng numpy (tùy chọn)
    X = X.toarray()

    # === Tự động chọn số cụm phù hợp bằng silhouette_score ===
    best_k = 2
    best_score = -1
    max_k = min(10, len(sentences))  # Giới hạn cụm tối đa
    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init='auto')
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    n_clusters = best_k
    
    # Huấn luyện lại với số cụm tối ưu
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans = kmeans.fit(X)

    # Xây dựng đoạn văn bản tóm tắt
    ordering = sorted(
        range(n_clusters),
        key=lambda cluster_id: np.mean(np.where(kmeans.labels_ == cluster_id)[0])
    )
    
    summary_sentences = []
    for cluster_id in ordering:
        indices_in_cluster = np.where(kmeans.labels_ == cluster_id)[0]
        if len(indices_in_cluster) == 0:
            continue
        closest_idx = indices_in_cluster[np.argmin(np.linalg.norm(X[indices_in_cluster] - kmeans.cluster_centers_[cluster_id], axis=1))] 
        summary_sentences.append(sentences[closest_idx])
    summary = ' '.join(summary_sentences)
    
    # Làm sạch lần cuối
    summary = normalize_summary_text(summary)

    return Response({"message": f"{summary}"})

def remove_stop_words(tokens):
    stop_words = load_stopwords()
    return ' '.join([word for word in tokens.split() if word.lower() not in stop_words])

def load_stopwords():
    with open(STOPWORDS_FILE, encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

# Tiền xử lý tiếng việt
# Thay thế cụm viết tắt, từ rác phổ biến (ví dụ: “ko” → “không”, “j” → “gì”, v.v.)
# Loại bỏ ký tự đặc biệt nhưng giữ lại ?, !, . để phân biệt câu
# Loại bỏ từ dư khi tokenize bị lỗi
def clean_text_vi(text):
    text = text.lower()
    text = re.sub(r'\s*\n\s*', '. ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'[!?]', '.', text)  # gom các dấu ?! lại thành .
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\–\-àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.strip()

def normalize_summary_text(text):
    """
    Làm sạch và chuẩn hóa văn bản tóm tắt:
    - Chuẩn hóa dấu chấm
    - Xóa khoảng trắng dư thừa
    - Viết hoa chữ cái đầu mỗi câu
    - Đảm bảo có dấu chấm kết thúc
    """
    # Chuẩn hóa dấu chấm
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return ''

    # Viết hoa sau mỗi dấu chấm
    sentences = [s.strip().capitalize() for s in text.split('.')]
    sentences = [s for s in sentences if s]  # bỏ câu rỗng
    text = '. '.join(sentences)

    # Đảm bảo kết thúc bằng dấu chấm
    if not text.endswith('.'):
        text += '.'

    return text

def viet_sent_tokenize(text):
    # Tách câu đơn giản theo dấu ., !, ?
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]