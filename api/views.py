from rest_framework.decorators import api_view
from rest_framework.response import Response
from pyvi import ViTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

    # TF-IDF vector hóa
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences_tokenized).toarray()

    # === Thêm thông tin vị trí câu vào vector đặc trưng ===
    sentence_positions = np.array([
        [(len(sentences) - i) / len(sentences)]  # Trọng số: câu đầu cao hơn
        for i in range(len(sentences))
    ])
    X_with_pos = np.hstack([X, sentence_positions])  # Nối vector TF-IDF với trọng số vị trí

    # === Tự động chọn số cụm tốt nhất ===
    best_k = 2
    best_score = -1
    max_k = min(10, len(sentences))

    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init='auto')
            labels = km.fit_predict(X_with_pos)
            score = silhouette_score(X_with_pos, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    # Huấn luyện lại với số cụm tối ưu
    kmeans = KMeans(n_clusters=best_k, random_state=0, n_init='auto')
    kmeans.fit(X_with_pos)

    # === Xây dựng đoạn văn bản tóm tắt ===
    ordering = sorted(
        range(best_k),
        key=lambda cluster_id: np.mean(np.where(kmeans.labels_ == cluster_id)[0])
    )
    
    summary_sentences = []
    for cluster_id in ordering:
        indices_in_cluster = np.where(kmeans.labels_ == cluster_id)[0]
        if len(indices_in_cluster) == 0:
            continue
        closest_idx = indices_in_cluster[np.argmin(
            np.linalg.norm(X_with_pos[indices_in_cluster] - kmeans.cluster_centers_[cluster_id], axis=1)
        )]
        summary_sentences.append(sentences[closest_idx])

    summary = ' '.join(summary_sentences)
    summary = normalize_summary_text(summary)

    return Response({"message": f"{summary}"})


def remove_stop_words(tokens):
    stop_words = load_stopwords()
    return ' '.join([word for word in tokens.split() if word.lower() not in stop_words])


def load_stopwords():
    with open(STOPWORDS_FILE, encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def clean_text_vi(text):
    text = text.lower()
    text = re.sub(r'\s*\n\s*', '. ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'[!?]', '.', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\–\-àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.strip()


def normalize_summary_text(text):
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return ''
    text = text[0].upper() + text[1:]
    if not text.endswith('.'):
        text += '.'
    return text


def viet_sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]
