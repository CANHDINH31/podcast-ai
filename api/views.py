from rest_framework.decorators import api_view
from rest_framework.response import Response
import nltk
from gensim.models import KeyedVectors 
from pyvi import ViTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import re

@api_view(['POST'])
def sumary(request):
    data = request.data.get('data')
    if not data or 'content' not in data:
        return Response({"message": "Thiếu nội dung cần tóm tắt."}, status=400)
   

    content = data['content']
    

    # Tiền xử lý văn bản
    contents_parsed = content.lower()
    contents_parsed = re.sub(r'\s*\n\s*', '. ', contents_parsed)  # xóa khoảng trắng trước/sau \n
    contents_parsed = re.sub(r'\.{2,}', '.', contents_parsed)     # loại bỏ dấu chấm lặp lại
    contents_parsed = re.sub(r'\s*\.\s*', '. ', contents_parsed)  # chuẩn hóa dấu chấm và khoảng trắng
    contents_parsed = contents_parsed.strip()

    # Tách các câu trong văn bản
    sentences = nltk.sent_tokenize(contents_parsed)
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

    avg = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = ' '.join([sentences[closest[idx]] for idx in ordering])
    
    # Làm sạch lần cuối
    summary = re.sub(r'\.{2,}', '.', summary)
    summary = re.sub(r'\s*\.\s*', '. ', summary)
    summary = summary.strip('. ') + '.'  # Đảm bảo có dấu chấm kết thúc



    return Response({"message": f"{summary}"})



def remove_stop_words(tokens):
    stop_words = set(['là', 'có', 'của', 'và', 'lúc', 'khi', 'với', 'cho', 'được', 'này', 'rằng', 'một', 'những'])  
    return ' '.join([word for word in tokens.split() if word.lower() not in stop_words])