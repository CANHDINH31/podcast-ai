from rest_framework.decorators import api_view
from rest_framework.response import Response
import nltk
from gensim.models import KeyedVectors 
from pyvi import ViTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 

nltk.download('punkt_tab')

@api_view(['POST'])
def sumary(request):
    data = request.data.get('data')
    if not data or 'content' not in data:
        return Response({"message": "Thiếu nội dung cần tóm tắt."}, status=400)

    content = data['content']

    # Tiền xử lý văn bản
    contents_parsed = content.lower() #Biến đổi hết thành chữ thường
    contents_parsed = contents_parsed.replace('\n', '. ') #Đổi các ký tự xuống dòng thành chấm câu
    contents_parsed = contents_parsed.strip() 

    # Tách các câu trong văn bản
    sentences = nltk.sent_tokenize(contents_parsed)

   # Tách từ trong câu
    sentences_tokenized = [ViTokenizer.tokenize(sentence) for sentence in sentences]

    # Khởi tạo TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Chuyển các câu thành vector
    X = vectorizer.fit_transform(sentences_tokenized)

    # Chuyển thành mảng numpy (tùy chọn)
    X = X.toarray()
    
    # Phân cụm
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans = kmeans.fit(X)

    # Xây dựng đoạn văn bản tóm tắt

    avg = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = ' '.join([sentences[closest[idx]] for idx in ordering])



    return Response({"message": f"{summary}"})
