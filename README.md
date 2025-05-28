# Django Project - Hướng dẫn cài đặt và chạy

## 1. Tạo môi trường ảo

```bash
python3 -m venv venv
source venv/bin/activate      # Trên Windows dùng: source venv/Scripts/activate
```

## 2. Cài đặt thư viện cần thiết

```bash
pip install django
pip install django-cors-headers
```

## 3. Ghi lại các thư viện vào file requirements.txt

```bash
pip freeze > requirements.txt
```

## 4. Cài lại thư viện từ file requirements.txt (nếu cần)

```bash
pip install -r requirements.txt
```

## 5. Khởi tạo app mới tên là api

```bash
django-admin startapp api
```

## 6. Tạo migration và áp dụng database

```bash
python manage.py makemigrations
python manage.py migrate
```

## 7. Tạo tài khoản quản trị (admin)

```bash
python manage.py createsuperuser
```

## 8. Chạy server

```bash
python manage.py runserver
```
