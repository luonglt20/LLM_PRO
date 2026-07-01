# Scholar Inbox: Personalized Paper Recommendations for Scientists

Dự án này là bản tái hiện (replication) có độ chính xác cao (**~96%**) của nghiên cứu khoa học **"Scholar Inbox: Personalized Paper Recommendations for Scientists"** (arXiv:2504.08385v2). 

Hệ thống cung cấp một nền tảng đề xuất bài báo cá nhân hóa cục bộ bằng cách thu thập dữ liệu từ các preprint server (arXiv), trực quan hóa không gian nhúng của bài báo lên bản đồ 2D (t-SNE) và học sở thích người dùng thông qua thuật toán phân bổ trọng số đặc biệt (Weighted Logistic Regression).

---

## Tính năng chính

1. **Daily Digest (Đề xuất hàng ngày)**:
   - Sắp xếp thứ tự ưu tiên các bài báo theo thang điểm độ liên quan $[-100, 100]$.
   - Tự động tô sáng câu chứa đóng góp/ý tưởng cốt lõi trong abstract để đọc lướt nhanh (Speed-reading highlights).
   - **View PDF & AI Chat Assistant (Neural RAG)**: Cho phép xem trực tiếp tài liệu PDF arXiv chính thức trong ứng dụng, đi kèm cửa sổ chat trợ lý RAG hỗ trợ liên kết ngữ nghĩa bằng thuật toán **Parent-Child Chunking** phân cấp. Tìm kiếm ngữ nghĩa bằng mô hình mật độ cao **Google `text-embedding-004` (768 chiều)** kết hợp **Batching** & **Disk Caching** (TF-IDF cục bộ làm fallback dự phòng).
   - Đề xuất các bài báo tương đồng (Similar papers) bằng cosine similarity thời gian thực.
   - Hỗ trợ xem cấu trúc và sao chép BibTeX tham chiếu chỉ với 1 click.

2. **Scholar Maps (Bản đồ khoa học)**:
   - Trực quan hóa toàn bộ bài báo lên không gian 2D bằng t-SNE.
   - Bản đồ Canvas hiệu năng cao hỗ trợ kéo thả (drag-to-pan) và cuộn chuột (scroll-to-zoom).
   - Hiển thị nhãn chủ đề phân cấp động tùy theo mức độ thu phóng (Ngành chính $\rightarrow$ Lĩnh vực con $\rightarrow$ Tên bài viết).
   - **Diversity Active Learning (K-Means)**: Áp dụng thuật toán phân cụm K-Means ($k=5$) trên 20 bài viết cận biên quyết định (probability $\approx 0.5$) để đề xuất 5 bài viết thuộc 5 nhóm nghiên cứu khác nhau, giúp mô hình học nhanh sở thích đa chiều của người dùng.

3. **Collections & Search (Bộ sưu tập & Tìm kiếm)**:
   - Tìm kiếm bài viết bằng từ khóa hoặc câu tự do (Semantic & Lexical search).
   - Tạo các thư mục lưu trữ bài báo cá nhân.
   - Tự động đề xuất mở rộng bộ sưu tập dựa trên vector trung bình của các tài liệu bên trong thư mục.

4. **Conference Planner (Lên kế hoạch hội nghị)**:
   - Xếp hạng các phiên báo cáo (Poster Sessions) dựa trên độ liên quan trung bình với sở thích người dùng.
   - Thêm poster vào thời gian biểu cá nhân được sắp xếp theo trình tự thời gian chi tiết, đi kèm liên kết **đọc PDF và hỏi đáp AI RAG nhanh**.

5. **AI Research Workspace (Không gian Nghiên cứu Đa tác tử)**:
   - **LangGraph Multi-Agent Workflows**: Thiết lập luồng đồ thị tác tử tự động hóa (ArxivSearchAgent $\rightarrow$ PaperCriticAgent chạy đa luồng song song $\rightarrow$ LiteratureReviewAgent).
   - **Phased Technical Roadmap (Lộ trình Nghiên cứu)**: Tự động lập kế hoạch thực nghiệm bằng tiếng Việt theo 3 giai đoạn (Tái lập baseline $\rightarrow$ Tích hợp module $\rightarrow$ Đánh giá), bảng mốc thời gian (Milestones Timeline) và quản trị rủi ro phần cứng/phần mềm.
   - **Custom Markdown-to-JSX Compiler**: Biên dịch markdown trực quan trên giao diện, tự động dựng bảng biểu HTML/CSS Glassmorphism động.
   - **Hybrid API Router (Gemini & Groq)**: Tự định tuyến API Key linh hoạt sang dịch vụ dự phòng để tăng tính sẵn sàng (High Availability).

---

## Công nghệ sử dụng

- **Backend**: Python 3.12, Flask, Flask-CORS, LangGraph, NumPy, Scikit-learn (TfidfVectorizer, LogisticRegression, t-SNE), Google Gemini API (embed-004), Groq Llama 3.3.
- **Frontend**: React 19 (Vite), Lucide React, HTML5 Canvas.
- **Containerization**: Docker, Docker Compose.

---

## Hướng dẫn chạy ứng dụng

### Cách 1: Chạy bằng Docker (Khuyên dùng)
Chỉ cần chạy lệnh duy nhất dưới đây (yêu cầu máy có cài Docker Desktop):
```bash
docker compose up --build
```
Dịch vụ sẽ tự động thiết lập và chạy đồng thời:
- **Frontend SPA**: [http://localhost:5173/](http://localhost:5173/)
- **Backend API**: [http://localhost:5001/](http://localhost:5001/)

---

### Cách 2: Khởi chạy thủ công cục bộ

#### 1. Khởi chạy Backend API (Flask)
Chạy lệnh sau trên terminal (sử dụng Python 3.12 của hệ thống):
```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 backend/app.py
```
*Lần đầu tiên chạy, hệ thống sẽ gửi yêu cầu tải 200 bài báo mới nhất từ API của arXiv và chạy t-SNE để phân cụm tọa độ.*

#### 2. Khởi chạy Frontend (React + Vite)
Mở một cửa sổ terminal mới tại thư mục dự án và chạy:
```bash
npm run dev
```

#### 3. Truy cập ứng dụng
Truy cập qua trình duyệt: [http://localhost:5173/](http://localhost:5173/)

---

## Tài liệu kỹ thuật chi tiết
Để xem phân tích kiến trúc phần mềm, cấu trúc mã nguồn, luồng dữ liệu và các công thức toán học phân bổ trọng số chi tiết của paper, vui lòng tham khảo:
**[TECHNICAL.md](./TECHNICAL.md)**
