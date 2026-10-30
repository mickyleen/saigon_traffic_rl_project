# Hướng dẫn chạy Mini Project SaigonTraffic-RL trên máy cá nhân (Local)

Bộ project gồm 2 thành phần có thể chạy độc lập:

## Cách 1: Chạy Bản Mô Phỏng Trực Quan (Web App HTML5 Canvas)
1. Giải nén file `saigon_traffic_rl_project.zip`.
2. Nhấp đúp chuột trực tiếp vào file **`saigon_traffic_rl_simulator.html`** để mở bằng trình duyệt bất kỳ (Chrome, Edge, Firefox, Safari) — **không cần cài đặt môi trường phức tạp**.
3. Hoặc nếu dùng VS Code, nhấn chuột phải vào `saigon_traffic_rl_simulator.html` -> chọn **Open with Live Server** (hoặc chạy `python -m http.server 8000` rồi mở `http://localhost:8000/saigon_traffic_rl_simulator.html`).

## Cách 2: Chạy Code Huấn Luyện Python & Xuất Biểu Đồ Báo Cáo (`saigon_traffic_rl.py`)
1. Cài đặt thư viện (chỉ cần `numpy` và `matplotlib`, rất nhẹ):
   ```bash
   pip install -r requirements.txt
   ```
2. Chạy kịch bản huấn luyện 200 Episodes và so sánh thực nghiệm:
   ```bash
   python saigon_traffic_rl.py
   ```
3. Kết quả sẽ in bảng thống kê trực tiếp ra Terminal và tự động xuất file ảnh biểu đồ **`saigon_traffic_rl_results.png`** (độ phân giải 150 DPI) để chèn vào báo cáo môn Reinforcement Learning.
