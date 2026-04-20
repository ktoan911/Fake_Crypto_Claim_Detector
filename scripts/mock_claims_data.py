import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.database.opensearch import OpenSearchKB


def generate_mock_claims(num_claims=100):
    # Khởi tạo kết nối tới OpenSearch index 'claims'
    kb = OpenSearchKB(
        index_name=os.getenv("OP_CLAIMS_INDEX", "claims"),
        embedding_dim=1,  # Bypass vector errors
    )

    # Các nhãn verdict trùng khớp với code gốc lấy từ OpenSearch
    verdicts = ["Đúng", "Sai", "Chưa chắc chắn"]

    claims = []
    now = datetime.now(timezone.utc)

    for i in range(num_claims):
        # Random lùi thời gian bất kỳ trong vòng 7 ngày qua (để có data rải rác test biểu đồ 7 ngày)
        random_hours_ago = random.randint(0, 7 * 24)
        random_minutes_ago = random.randint(0, 60)
        past_time = now - timedelta(hours=random_hours_ago, minutes=random_minutes_ago)

        doc_id = str(uuid.uuid4())

        templates = [
            f"Đầu tư tiền mã hóa nhận lãi suất {random.randint(50, 400)}% mỗi tháng.",
            f"Chuyên gia khuyến nghị mua cổ phiếu tăng trưởng, cam kết lãi {random.randint(20, 100)}% trong 1 tuần.",
            "Tin đồn ngân hàng lớn gặp sự cố thanh khoản khiến khách hàng ồ ạt rút tiền.",
            f"Nhận ngay {random.randint(10, 50)} triệu VNĐ khi truy cập đường link và nhập thông tin cá nhân.",
            "Lãnh đạo tập đoàn lớn bị điều tra do nghi vấn thao túng thị trường chứng khoán.",
            f"Dự án bất động sản cam kết sinh lời {random.randint(30, 70)}% trong thời gian ngắn.",
            "Sàn Forex quảng cáo lợi nhuận cao, không rủi ro và có thể rút tiền bất kỳ lúc nào.",
            f"Ứng dụng vay tiền online giải ngân trong {random.randint(5, 30)} phút với lãi suất lên tới {random.randint(100, 500)}%/năm.",
            f"Quỹ đầu tư huy động vốn với cam kết trả lãi {random.randint(10, 30)}% mỗi tháng.",
            # thêm đa dạng chủ đề
            f"Đồng coin mới được quảng cáo sẽ tăng {random.randint(5, 50)} lần trong thời gian ngắn.",
            f"Doanh nghiệp công bố lợi nhuận tăng đột biến {random.randint(100, 500)}% nhưng không có báo cáo kiểm toán.",
            "Cổ phiếu bị đẩy giá mạnh trước khi đội lái xả hàng.",
            "Tin đồn sáp nhập giữa hai ngân hàng lớn khiến cổ phiếu tăng trần liên tục.",
            "Ứng dụng đầu tư giả mạo yêu cầu nạp tiền để mở khóa tính năng rút vốn.",
            "Dự án NFT hứa hẹn lợi nhuận cao nhưng không có sản phẩm thực tế.",
            "Sàn tiền điện tử đột ngột khóa tài khoản người dùng mà không rõ lý do.",
            f"Chương trình hoàn tiền lên đến {random.randint(50, 100)}% khi giao dịch qua app lạ.",
            "Công ty đa cấp tài chính tuyển người với thu nhập thụ động hàng chục triệu mỗi tháng.",
            "Token mới ra mắt được quảng bá bởi KOL và tăng giá bất thường.",
            "Tin nội bộ cho biết doanh nghiệp sắp phá sản nhưng chưa công bố chính thức.",
            "Ứng dụng giả mạo ngân hàng yêu cầu cập nhật thông tin để tránh bị khóa tài khoản.",
            f"Đầu tư vàng online cam kết lợi nhuận {random.randint(10, 40)}% mỗi tháng.",
            "Trang web giả mạo ví điện tử yêu cầu nhập mã OTP để xác minh tài khoản.",
            "Tin đồn tăng lãi suất khiến thị trường chứng khoán giảm mạnh.",
            "Công ty phát hành trái phiếu với lãi suất cao bất thường nhưng không có tài sản đảm bảo.",
            "Dự án DeFi cam kết lợi nhuận cao nhưng không minh bạch dòng tiền.",
            "Ứng dụng đầu tư tự động hứa hẹn lợi nhuận ổn định mỗi ngày.",
            "Cổ phiếu penny được quảng bá mạnh trên mạng xã hội để thu hút nhà đầu tư nhỏ lẻ.",
            "Tin đồn doanh nghiệp bị thanh tra thuế khiến cổ phiếu lao dốc.",
            "Chương trình đầu tư nhận thưởng khi giới thiệu thêm người tham gia.",
            "Sàn giao dịch bị tố thao túng giá khiến nhà đầu tư thua lỗ.",
            "Token được tạo ra chỉ để lừa đảo và rút thanh khoản.",
            "Ứng dụng vay tiền yêu cầu truy cập danh bạ và đe dọa người thân.",
            "Tin giả về chính sách tài chính gây hoang mang thị trường.",
            "Công ty công nghệ bị cáo buộc thổi phồng doanh thu.",
            "Quỹ đầu tư không rõ nguồn gốc quảng cáo lợi nhuận cao.",
            "Dự án bất động sản chưa có pháp lý vẫn mở bán rầm rộ.",
            "Ứng dụng giả mạo sàn chứng khoán yêu cầu nạp tiền để giao dịch.",
            "Tin đồn phá giá tiền tệ khiến người dân tích trữ ngoại tệ.",
            "Cổ phiếu tăng trần nhiều phiên liên tiếp do tin nội gián.",
            "Doanh nghiệp bị nghi ngờ gian lận báo cáo tài chính.",
            "Ứng dụng đầu tư yêu cầu nâng cấp tài khoản để rút tiền.",
            "Token bị rug pull sau khi thu hút lượng lớn nhà đầu tư.",
            "Tin đồn thay đổi chính sách thuế ảnh hưởng đến thị trường.",
            "Công ty tài chính cung cấp khoản vay nhanh nhưng phí ẩn rất cao.",
            "Sàn giao dịch yêu cầu đóng phí trước khi rút tiền.",
            "Doanh nghiệp công bố dự án lớn nhưng không có bằng chứng triển khai.",
            "Tin đồn CEO từ chức khiến cổ phiếu giảm mạnh.",
            "Ứng dụng đầu tư yêu cầu xác minh danh tính bằng cách gửi ảnh nhạy cảm.",
            "Cổ phiếu bị thao túng bởi nhóm nhà đầu tư lớn.",
            "Chương trình đầu tư lợi nhuận cao nhưng yêu cầu giữ tiền trong thời gian dài.",
            "Dự án blockchain không có sản phẩm thật nhưng vẫn gọi vốn.",
            "Tin giả về hợp tác giữa các tập đoàn lớn để đẩy giá cổ phiếu.",
            "Ứng dụng tài chính giả mạo yêu cầu cung cấp thông tin thẻ ngân hàng.",
        ]

        doc = {
            "id": doc_id,  # Thêm id để OpenSearchKB.insert_many xử lý upsert được
            "claim": random.choice(templates),
            "verdict": random.choice(verdicts),
            "checked_at": past_time.isoformat(),
            "evidence": ["Nguồn bịa đặt", "Nguồn tham khảo phụ"],
        }
        claims.append(doc)

    print(f"Bắt đầu đẩy {num_claims} mock claims vào OpenSearch index '{kb.index}'...")

    try:
        # Gọi hàm gửi hàng loạt
        res = kb.insert_many(claims, upsert=True)
        print("✅ Kết quả đẩy data:", res)
        print(
            "Bây giờ bạn có thể chạy: \n  python scripts/calculate_claims_stats.py\nđể xem kết quả thống kê."
        )
    except Exception as e:
        print("❌ Lỗi khi gửi dữ liệu:", str(e))


if __name__ == "__main__":
    generate_mock_claims(4000)
