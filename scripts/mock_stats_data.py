import os
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.database.opensearch import OpenSearchKB


def generate_mock_stats(days_back=14):
    kb = OpenSearchKB(
        index_name=os.getenv("OP_STATS_INDEX", "stats"),
        embedding_dim=1,
    )

    docs = []
    now = datetime.now(timezone.utc)

    # Chúng ta sinh data từ quá khứ đến hôm qua để không đè lên hôm nay
    past_dates = [
        (now - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_back, 0, -1)
    ]

    daily_total_dict = {}
    daily_false_dict = {}

    for date_str in past_dates:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

        dung = random.randint(5, 50)
        sai = random.randint(10, 80)
        ccc = random.randint(0, 20)

        total = dung + sai + ccc

        stats_24h = {
            "đúng": dung,
            "sai": sai,
            "chưa chắc chắn": ccc,
            "percent_đúng": round(dung / total * 100, 2) if total else 0,
            "percent_sai": round(sai / total * 100, 2) if total else 0,
        }

        daily_total_dict[date_str] = total
        daily_false_dict[date_str] = sai

        # Chỉ giữ lại tối đa 7 ngày gần nhất trong mảng hiển thị biểu đồ
        keys = sorted(daily_total_dict.keys())
        if len(keys) > 7:
            for k in keys[:-7]:
                daily_total_dict.pop(k, None)
                daily_false_dict.pop(k, None)

        doc = {
            "id": date_str,
            "date": date_str,
            "timestamp": target_date.isoformat(),
            "recent_claims": [],  # Bạn có thể mock thêm chi tiết nếu cần thiết
            "stats_24h": stats_24h,
            "daily_total": dict(daily_total_dict),
            "daily_false": dict(daily_false_dict),
            "cluster": {},  # mock trống, calculate_claims_stats.py sẽ điền thật
        }
        docs.append(doc)

    print(
        f"Bắt đầu đẩy {len(docs)} document lịch sử thống kê vào OpenSearch index '{kb.index}'..."
    )
    try:
        # insert_many() tự kiểm tra và tạo index nếu chưa tồn tại
        res = kb.insert_many(docs, upsert=True)
        print("✅ Kết quả đẩy data:", res)
        print(
            "Đã cấp đủ historical data cho API /claims/stats! Bạn có thể test gọi API để thấy biểu đồ ngay."
        )
    except Exception as e:
        print("❌ Lỗi khi gửi dữ liệu:", str(e))


if __name__ == "__main__":
    generate_mock_stats(14)
