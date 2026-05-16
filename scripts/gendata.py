import json
import os
import random
import time
from collections import Counter
    
from dotenv import load_dotenv

from together import Together
from tqdm import tqdm

load_dotenv()

# ======================
# CONFIG
# ======================
N_SAMPLES = 150
BASE_SLEEP = 2
MAX_RETRIES = 6
TIMEOUT_PER_CALL = 30

LABELS = ["đúng", "sai", "thiếu thông tin"]

LABEL_FILE_MAP = {
    "đúng": "dataset_dung.json",
    "sai": "dataset_sai.json",
    "thiếu thông tin": "dataset_thieu_thong_tin.json",
}

FINAL_OUTPUT = "dataset_all.json"


# ======================
# Prompt
# ======================
PROMPT_TEMPLATE = """
Bạn là hệ thống tạo dữ liệu fact-checking về tin tức tài chính ngân hàng tại việt nam.

Sinh ra 1 sample JSON DUY NHẤT với format:

{{
  "claim": "...",
  "evidence": [
    {{
      "content": "...",
      "timestamp": "YYYY-MM-DD"
    }}
  ],
  "label": "{target_label}"
}}

Yêu cầu:
- label PHẢI là "{target_label}"
- claim PHẢI xoay quanh chủ đề: "{topic}"
- KHÔNG được tạo claim ngoài chủ đề
- claim phải cụ thể, tự nhiên, giống tin tức thật

- 2-4 evidence
- timestamp phải khác nhau
- evidence có thể mâu thuẫn

Quy tắc theo label:
- Nếu label = "đúng":
  evidence hỗ trợ claim

- Nếu label = "sai":
  evidence phản bác claim rõ ràng

- Nếu label = "thiếu thông tin":
  evidence không đủ để kết luận

- ưu tiên evidence mới hơn đáng tin hơn
- không giải thích
- chỉ trả JSON hợp lệ
"""

# ======================
# Client
# ======================
client = Together(api_key="tgp_v1_usKc8DNdPBLd0axzXnxvAr4zxLjtYV8ZNquMCbAnNH8")


# ======================
# Utils
# ======================
def load_existing(file):
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_json(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ======================
# Generate
# ======================
def generate_sample(topic, target_label):

    prompt = PROMPT_TEMPLATE.format(topic=topic, target_label=target_label)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            content = resp.choices[0].message.content

            if not content:
                raise ValueError("Empty response")

            return content

        except Exception as e:
            wait = (2**attempt) + random.uniform(1, 3)

            print(f"⚠️ API lỗi ({attempt + 1}/{MAX_RETRIES}): {e}")
            print(f"⏳ Sleep {wait:.2f}s...")

            time.sleep(wait)

    return None


# ======================
# Parse
# ======================
def parse_sample(text):

    try:
        data = json.loads(text)

        assert "claim" in data
        assert "evidence" in data
        assert "label" in data

        assert data["label"] in LABELS

        assert isinstance(data["evidence"], list)
        assert 2 <= len(data["evidence"]) <= 4

        timestamps = set()

        for ev in data["evidence"]:
            assert "content" in ev
            assert "timestamp" in ev

            timestamps.add(ev["timestamp"])

        assert len(timestamps) == len(data["evidence"])

        return data

    except Exception as e:
        print("❌ Parse fail:", e)

        return None


# ======================
# Create generation plan
# ======================
def create_generation_plan(topics):

    per_label = N_SAMPLES // len(LABELS)

    # mỗi label cần 50 sample
    label_pool = []

    for label in LABELS:
        label_pool.extend([label] * per_label)

    random.shuffle(label_pool)

    generation_plan = []

    # ======================
    # đảm bảo mọi topic xuất hiện ít nhất 1 lần
    # ======================
    for topic in topics:
        label = label_pool.pop()

        generation_plan.append({"topic": topic, "label": label})

    # ======================
    # còn thiếu bao nhiêu sample thì random topic
    # ======================
    remaining = N_SAMPLES - len(generation_plan)

    for _ in range(remaining):
        topic = random.choice(topics)

        label = label_pool.pop()

        generation_plan.append({"topic": topic, "label": label})

    random.shuffle(generation_plan)

    return generation_plan


# ======================
# Build dataset
# ======================
TOPICS = [
    "NHNN tiếp tục giữ nguyên lãi suất điều hành để hỗ trợ tăng trưởng kinh tế quý II/2026.",
    "Tỷ giá USD/VND biến động nhẹ do áp lực từ chỉ số DXY quốc tế.",
    "NHNN bơm ròng hơn 20.000 tỷ đồng qua kênh OMO để hỗ trợ thanh khoản hệ thống.",
    "Dự trữ ngoại hối của Việt Nam đạt mốc kỷ lục mới trong tháng 5/2026.",
    "Triển khai quyết liệt Nghị định mới về quản lý thị trường vàng, thu hẹp khoảng cách vàng SJC và thế giới.",
    "NHNN yêu cầu các tổ chức tín dụng (TCTD) tiết giảm chi phí để giảm lãi suất cho vay thêm 0,5%.",
    "Dự thảo quy định mới về giới hạn tỷ lệ sở hữu cổ phần tại các ngân hàng thương mại (NHTM).",
    "Tín dụng toàn hệ thống tính đến 10/5/2026 tăng 5,8% so với đầu năm.",
    "Việt Nam tiếp tục nằm trong danh sách theo dõi chính sách tiền tệ của Bộ Tài chính Mỹ nhưng với đánh giá tích cực.",
    "NHNN đẩy mạnh thanh tra các tiệm vàng không có giấy phép kinh doanh ngoại tệ.",
    "Đề xuất kéo dài thời gian cơ cấu lại thời hạn trả nợ theo Thông tư 02 đến hết năm 2026.",
    "Tăng cường kiểm soát dòng vốn chảy vào các lĩnh vực rủi ro như bất động sản đầu cơ.",
    "Thúc đẩy thanh toán không dùng tiền mặt tại các vùng sâu, vùng xa qua Mobile Money.",
    "NHNN phối hợp với Bộ Công an triển khai xác thực sinh trắc học bắt buộc cho giao dịch trên 10 triệu đồng.",
    "Hoàn thiện khung pháp lý cho cơ chế thử nghiệm (Sandbox) Fintech.",
    "Nhắc nhở các ngân hàng về việc đảm bảo an toàn hệ thống trong dịp lễ 1/5.",
    "Chỉ đạo xử lý dứt điểm các ngân hàng yếu kém trong diện kiểm soát đặc biệt.",
    "Phát hành tín phiếu NHNN để hút bớt tiền dư thừa, ổn định lạm phát mục tiêu dưới 4%.",
    "Cập nhật danh sách các ngân hàng có tầm quan trọng hệ thống năm 2026.",
    "Hợp tác tài chính đa phương giữa NHNN và các ngân hàng trung ương ASEAN+3.",
    "Vietcombank báo lãi kỷ lục trong quý I/2026, dẫn đầu hệ thống về lợi nhuận.",
    "BIDV đẩy mạnh gói tín dụng xanh 10.000 tỷ đồng cho các dự án năng lượng tái tạo.",
    "Agribank triển khai xe ngân hàng lưu động phục vụ bà con nông dân mùa thu hoạch.",
    "VietinBank hoàn tất việc tăng vốn điều lệ thông qua chia cổ tức bằng cổ phiếu.",
    "Techcombank ra mắt tính năng quản lý tài chính cá nhân tích hợp AI thế hệ mới.",
    "MB Bank đạt mốc 30 triệu người dùng trên ứng dụng ngân hàng số.",
    "VPBank ký kết hợp đồng vay vốn quốc tế trị giá 500 triệu USD để tài trợ SME.",
    "ACB duy trì tỷ lệ nợ xấu dưới 1%, thấp nhất nhóm ngân hàng tư nhân.",
    "Sacombank đấu giá thành công các khoản nợ xấu lớn, đẩy nhanh tiến độ tái cơ cấu.",
    "VIB tập trung vào mảng cho vay ô tô và thẻ tín dụng với nhiều ưu đãi hè 2026.",
    "TPBank ứng dụng Robot (RPA) vào quy trình phê duyệt khoản vay tự động.",
    "HDBank mở rộng mạng lưới tại các khu công nghiệp trọng điểm phía Nam.",
    "LPBank đổi tên thương hiệu và công bố chiến lược phát triển giai đoạn mới.",
    "Nam A Bank chính thức niêm yết trên sàn HOSE với thanh khoản ấn tượng.",
    "OCB đẩy mạnh tài trợ chuỗi cung ứng (Supply Chain Finance) cho các doanh nghiệp xuất khẩu.",
    "MSB ra mắt thẻ tín dụng chuyên biệt cho Gen Z với tính năng hoàn tiền tùy chỉnh.",
    "SeABank nhận khoản đầu tư từ các tổ chức tài chính phát triển châu Âu.",
    "Eximbank thay đổi nhân sự cấp cao tại Đại hội đồng cổ đông thường niên.",
    "Bac A Bank tập trung vốn cho các dự án nông nghiệp công nghệ cao.",
    "VietCapital Bank (Bản Việt) tăng cường hiện diện tại thị trường bán lẻ phía Bắc.",
    "Shinhan Bank Việt Nam mở thêm 3 chi nhánh mới tại Hà Nội và TP.HCM.",
    "HSBC Việt Nam cam kết hỗ trợ doanh nghiệp thực hiện báo cáo ESG.",
    "Standard Chartered dự báo kinh tế Việt Nam tăng trưởng 6,5% trong năm 2026.",
    "UOB Việt Nam đẩy mạnh số hóa quy trình mở tài khoản cho khách hàng doanh nghiệp.",
    "Các ngân hàng nhỏ đua nhau tăng lãi suất huy động để giữ chân khách hàng.",
    "VN-Index chính thức vượt ngưỡng kháng cự 1.350 điểm nhờ dòng tiền nội mạnh mẽ.",
    "Khối ngoại quay lại mua ròng sau chuỗi ngày bán tháo kéo dài.",
    "Hoàn tất triển khai hệ thống giao dịch mới (KRX) giúp giao dịch T+0 trở nên gần hơn.",
    "Nhiều doanh nghiệp lớn công bố kế hoạch chia cổ tức bằng tiền mặt tỷ lệ cao.",
    "Nhóm cổ phiếu ngân hàng đóng vai trò 'đầu tàu' dẫn dắt thị trường đi lên.",
    "Giá trị phát hành trái phiếu doanh nghiệp trong tháng 4 tăng 15% so với tháng trước.",
    "Thành lập Hiệp hội các nhà đầu tư trái phiếu chuyên nghiệp Việt Nam.",
    "Ủy ban Chứng khoán xử phạt nặng các hành vi thao túng giá cổ phiếu Penny.",
    "Làn sóng doanh nghiệp bất động sản mua lại trái phiếu trước hạn để giảm áp lực nợ.",
    "Quỹ ETF nội thu hút dòng vốn lớn từ các nhà đầu tư cá nhân.",
    "Cổ phiếu ngành công nghệ bứt phá nhờ làn sóng đầu tư vào AI bán dẫn.",
    "Dragon Capital dự báo lợi nhuận doanh nghiệp niêm yết tăng 20% trong năm nay.",
    "Bộ Tài chính lấy ý kiến về việc nâng hạng thị trường chứng khoán từ cận biên lên mới nổi.",
    "Xuất hiện các loại chứng chỉ quỹ mới mô phỏng chỉ số phát triển bền vững.",
    "Lượng tài khoản chứng khoán mở mới trong tháng 4 đạt mức cao nhất trong 2 năm.",
    "VN30 ghi nhận sự thay đổi cơ cấu với việc thêm mới các cổ phiếu bán lẻ.",
    "Giao dịch phái sinh tăng trưởng mạnh về khối lượng do biến động thị trường.",
    "VinFast công bố báo cáo tài chính quý I với doanh số xe điện ấn tượng tại Mỹ.",
    "Các công ty chứng khoán đua nhau hạ lãi suất Margin để cạnh tranh thị phần.",
    "Thị trường chứng khoán phái sinh bổ sung hợp đồng tương lai trái phiếu chính phủ 10 năm.",
    "Tập đoàn Hòa Phát (HPG) khởi công giai đoạn 3 của khu liên hợp Dung Quất.",
    "Sabeco và Habeco đối mặt với thách thức chi phí nguyên liệu đầu vào tăng.",
    "Cổ phiếu ngành vận tải biển hưởng lợi từ việc giá cước logistics tăng cao.",
    "FPT công bố hợp đồng tỷ USD mới trong lĩnh vực chuyển đổi số tại Nhật Bản.",
    "Kiểm soát chặt chẽ hoạt động của các hội nhóm 'phím hàng' trên mạng xã hội.",
    "MoMo chính thức tích hợp thanh toán Apple Pay cho toàn bộ hệ sinh thái.",
    "ZaloPay ra mắt dịch vụ 'Mua trước trả sau' (BNPL) hợp tác với các ngân hàng.",
    "VNPay-QR phủ sóng hơn 500.000 điểm chấp nhận thanh toán trên toàn quốc.",
    "Viettel Money triển khai rút tiền không cần thẻ tại tất cả các điểm giao dịch.",
    "Ứng dụng Blockchain vào việc truy xuất nguồn gốc trong tài trợ thương mại.",
    "Ví điện tử VNPAY đạt chứng nhận bảo mật quốc tế PCI DSS cấp độ cao nhất.",
    "Cảnh báo gia tăng các vụ lừa đảo Deepfake nhắm vào khách hàng ngân hàng.",
    "Nhiều startup Fintech Việt Nam gọi vốn thành công vòng Series B.",
    "Ra mắt nền tảng Open Banking dùng chung cho toàn ngành ngân hàng Việt Nam.",
    "Ứng dụng sinh trắc học khuôn mặt vào thanh toán tại các cửa hàng tiện lợi.",
    "Fintech Việt Nam đẩy mạnh xuất khẩu giải pháp sang thị trường Đông Nam Á.",
    "ShopeePay đẩy mạnh liên kết với các ngân hàng số để tối ưu trải nghiệm mua sắm.",
    "Ngân hàng thuần số (Digital-only bank) tại Việt Nam bắt đầu có lãi.",
    "Các giải pháp eKYC tích hợp dữ liệu dân cư quốc gia giúp giảm 90% thời gian mở thẻ.",
    "Ra mắt nền tảng kết nối cho vay ngang hàng (P2P Lending) dưới sự giám sát của NHNN.",
    "Chỉ số sản xuất công nghiệp (IIP) tháng 4 tăng trưởng 7,2% so với cùng kỳ.",
    "Xuất khẩu nông sản (gạo, sầu riêng) mang về nguồn ngoại tệ lớn cho ngân hàng.",
    "FDI vào Việt Nam trong 4 tháng đầu năm tập trung mạnh vào lĩnh vực công nghệ cao.",
    "Thị trường bất động sản phân khúc chung cư tại Hà Nội tiếp tục neo giá cao.",
    "Các dự án nhà ở xã hội được đẩy nhanh tiến độ nhờ gói tín dụng 120.000 tỷ đồng.",
    "Giá xăng dầu trong nước điều chỉnh theo xu hướng thế giới, áp lực lên CPI.",
    "Nhiều doanh nghiệp dệt may nhận đủ đơn hàng đến hết quý III/2026.",
    "Thu ngân sách nhà nước đạt hơn 45% dự toán năm chỉ sau hơn 4 tháng.",
    "Việt Nam ký kết thêm Hiệp định thương mại tự do (FTA) mới với khu vực Trung Đông.",
    "Du lịch phục hồi mạnh mẽ giúp doanh thu dịch vụ tài chính du lịch tăng vọt.",
    "Cảnh báo rủi ro bong bóng bất động sản tại các khu vực ven đô có quy hoạch mới.",
    "Các tập đoàn bán lẻ quốc tế mở rộng diện tích kho bãi tại Việt Nam.",
    "Doanh nghiệp bán lẻ (Masan, MWG) tái cấu trúc danh mục đầu tư để tối ưu lợi nhuận.",
    "Giá điện tăng nhẹ gây áp lực lên chi phí sản xuất của các doanh nghiệp thép, xi măng.",
    "Việt Nam được các tổ chức quốc tế dự báo là điểm sáng tăng trưởng kinh tế khu vực châu Á - Thái Bình Dương năm 2026.",
]


def build_dataset():

    generation_plan = create_generation_plan(TOPICS)

    dataset = load_existing(FINAL_OUTPUT)

    existing_claims = set()

    for item in dataset:
        existing_claims.add(item["claim"])

    progress = len(dataset)

    with tqdm(total=N_SAMPLES, initial=progress) as pbar:
        for task in generation_plan[progress:]:
            topic = task["topic"]
            label = task["label"]

            MAX_SAMPLE_ATTEMPTS = 10

            for sample_attempt in range(MAX_SAMPLE_ATTEMPTS):
                print(f"\n📌 Topic: {topic}")
                print(f"🏷️ Label: {label}")

                raw = generate_sample(topic=topic, target_label=label)

                if raw is None:
                    continue

                sample = parse_sample(raw)

                if not sample:
                    time.sleep(BASE_SLEEP)

                    continue

                if sample["label"] != label:
                    continue

                if sample["claim"] in existing_claims:
                    continue

                # add metadata
                sample["topic"] = topic

                dataset.append(sample)

                existing_claims.add(sample["claim"])

                save_json(FINAL_OUTPUT, dataset)

                pbar.update(1)

                break

            time.sleep(BASE_SLEEP + random.uniform(0.5, 2))

    # ======================
    # stats
    # ======================
    label_stats = Counter()

    for x in dataset:
        label_stats[x["label"]] += 1

    print("\n📊 Label Distribution")
    print(label_stats)

    print(f"\n✅ Done: {len(dataset)} samples")


# ======================
# Run
# ======================
if __name__ == "__main__":
    build_dataset()
