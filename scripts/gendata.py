import json
import time
import random
import os
from tqdm import tqdm
from openai import OpenAI

# ======================
# CONFIG
# ======================
N_SAMPLES = 120
BASE_SLEEP = 2
MAX_RETRIES = 6
TIMEOUT_PER_CALL = 30

LABELS = ["đúng", "sai", "thiếu thông tin"]

# map label -> filename
LABEL_FILE_MAP = {
    "đúng": "dataset_dung.json",
    "sai": "dataset_sai.json",
    "thiếu thông tin": "dataset_thieu_thong_tin.json"
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
- 2-4 evidence
- timestamp phải khác nhau
- có thể mâu thuẫn

Quy tắc theo label:
- Nếu label = "đúng": evidence hỗ trợ claim
- Nếu label = "sai": evidence phản bác claim rõ ràng
- Nếu label = "thiếu thông tin": evidence không đủ để kết luận

- ưu tiên evidence mới hơn đáng tin hơn
- không giải thích
- chỉ trả JSON hợp lệ, không markdown
"""

# ======================
# Client
# ======================
client = None
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
def generate_sample(target_label):
    prompt = PROMPT_TEMPLATE.format(target_label=target_label)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model="gemini-3.0-flash",
                messages=[{"role": "user", "content": prompt}],
                timeout=TIMEOUT_PER_CALL,
            )

            content = resp.choices[0].message.content

            if not content:
                raise ValueError("Empty response")

            return content

        except Exception as e:
            wait = (2 ** attempt) + random.uniform(1, 3)
            print(f"⚠️ API lỗi ({attempt+1}/{MAX_RETRIES}): {e}")
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
# Build dataset
# ======================
def build_dataset(n_samples=N_SAMPLES):
    assert n_samples % len(LABELS) == 0

    per_label = n_samples // len(LABELS)

    global_seen_claims = set()
    final_dataset = []

    for label in LABELS:
        print(f"\n🔹 Label: {label}")

        file_name = LABEL_FILE_MAP[label]

        # load file cũ nếu có (resume được)
        label_dataset = load_existing(file_name)

        # add claim đã có
        for item in label_dataset:
            global_seen_claims.add(item["claim"])

        count = len(label_dataset)

        with tqdm(total=per_label, initial=count) as pbar:
            while count < per_label:

                raw = generate_sample(label)

                if raw is None:
                    continue

                sample = parse_sample(raw)

                if not sample:
                    time.sleep(BASE_SLEEP)
                    continue

                if sample["label"] != label:
                    continue

                if sample["claim"] in global_seen_claims:
                    continue

                # add
                global_seen_claims.add(sample["claim"])
                label_dataset.append(sample)

                # 🔥 SAVE NGAY
                save_json(file_name, label_dataset)

                count += 1
                pbar.update(1)

                time.sleep(BASE_SLEEP + random.uniform(0.5, 2))

        final_dataset.extend(label_dataset)

    # ======================
    # Save final merged file
    # ======================
    random.shuffle(final_dataset)
    save_json(FINAL_OUTPUT, final_dataset)

    print(f"\n✅ Done. Total samples: {len(final_dataset)}")
    print(f"📁 Saved merged file: {FINAL_OUTPUT}")


# ======================
# Run
# ======================
if __name__ == "__main__":
    build_dataset()