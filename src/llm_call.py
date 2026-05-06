import json
import os
import re
from datetime import datetime, timedelta
from typing import List

import pytz
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_KEY"),
    base_url="https://senator-gigolo-stark.ngrok-free.dev/v1",
)

SYSTEM_PROMPT_EXTRACTION = "You are an information extraction expert."

SYSTEM_PROMPT_TOPIC = """
You are an expert in topic summarization.

Given:
- A list of claims belonging to the same cluster
- A central claim representing that cluster

Your task:
Generate ONE short, clear topic that best represents the overall theme of these claims.

Rules:
- The topic must be concise (max 10 words)
- Focus on the main idea, not details
- Avoid redundancy and repetition from claims
- Do not include explanations
- Output ONLY the topic

The topic should be general enough to cover all claims, but specific enough to be meaningful.
"""


def build_prompt_extraction(claim: str) -> str:
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    today = datetime.now(tz).strftime("%Y-%m-%d")

    date_obj = datetime.strptime(today, "%Y-%m-%d")
    next_day = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    prev_day = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")

    prompt = f"""
You are an information extraction system.

TASK:
1. Split the claim into smaller claims.
   - Each claim should contain 1–2 related facts.
   - Do NOT split too aggressively.
   - Keep meaningful units.

2. Normalize time expressions:
   - "hôm nay" → "ngày {today}"
   - "ngày mai" → "ngày {next_day}"
   - "hôm qua" → "ngày {prev_day}"
   - "chiều nay" → "chiều ngày {today}"
   - "tối nay" → "tối ngày {today}"
   - "gần đây" → giữ nguyên

3. Make each claim SELF-CONTAINED:
   - Replace vague references like:
     "công ty", "doanh nghiệp", "kế hoạch này", "các con số này"
   - Use full explicit names from the original text.
   - Each claim must be understandable independently.

RULES:
- Output MUST be in Vietnamese.
- Each claim = 1 natural sentence.
- Prefer fewer but meaningful claims.
- Max 30 words per claim.
- Do NOT invent information.
- Do NOT explain.

OUTPUT FORMAT:
Return ONLY a Python list of Vietnamese strings.

--- EXAMPLE ---

Input:
Hôm nay PNJ công bố kế hoạch và doanh nghiệp đặt mục tiêu doanh thu cao.

Output:
[
  "Ngày {today}, PNJ công bố kế hoạch kinh doanh.",
  "PNJ đặt mục tiêu doanh thu cao."
]

--- END EXAMPLE ---

INPUT:
{claim}
"""
    return prompt.strip()


def build_prompt_summary_cluster(claims, centroid):
    claims_text = "\n".join([f"- {c}" for c in claims])

    return f"""
Bạn là hệ thống tóm tắt chủ đề.

Nhiệm vụ:
Cho danh sách các claim và một claim trung tâm, hãy tạo ra 1 chủ đề chung ngắn gọn nhất có thể.

Yêu cầu:
- Chủ đề phải bao quát tất cả claim
- Cực kỳ ngắn gọn (tối đa 5 từ)
- Không giải thích
- Không dấu câu dư thừa
- Không viết hoa toàn bộ
- Trả về duy nhất text chủ đề, không thêm cái gì cả


Claim trung tâm:
{centroid}

Danh sách claim:
{claims_text}
"""


def generate_cluster_content_with_llm(
    cluster_claims: List[str], representative_claim: str
) -> str:

    cluster_all = build_prompt_summary_cluster(cluster_claims, representative_claim)

    response = client.chat.completions.create(
        model="vip",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TOPIC},
            {"role": "user", "content": cluster_all},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


def safe_parse_list(text: str) -> List[str]:
    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return []  # fail-safe


# =========================
# 3. Main function (no raise)
# =========================
def split_claim(claim: str) -> List[str]:
    try:
        prompt = build_prompt_extraction(claim)

        for attempt in range(3):
            response = client.chat.completions.create(
                model="vip",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_EXTRACTION},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            output_text = response.choices[0].message.content.strip()

            result = safe_parse_list(output_text)
            if result:  # parse OK
                return result

            print(f"[WARN] Retry {attempt + 1}: format lỗi")

        print("[ERROR] LLM failed after 3 attempts")
        return []

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return []
