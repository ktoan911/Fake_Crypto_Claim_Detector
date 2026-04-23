import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_KEY"),
    base_url="https://senator-gigolo-stark.ngrok-free.dev/v1",
)

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

def 


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
