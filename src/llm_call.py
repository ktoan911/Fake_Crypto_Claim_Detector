import json
import os
import random
import re
import time
from datetime import datetime, timedelta
from typing import List

import pytz
from dotenv import load_dotenv
from together import Together
from together import error as together_error

load_dotenv()

# Together SDK mặc định lấy timeout từ httpx — với model lớn + prompt dài
# (cluster có thể chứa hàng trăm claim) request dễ vượt ngưỡng và bắn
# together.APITimeoutError. Đặt timeout tường minh, dài hơn default.
_TOGETHER_TIMEOUT = float(os.getenv("TOGETHER_TIMEOUT", "120"))
_TOGETHER_MAX_RETRIES = int(os.getenv("TOGETHER_MAX_RETRIES", "3"))

client = Together(
    timeout=_TOGETHER_TIMEOUT,
)  # auth defaults to os.environ.get("TOGETHER_API_KEY")


_RETRYABLE_ERRORS = (
    together_error.APITimeoutError,
    together_error.APIConnectionError,
    together_error.RateLimitError,
)


class _StreamedMessage:
    def __init__(self, content: str):
        self.content = content


class _StreamedChoice:
    def __init__(self, content: str):
        self.message = _StreamedMessage(content)


class _StreamedResponse:
    """Wraps a collected stream so callers can use .choices[0].message.content."""
    def __init__(self, content: str):
        self.choices = [_StreamedChoice(content)]


def _collect_stream(stream) -> _StreamedResponse:
    content = "".join(
        (chunk.choices[0].delta.content or "") for chunk in stream
    )
    return _StreamedResponse(content)


def _chat_completion_with_retry(**kwargs):
    """Wrap client.chat.completions.create với retry + exponential backoff cho
    các lỗi mạng/timeout/rate-limit. Lỗi non-retryable (Auth, BadRequest...)
    được raise ngay để không che lỗi cấu hình.
    Nếu model yêu cầu streaming, tự động retry với stream=True và gộp chunks."""
    last_err = None
    for attempt in range(_TOGETHER_MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except together_error.BadRequestError as e:
            # Some models only support streaming — retry once transparently
            if "streaming_required" in str(e) and not kwargs.get("stream"):
                stream = client.chat.completions.create(**kwargs, stream=True)
                return _collect_stream(stream)
            raise
        except _RETRYABLE_ERRORS as e:
            last_err = e
            if attempt == _TOGETHER_MAX_RETRIES - 1:
                break
            # Exponential backoff + jitter: 2s, 4s, 8s ...
            sleep_s = (2 ** attempt) + random.uniform(0, 1)
            print(
                f"[WARN] Together API {type(e).__name__} "
                f"(attempt {attempt + 1}/{_TOGETHER_MAX_RETRIES}), "
                f"retry sau {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)
    raise last_err

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

SYSTEM_PROMPT_RUMOR_GENERATION = "Bạn là hệ thống tạo tin đồn tài chính tiếng Việt."


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


# Output tối đa 5 từ → ~32 token là dư. Cap để tránh model generate lan man,
# vốn là một nguyên nhân chính khiến request timeout.
_CLUSTER_SUMMARY_MAX_TOKENS = int(os.getenv("CLUSTER_SUMMARY_MAX_TOKENS", "64"))
# Giới hạn số claim đưa vào prompt — cluster lớn có thể có hàng trăm claim,
# prompt dài làm inference chậm và dễ timeout. Lấy mẫu là đủ để LLM nắm topic.
_CLUSTER_SUMMARY_MAX_CLAIMS = int(os.getenv("CLUSTER_SUMMARY_MAX_CLAIMS", "30"))


def build_prompt_summary_cluster(claims, centroid):
    if len(claims) > _CLUSTER_SUMMARY_MAX_CLAIMS:
        claims = claims[:_CLUSTER_SUMMARY_MAX_CLAIMS]
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


def build_prompt_generate_rumors_from_news(news_items, target_count=50):
    news_text = []
    for idx, item in enumerate(news_items, start=1):
        source_ref = item.get("source_ref") or idx
        title = str(item.get("title") or "")[:220]
        content = str(item.get("content") or item.get("description") or "")[:700]
        news_text.append(f"[{source_ref}] {title}\n{content}")

    return f"""
Dựa vào các tin tức sau, hãy bịa ra đúng {target_count} tin đồn/claim tài chính tiếng Việt.

Yêu cầu:
- Trả về ONLY JSON array, không giải thích.
- Mỗi item có format: {{"claim": "...", "source_ref": 1}}
- Mỗi claim chỉ 1-2 câu ngắn.
- Có thể đúng, sai, thiếu chắc chắn hoặc chỉ cùng chủ đề.
- source_ref là số trong [] của tin nguồn liên quan nhất.

Tin nguồn:
{chr(10).join(news_text)}
""".strip()


def generate_rumor_claims_from_news(news_items, target_count=50):
    prompt = build_prompt_generate_rumors_from_news(news_items, target_count)
    response = _chat_completion_with_retry(
        model="Qwen/Qwen3.6-Plus",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_RUMOR_GENERATION},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.8,
    )
    return (response.choices[0].message.content or "").strip()


def generate_cluster_content_with_llm(
    cluster_claims: List[str], representative_claim: str
) -> str:

    cluster_all = build_prompt_summary_cluster(cluster_claims, representative_claim)

    try:
        response = _chat_completion_with_retry(
            model="Qwen/Qwen3.6-Plus",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TOPIC},
                {"role": "user", "content": cluster_all},
            ],
            max_tokens=_CLUSTER_SUMMARY_MAX_TOKENS,
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()
    except _RETRYABLE_ERRORS as e:
        # Sau khi đã retry vẫn timeout/connect lỗi → fallback dùng representative
        # claim làm topic thay vì để cả pipeline cluster crash.
        print(
            f"[ERROR] generate_cluster_content_with_llm fallback do {type(e).__name__}: {e}"
        )
        return representative_claim


def safe_parse_list(text: str) -> List[str]:
    """Trích danh sách claim từ output của LLM theo nhiều format khác nhau."""
    if not text:
        return []

    def _clean(items):
        out = []
        for x in items:
            s = str(x).strip().strip('"').strip("'").strip("`")
            if s:
                out.append(s)
        return out

    # 1) Raw JSON array
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _clean(parsed)
    except Exception:
        pass

    # 2) JSON array nằm đâu đó trong text (kèm prefix/giải thích)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return _clean(parsed)
        except Exception:
            pass

    # 3) Bullet list:  - foo  /  * foo  /  • foo
    bullets = re.findall(r"^\s*[-*•]\s*(.+?)\s*$", text, re.MULTILINE)
    if bullets:
        return _clean(bullets)

    # 4) Numbered list: 1. foo  /  2) foo
    numbered = re.findall(r"^\s*\d+[\.\)]\s*(.+?)\s*$", text, re.MULTILINE)
    if numbered:
        return _clean(numbered)

    return []


def _should_skip_split(text: str) -> bool:
    """Tránh gọi LLM cho input ngắn / không có dấu hiệu nhiều fact."""
    if len(text) < 30:
        return True
    # Bắt đầu bằng số thứ tự ("1. ", "2) ") + ngắn → 1 mục trong list, không phải multi-claim
    if re.match(r"^\s*\d+[\.\)]\s", text) and len(text) < 120:
        return True
    # Không có dấu chấm câu kết và ngắn → có khả năng là cụm từ đơn lẻ
    if len(text) < 80 and not re.search(r"[.!?。\n]", text):
        return True
    return False


# =========================
# 3. Main function (no raise)
# =========================
def split_claim(claim: str) -> List[str]:
    text = (claim or "").strip()
    if not text:
        return []

    # Heuristic: input ngắn / không phải multi-claim → khỏi gọi LLM, dùng nguyên claim
    if _should_skip_split(text):
        return [text]

    try:
        prompt = build_prompt_extraction(text)

        for attempt in range(3):
            response = _chat_completion_with_retry(
                model="Qwen/Qwen3.6-Plus",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_EXTRACTION},
                    {"role": "user", "content": prompt},
                ],
            )
            output_text = (response.choices[0].message.content or "").strip()

            result = safe_parse_list(output_text)
            if result:
                return result

            print(
                f"[WARN] Retry {attempt + 1}: format lỗi — raw response: "
                f"{output_text[:200]!r}"
            )

        print("[ERROR] LLM split failed after 3 attempts, fallback to original claim")
        return [text]

    except Exception as e:
        print(f"[ERROR] split_claim exception: {e}, fallback to original claim")
        return [text]
