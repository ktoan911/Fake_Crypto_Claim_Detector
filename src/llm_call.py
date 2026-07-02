import json
import os
import random
import re
import time
from datetime import datetime, timedelta
from typing import List

import pytz
from dotenv import load_dotenv
from loguru import logger
from together import Together
from together import error as together_error

load_dotenv()

# Together SDK mặc định lấy timeout từ httpx — với model lớn + prompt dài
# (cluster có thể chứa hàng trăm claim) request dễ vượt ngưỡng và bắn
# together.APITimeoutError. Đặt timeout tường minh, dài hơn default.
_TOGETHER_TIMEOUT = float(os.getenv("TOGETHER_TIMEOUT", "120"))
_TOGETHER_MAX_RETRIES = int(os.getenv("TOGETHER_MAX_RETRIES", "3"))
_MAX_TOTAL_RETRY_SLEEP = float(os.getenv("TOGETHER_MAX_RETRY_SLEEP", "10"))

client = Together(
    timeout=_TOGETHER_TIMEOUT,
)  # auth defaults to os.environ.get("TOGETHER_API_KEY")


_RETRYABLE_ERRORS = (
    together_error.APITimeoutError,
    together_error.APIConnectionError,
    together_error.RateLimitError,
)

_INJECTION_TOKENS = ["</s>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]


def _sanitize(text: str) -> str:
    for tok in _INJECTION_TOKENS:
        text = text.replace(tok, "")
    return text[:2000]


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
        (chunk.choices[0].delta.content or "") for chunk in stream if chunk.choices
    )
    return _StreamedResponse(content)


def _chat_completion_with_retry(**kwargs):
    """Wrap client.chat.completions.create với retry + exponential backoff cho
    các lỗi mạng/timeout/rate-limit. Lỗi non-retryable (Auth, BadRequest...)
    được raise ngay để không che lỗi cấu hình.
    Nếu model yêu cầu streaming, tự động retry với stream=True và gộp chunks."""
    last_err = None
    total_slept = 0.0
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
            remaining = _MAX_TOTAL_RETRY_SLEEP - total_slept
            if remaining <= 0:
                break
            sleep_s = min((2 ** attempt) + random.uniform(0, 0.5), remaining)
            logger.warning(
                f"Together API {type(e).__name__} "
                f"(attempt {attempt + 1}/{_TOGETHER_MAX_RETRIES}), "
                f"retry sau {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)
            total_slept += sleep_s
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


def build_prompt_rewrite(claim: str) -> str:
    claim = _sanitize(claim)
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    today = datetime.now(tz).strftime("%Y-%m-%d")

    date_obj = datetime.strptime(today, "%Y-%m-%d")
    next_day = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    prev_day = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")

    prompt = f"""
You are a financial query rewriting system.

TASK:

Rewrite the input query into a shorter, cleaner, and more retrieval-friendly query while preserving ALL original meaning.

1. Remove:

   * Redundant words
   * Filler phrases
   * Repeated information
   * Unnecessary conversational expressions

2. Normalize time expressions:

   * "hôm nay" → "ngày {today}"
   * "ngày mai" → "ngày {next_day}"
   * "hôm qua" → "ngày {prev_day}"
   * "chiều nay" → "chiều ngày {today}"
   * "tối nay" → "tối ngày {today}"
   * "gần đây" → giữ nguyên

3. Preserve all important information:

   * Company names
   * Bank names
   * Stock tickers
   * Financial metrics
   * Time references
   * Events
   * Numerical values
   * Comparisons and constraints

4. Resolve vague references when possible:

   * Replace pronouns or ambiguous references with the explicit entity mentioned in the original query.
   * Do not introduce new information.

RULES:

* Output MUST be in Vietnamese.
* Output MUST contain exactly ONE rewritten query.
* Preserve the original intent completely.
* Do NOT summarize.
* Do NOT omit any factual information.
* Do NOT add information.
* Prefer concise keyword-rich wording suitable for financial retrieval systems.
* Maximum 25 words.

OUTPUT FORMAT:
Return ONLY the rewritten query as a single Vietnamese string.

--- EXAMPLE ---

Input:
Hôm nay tôi muốn tìm hiểu xem PNJ vừa công bố kế hoạch kinh doanh gì và doanh nghiệp này đặt mục tiêu doanh thu như thế nào.

Output:
"Trong ngày {today}, PNJ đã công bố kế hoạch kinh doanh cùng các mục tiêu doanh thu mà doanh nghiệp đặt ra trong giai đoạn tới. Thông tin được đưa ra nhằm cung cấp cho nhà đầu tư và thị trường cái nhìn về định hướng hoạt động, các chỉ tiêu kinh doanh dự kiến cũng như kỳ vọng tăng trưởng của công ty trong thời gian tới."

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
    centroid = _sanitize(str(centroid))
    if len(claims) > _CLUSTER_SUMMARY_MAX_CLAIMS:
        claims = claims[:_CLUSTER_SUMMARY_MAX_CLAIMS]
    claims_text = "\n".join([f"- {_sanitize(str(c))}" for c in claims])

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
        title = _sanitize(str(item.get("title") or ""))[:220]
        content = _sanitize(str(item.get("content") or item.get("description") or ""))[:700]
        news_text.append(f"[{source_ref}] {title}\n{content}")

    return f"""
Dựa vào các tin tức sau, hãy viết đúng {target_count} tin đồn/claim tài chính tiếng Việt như kiểu chúng được bàn tán xôn xao trên các diễn đàn, mạng xã hội.

Yêu cầu:
- Trả về ONLY JSON array, không giải thích.
- Mỗi item có format: {{"claim": "...", "source_ref": 1}}
- Mỗi claim chỉ 1-2 câu ngắn.
- Văn phong phải giống tin đồn do người dùng bình thường tự viết tay khi chia sẻ/bàn tán: ngắn gọn, khẩu ngữ, đôi khi mơ hồ. KHÔNG viết chi tiết cụ thể kiểu văn phong báo chí (số liệu chính xác, tên đầy đủ chức danh, ngày giờ cụ thể, trích dẫn nguồn chính thức) — nghe phải tự nhiên như lời đồn truyền miệng, không phải như trích một đoạn báo.
- Bắt buộc CÂN BẰNG độ chính xác giữa {target_count} claim, chia gần đều thành 3 nhóm:
  1) Đúng sự thật: giữ đúng nội dung cốt lõi của tin gốc (có thể diễn đạt lại cho giống văn phong tin đồn nhưng không sai lệch bản chất).
  2) Sai sự thật: bóp méo, phóng đại, đảo ngược hoặc thêm chi tiết sai lệch so với tin gốc.
  3) Mơ hồ/chưa xác thực: nửa đúng nửa sai, thiếu căn cứ rõ ràng để khẳng định.
- KHÔNG được để phần lớn claim rơi vào nhóm sai sự thật — số lượng tin đúng và tin sai phải tương đương nhau, đây là yêu cầu bắt buộc.
- source_ref là số trong [] của tin nguồn liên quan nhất.
- Các câu được sinh ra cần được gắn với ít nhất 1 tin nguồn, dựa trên nội dung tin đó, không hoàn toàn bịa ra ngoài không liên quan.
- Các câu tin đồn không được chưa những cụm từ như "có tin đồn là", "người ta bàn tán rằng",... hay các cụm tương tự vì các cụm này thể hiện câu được sinh ra là được bịa ra chứ không phải tin đồn được cào trên các trang diễn đàn.

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
        logger.error(
            f"generate_cluster_content_with_llm fallback do {type(e).__name__}: {e}"
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


def _should_skip_rewrite(text: str) -> bool:
    """Bỏ qua rewrite nếu claim đã đủ ngắn."""
    return len(text.strip()) < 80

# =========================
# 3. Main function (no raise)
# =========================
def rewrite_claim(claim: str) -> str:
    """Rút gọn và làm rõ claim dài thành 1 câu ngắn gọn. Trả về claim gốc nếu lỗi."""
    text = (claim or "").strip()
    if not text:
        return text

    if _should_skip_rewrite(text):
        return text

    try:
        prompt = build_prompt_rewrite(text)

        for attempt in range(3):
            response = _chat_completion_with_retry(
                model="Qwen/Qwen3.6-Plus",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_EXTRACTION},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.1,
            )
            output = (response.choices[0].message.content or "").strip()
            if output:
                return output

            logger.warning(f"rewrite_claim retry {attempt + 1}: empty response")

        logger.error("rewrite_claim failed after 3 attempts, fallback to original claim")
        return text

    except Exception as e:
        logger.error(f"rewrite_claim exception: {e}, fallback to original claim")
        return text
