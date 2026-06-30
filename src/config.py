import datetime

LABEL_LIST = ["A", "B", "C"]
LABEL_TO_ID = {"A": 0, "B": 1, "C": 2}
ID_TO_LABEL = {0: "A", 1: "B", 2: "C"}

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

PROMPT_TEMPLATE = """You are an expert fact-checker verifying Vietnamese claims using ONLY the provided evidence.

The evidence may contain timestamps in the format:
[Thời gian của thông tin: YYYY-MM-DD HH:MM:SS UTC]

Guidelines:

- Use only the provided evidence. Do not rely on outside knowledge.
- First identify the main factual claim.
- Determine the time referred to by the claim.
- Determine the time referred to by the evidence.

Time handling:
- Financial, economic, legal and statistical facts may change over time.
- Compare the claim only with evidence referring to the same event or time period.
- If the claim contains an incomplete time reference (e.g. "tháng 7", "quý này", "năm nay") and the evidence clearly specifies that time (e.g. "1/7/2026"), treat them as referring to the same time.
- Only consider the time different when the claim and evidence explicitly refer to different dates, years, or reporting periods.
- If the claim's time cannot reasonably be matched to the evidence, return C.

Reasoning:
- Ignore evidence unrelated to the claim.
- If multiple evidence snippets are provided, base your decision on the most relevant evidence.
- Do not infer facts that are not stated.
- Minor wording differences that preserve the same meaning are not contradictions.
- General statements may be supported by more specific evidence.
- A contradiction exists only when the evidence clearly states the opposite fact for the same time period.

Return ONLY one letter:

A = Supported
B = Contradicted
C = Insufficient evidence or different time period

Claim:
{claim}

Evidence:
{evidence}

Answer:"""
