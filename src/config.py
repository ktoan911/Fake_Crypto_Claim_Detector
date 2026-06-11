import datetime

LABEL_LIST = ["A", "B", "C"]
LABEL_TO_ID = {"A": 0, "B": 1, "C": 2}
ID_TO_LABEL = {0: "A", 1: "B", 2: "C"}

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

PROMPT_TEMPLATE = """You are an expert fact-checker verifying Vietnamese claims based on the provided evidence.

Each evidence item includes a timestamp [Thời gian của thông tin: ...]. Financial data changes over time — only classify as B if the evidence and claim refer to the same time period.

Answer with ONLY a single letter:
- A: Evidence supports the claim
- B: Evidence contradicts the claim (same time period)
- C: Not enough evidence, or evidence is from a different time period

Claim: {claim}

Evidence: {evidence}

Conclusion: """
