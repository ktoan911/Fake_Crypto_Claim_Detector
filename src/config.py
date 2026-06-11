import datetime

LABEL_LIST = ["A", "B", "C"]
LABEL_TO_ID = {"A": 0, "B": 1, "C": 2}
ID_TO_LABEL = {0: "A", 1: "B", 2: "C"}

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

PROMPT_TEMPLATE = """You are an expert fact-checker verifying Vietnamese claims based on the provided evidence.

Evidence contains timestamps: [Thời gian của thông tin: ...]

Financial facts change over time.
Compare claim and evidence only if they refer to the same time period.

Answer ONLY one letter:
A = Supported
B = Contradicted (same time period)
C = Insufficient evidence, unclear time, or different time period

Claim: {claim}

Evidence: {evidence}

Conclusion:"""
