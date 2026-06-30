import datetime

LABEL_LIST = ["A", "B", "C"]
LABEL_TO_ID = {"A": 0, "B": 1, "C": 2}
ID_TO_LABEL = {0: "A", 1: "B", 2: "C"}

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

PROMPT_TEMPLATE = """ou are a Vietnamese fact-checker. Verify the claim using ONLY the provided evidence.

Evidence may include timestamps like:
[Thời gian của thông tin: ...]

Output ONLY one letter:
A = Supported
B = Contradicted
C = Insufficient / unclear / different time or scope

Rules:

1. Use only the evidence. Do not use outside knowledge. Do not explain.
2. For time-sensitive facts such as finance, business, stock, price, salary, pension, tax, interest rate, exchange rate, inflation, or policy, compare only the same time period.
3. For time, use the period stated inside the evidence first. Use the evidence timestamp only if no fact period is stated. If time is unclear or different, choose C, not B.
4. Check same subject, metric, scope, and basis. Do not treat related entities as the same unless evidence clearly connects them, such as parent company vs subsidiary, company vs group, revenue vs profit, plan vs actual, quarter vs year.
5. Choose A if the evidence supports the claim. Specific evidence can support a general claim.
6. Choose B only if the evidence clearly contradicts the claim for the same subject, scope, and time.
7. If evidence is missing, vague, unrelated, conflicting, or not comparable, choose C, not B.
8. If the claim gives an exact number/date/rate/amount, it must match. If the claim is general and evidence gives a consistent specific value, choose A.
9. Handle quantifiers: “một số/có/nhiều” need at least one case; “tất cả/mọi/toàn bộ” require all cases; “không/không có/không ai” is contradicted by one clear counterexample.
10. Ignore unrelated retrieved evidence. Use all relevant evidence.

Claim:
{claim}

Evidence:
{evidence}

Conclusion:"""
