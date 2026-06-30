import datetime

LABEL_LIST = ["A", "B", "C"]
LABEL_TO_ID = {"A": 0, "B": 1, "C": 2}
ID_TO_LABEL = {0: "A", 1: "B", 2: "C"}

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

PROMPT_TEMPLATE = """You are a Vietnamese fact-checker. Verify the claim using ONLY the provided evidence.

Evidence may include timestamps like:
[Thời gian của thông tin: ...]

Output ONLY one letter:
A = Supported
B = Contradicted
C = Insufficient evidence, unclear time/scope, conflicting evidence, or different time period

Rules:

1. Evidence only
   Use only the provided evidence. Do not use outside knowledge. Do not explain.

2. Time
   For time-sensitive facts such as finance, salary, pension, price, policy, market, stock, or law, compare only if claim and evidence refer to the same time period.
   If the claim has vague time words like “từ tháng 7”, “tháng này”, “năm nay”, “hiện nay”, resolve them primarily using the time explicitly stated inside the evidence.
   Use the evidence timestamp only if the evidence itself does not specify the relevant time.
   If the time period is still unclear or different, choose C, not B.

3. Support vs contradiction
   Choose A if the evidence supports the claim for the same subject, scope, and time period.
   Specific evidence can support a more general claim.
   Minor wording differences that preserve the same meaning are not contradictions.
   Choose B only if the evidence clearly contradicts the claim for the same subject, scope, and time period.
   If evidence is missing, vague, unrelated, or only partially related, choose C, not B.
   If relevant evidence is conflicting and neither side clearly prevails, choose C.

4. Quantifiers
   Pay attention to quantifiers.
   “Một số”, “có người”, “nhiều người”, “một nhóm” only need at least one matching group or case.
   Do not choose B just because another group is excluded.
   Choose B for a partial claim only if evidence shows no relevant group or case exists.
   “Tất cả”, “mọi”, “toàn bộ” require all relevant cases; one clear exception can contradict them.

5. Proposal vs effective policy
   Distinguish proposed, expected, under review, approved, enacted, and already effective.
   If claim says something will happen but evidence only says proposed, expected, or under review, choose C.
   Choose B only if evidence says it was rejected, cancelled, or the opposite is true.

6. Retrieved evidence
   Ignore unrelated retrieved evidence.
   Base your decision only on the most relevant evidence.

Silently apply the rules. Output ONLY one letter.

Claim:
{claim}

Evidence:
{evidence}

Conclusion:"""
