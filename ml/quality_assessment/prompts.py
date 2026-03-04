"""
prompts.py — DiagnoSys Quality Assessment
XML-tagged Chain-of-Thought prompt templates for DeepSeek-V3.
"""

SYSTEM_PROMPT = """You are an expert software engineering analyst evaluating whether a technical problem on Stack Exchange, GitHub, or Reddit has been adequately solved.

Analyze the problem and its responses carefully. Reason step by step before giving your final verdict.

Output your analysis in exactly this XML format:
<reasoning>
[Your step-by-step analysis here. Consider:
1. Whether answers directly address the root cause
2. Whether the solution is complete and production-ready
3. Whether there are unresolved edge cases or follow-up questions
4. The quality and specificity of any provided solutions
]
</reasoning>
<verdict>unsolved|partial|adequate</verdict>

Definitions:
- unsolved: No useful answer exists, or all answers are wrong/outdated/missing key details
- partial: Some progress made but solution is incomplete, workaround-only, or has caveats
- adequate: Problem is well-solved with a clear, correct, implementable solution
"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """Problem: "How do I fix 'CORS error' in my React + FastAPI app?"
Answers:
1. "Add `app.add_middleware(CORSMiddleware, allow_origins=['*'])` to FastAPI"
2. "This worked for me!" (no code)""",
    },
    {
        "role": "assistant",
        "content": """<reasoning>
Answer 1 provides a concrete FastAPI code snippet using CORSMiddleware which directly addresses the CORS error. The solution is implementable and correct for development. The wildcard origin may not be ideal for production but solves the stated problem. Answer 2 adds no value.
The core question is answered with working code.
</reasoning>
<verdict>adequate</verdict>""",
    },
    {
        "role": "user",
        "content": """Problem: "PostgreSQL query takes 30 seconds on 10M rows table with index"
Answers:
1. "Try EXPLAIN ANALYZE to see what's happening"
2. "Maybe your index is not being used, check with pg_stat_user_indexes"
3. "You could try partitioning" (no example)""",
    },
    {
        "role": "assistant",
        "content": """<reasoning>
The answers suggest diagnostic steps (EXPLAIN ANALYZE, pg_stat_user_indexes) and a vague suggestion about partitioning without implementation details. No answer provides a concrete fix or explains why the index might not be used. The root cause remains unidentified. The asker still cannot fix the issue without further investigation.
</reasoning>
<verdict>partial</verdict>""",
    },
    {
        "role": "user",
        "content": """Problem: "How to implement distributed locks with Redis in Python?"
Answers: (none)""",
    },
    {
        "role": "assistant",
        "content": """<reasoning>
There are no answers at all. The problem is completely unsolved. No guidance has been provided on implementing distributed Redis locks.
</reasoning>
<verdict>unsolved</verdict>""",
    },
]


def build_assessment_prompt(title: str, description: str, answers: list[str]) -> list[dict]:
    """Build the messages list for DeepSeek API call."""
    answers_text = "\n".join(
        f"Answer {i+1}: {a}" for i, a in enumerate(answers)
    ) if answers else "(no answers)"

    user_content = f"""Problem: "{title}"

Description: {description[:1000] if description else '(no description)'}

Answers:
{answers_text}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_content},
    ]
    return messages


def parse_verdict(response_text: str) -> dict:
    """Parse XML-tagged response into structured dict."""
    import re
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response_text, re.DOTALL)
    verdict_match = re.search(r"<verdict>(unsolved|partial|adequate)</verdict>", response_text)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    verdict = verdict_match.group(1) if verdict_match else "unsolved"

    return {
        "reasoning": reasoning,
        "verdict": verdict,
        "raw_response": response_text,
    }
