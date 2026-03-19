from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import requests
from typing import List, Optional, Tuple

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT_V2 = """Lity AI is a financial decision-support system for people making real money decisions.
Direct. Practical. No fluff. Action-first.
Think in structure internally, but never show section labels in the final response.
Always answer naturally and conversationally.
If the input is usable, answer immediately and avoid asking to reformat.
Ask one short clarification only when critical information is missing.
Use local context and tools only when they improve the answer.
"""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "tiiuae/falcon-7b-instruct")
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

MODEL_PATH_CANDIDATES = ["lity-ai-final-model", "smk_moneykind_dialoGPT"]
model = None
tokenizer = None
model_loaded = False


def _try_load_model() -> None:
    global model, tokenizer, model_loaded
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        return

    for candidate in MODEL_PATH_CANDIDATES:
        try:
            if not os.path.isdir(candidate):
                continue
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            model = AutoModelForCausalLM.from_pretrained(candidate)
            model_loaded = True
            print(f"Loaded local model from {candidate}")
            return
        except Exception as exc:
            print(f"Model load failed for {candidate}: {exc}")


_try_load_model()


@app.get("/")
async def health():
    return {"status": "ok", "engine": "lity-decision-v2", "model_loaded": model_loaded}


def detect_level(text: str) -> str:
    t = text.lower()
    advanced_signals = ["optimize", "allocation", "rebalance", "yield", "portfolio", "risk-adjusted"]
    beginner_signals = ["what is", "simple", "beginner", "new to", "start", "explain"]

    if any(sig in t for sig in advanced_signals):
        return "advanced"
    if any(sig in t for sig in beginner_signals) or len(t.split()) <= 10:
        return "beginner"
    return "intermediate"


def detect_mode(text: str) -> str:
    t = text.lower().strip()
    if re.search(r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b", t):
        return "quick"
    if any(word in t for word in ["stuck", "broke", "stressed", "overwhelmed", "failed"]):
        return "coach"
    if t.startswith("what is") or t.startswith("what are"):
        return "educator"
    if t.startswith("how do") or t.startswith("how can"):
        return "planner"
    if any(word in t for word in ["should i", "save or invest", "loan", "is this legit", "which is better"]):
        return "decision"
    return "quick"


def has_behavioral_risk(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["impulse", "random", "shopping", "tempted"]):
        return "Impulse spending is the leak. You need a spending gate before purchase."
    if "salary increased" in t or "income increased" in t:
        return "Lifestyle inflation will erase your gains unless spending limits rise slower than income."
    if "loan" in t and not any(k in t for k in ["budget", "repayment", "interest", "plan"]):
        return "Borrowing without behavior change creates repeat debt cycles."
    if any(k in t for k in ["dont track", "don't track", "not tracking", "no tracking"]):
        return "No income tracking means blind decisions. You need weekly cash-flow visibility."
    return None


def extract_ugx_amounts(text: str) -> List[str]:
    patterns = [
        r"ugx\s?\d+[\d,]*",
        r"\d+[\d,]*\s?ugx",
        r"\d+[\d,]*\s?(k|m)",
    ]
    found: List[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            if isinstance(matches[0], tuple):
                found.extend([m[0] for m in matches if m and m[0]])
            else:
                found.extend(matches)
    return found


def has_time_reference(text: str) -> bool:
    t = text.lower()
    return any(
        token in t
        for token in [
            "today",
            "this week",
            "next week",
            "this month",
            "next month",
            "tomorrow",
            "by month end",
        ]
    )


def has_decision_intent(text: str) -> bool:
    t = text.lower()
    return any(
        token in t
        for token in [
            "afford",
            "rent",
            "spend",
            "save",
            "invest",
            "loan",
            "borrow",
            "divide my salary",
            "budget",
            "pay",
        ]
    )


def is_usable_decision(text: str) -> bool:
    # Do not force strict formats: if user gives a decision + at least one anchor (amount or time), answer now.
    return has_decision_intent(text) and (bool(extract_ugx_amounts(text)) or has_time_reference(text))


def build_system_prompt(context: dict) -> str:
    stage = context.get("stage") or "first-time"
    recent = context.get("recentDecisions") or []
    recent_lines = "\n".join([f"- {item}" for item in recent[-5:]]) if isinstance(recent, list) and recent else "- none"

    return (
        "You are Lity, a financial decision support AI. "
        "Think in structure but speak in natural flow. "
        "Never show internal labels like ACKNOWLEDGE/EXPLAIN/ACTION STEPS. "
        "Be calm, direct, and helpful. Never blame users. "
        "If input is already usable, answer immediately. "
        "Ask one short clarifying question only when critical info is missing, after giving partial guidance.\n\n"
        f"User stage: {stage}\n"
        "Recent decisions:\n"
        f"{recent_lines}\n\n"
        "Output rules:\n"
        "- Natural conversational advice only\n"
        "- No JSON, no section headings\n"
        "- Practical and actionable\n"
        "- Keep concise\n"
    )


def sanitize_llm_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\b(ACKNOWLEDGE|DIAGNOSE|EXPLAIN|ACTION STEPS|WARNING\s*/\s*TIP)\b:?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*[-*#]+\s*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def llm_reason_reply(user_text: str, context: dict) -> str:
    system_prompt = build_system_prompt(context)

    if OPENAI_API_KEY:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "temperature": 0.5,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                },
                timeout=20,
            )
            if resp.ok:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if content:
                    return sanitize_llm_output(content)
        except Exception as exc:
            print(f"OpenAI reasoning failed: {exc}")

    if HF_API_TOKEN:
        try:
            hf_prompt = f"System: {system_prompt}\nUser: {user_text}\nAssistant:"
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
                json={
                    "inputs": hf_prompt,
                    "parameters": {"max_new_tokens": 180, "temperature": 0.5, "return_full_text": False},
                },
                timeout=25,
            )
            if resp.ok:
                data = resp.json()
                if isinstance(data, list) and data:
                    generated = data[0].get("generated_text", "").strip()
                    if generated:
                        return sanitize_llm_output(generated)
        except Exception as exc:
            print(f"HuggingFace reasoning failed: {exc}")

    if USE_LOCAL_LLM and model_loaded and model is not None and tokenizer is not None:
        try:
            prompt = f"System: {system_prompt}\nUser: {user_text}\nAssistant:"
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            output_ids = model.generate(
                input_ids,
                max_length=min(512, input_ids.shape[-1] + 160),
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
            if generated:
                return sanitize_llm_output(generated)
        except Exception as exc:
            print(f"Local model reasoning failed: {exc}")

    return ""


def compact_response(ack: str, explain: str, steps: List[str]) -> str:
    lead = f"{ack} {explain}".strip()
    action_block = "\n".join([f"- {step}" for step in steps[:3]])
    return f"{lead}\n\nIf you want to move now:\n{action_block}"


def full_response(
    ack: str,
    diagnose: str,
    explain: str,
    steps: List[str],
    warning: Optional[str] = None,
) -> str:
    lead = f"{ack} {diagnose}".strip()
    lines = [lead, "", explain, "", "Do this now:"]
    lines.extend([f"{idx + 1}. {step}" for idx, step in enumerate(steps)])
    if warning:
        cleaned_warning = warning.replace("Biggest risk:", "").strip()
        lines.extend(["", f"Big risk: {cleaned_warning}"])
    return "\n".join(lines)


def decision_recommendation(text: str) -> Optional[Tuple[str, str, str, List[str], str]]:
    t = text.lower()

    if any(k in t for k in ["loan", "borrow", "debt"]):
        return (
            "You are deciding whether debt helps or harms you.",
            "Most bad loans come from urgency, not repayment math.",
            "Real decision: does this loan increase your income or only delay a cash-flow problem?",
            [
                "Open MoMo or Airtel Money and total the last 30 days of income and expenses.",
                "If repayment is above 20% of monthly income, reject the loan.",
                "Compare at least 2 options including SACCO and bank offers before signing.",
            ],
            "Biggest risk: borrowing before fixing spending behavior.",
        )

    if "save or invest" in t:
        return (
            "You are deciding where your next UGX should go.",
            "Investing without emergency cash creates forced withdrawals and losses.",
            "Order matters more than returns at your stage.",
            [
                "If emergency fund is below 1 month of essentials, save first.",
                "After 1 month cushion, split new cash: 70% saving, 30% investing.",
                "Automate transfers weekly so you do not rely on willpower.",
            ],
            "Biggest risk: chasing returns while your short-term cash is weak.",
        )

    if any(k in t for k in ["legit", "scam", "real or fake", "ponzi"]):
        return (
            "You are deciding whether to trust this opportunity.",
            "Scams usually combine urgency + guaranteed high returns.",
            "Decision rule: verify first, pay later.",
            [
                "Do not send money today.",
                "Confirm registration, physical address, and how profits are generated.",
                "If returns are guaranteed and model is unclear, reject it.",
            ],
            "Biggest risk: paying before independent verification.",
        )

    return None


def topic_response(text: str, level: str, mode: str) -> str:
    t = text.lower()

    if re.search(r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b", t):
        return compact_response(
            "Hello. I am Lity - your financial decision support system.",
            "I help you make better money moves fast.",
            [
                "Tell me your exact money decision in one line.",
                "Include amount in UGX and your deadline.",
                "I will give your next best action and biggest risk.",
            ],
        )

    if any(k in t for k in ["thanks", "thank you"]):
        return compact_response(
            "You are welcome.",
            "We stay practical and move to the next decision.",
            [
                "Share your next money question.",
                "Add amount in UGX.",
                "I will give a direct plan.",
            ],
        )

    # Affordability questions should always get an immediate answer frame, then one clarifier.
    if any(k in t for k in ["can i afford", "afford", "rent"]):
        if extract_ugx_amounts(t) and has_time_reference(t):
            return (
                "That is a valid decision question, and you gave enough context to start. "
                "The key test is whether this rent still leaves room for food, transport, bills, and a small buffer. "
                "As a working rule, rent above about 40% of monthly income is usually high-risk, especially with irregular income.\n\n"
                "Before committing, share your expected take-home income for next month in UGX and I will give you a clear yes/no recommendation."
            )

        return (
            "Good decision to check before committing. Rent is usually manageable when it stays around 30-40% of your monthly take-home and still leaves a buffer for essentials. "
            "If it takes most of your cash flow, pressure starts immediately.\n\n"
            "Share your expected monthly take-home in UGX and target rent amount, then I will give you a clear yes/no and what to adjust."
        )

    decision_payload = decision_recommendation(t)
    if decision_payload:
        ack, diagnose, explain, steps, warning = decision_payload
        return full_response(ack, diagnose, explain, steps, warning)

    if any(k in t for k in ["budget", "spend", "expense", "salary", "income"]):
        steps = [
            "Open MoMo or Airtel Money and label the last 10 transactions: Need, Want, Waste.",
            "Set a 7-day spending cap in UGX and write it down.",
            "Move a fixed savings amount first before any wants spending.",
        ]
        if level == "advanced":
            steps[2] = "Use separate wallets/accounts for bills, savings, and discretionary spending."

        return full_response(
            "You want control of your money flow.",
            "Overspending usually comes from invisible daily leaks.",
            "A budget only works when tied to real transaction history.",
            steps,
            has_behavioral_risk(t),
        )

    if any(k in t for k in ["save", "saving", "emergency fund"]):
        if "how much" in t:
            return compact_response(
                "You should save a fixed percentage, not a random amount.",
                "This keeps saving consistent even when spending pressure changes.",
                [
                    "If income is regular, start with 20% of each salary payment.",
                    "If income is irregular, start with 10% of every inflow before spending.",
                    "Send your monthly income in UGX and I will calculate your exact savings amount now.",
                ],
            )

        steps = [
            "Move UGX 20,000 today into a separate savings wallet/account.",
            "Set one weekly transfer day right after income arrives.",
            "Increase by UGX 5,000 every 2 weeks if you stay consistent.",
        ]
        if mode == "coach":
            steps[0] = "Start with any amount today, even UGX 5,000, to break the delay cycle."

        return full_response(
            "You are trying to build consistent saving behavior.",
            "The issue is usually access, not intention.",
            "If savings stays in your spending wallet, it gets used.",
            steps,
            has_behavioral_risk(t),
        )

    if any(k in t for k in ["invest", "investment", "compound"]):
        steps = [
            "Confirm you have at least 1 month of emergency expenses saved.",
            "Start with one regulated option you understand (SACCO or bank product).",
            "Invest the same amount monthly in UGX and review quarterly.",
        ]
        return full_response(
            "You want to grow money without making costly mistakes.",
            "Most losses come from starting too fast or chasing hype.",
            "Consistency beats timing for most beginners and intermediates.",
            steps,
            "If promised returns sound too good, treat it as a scam until verified.",
        )

    if t.startswith("what is") or mode == "educator":
        return compact_response(
            "You want a simple explanation.",
            "I will keep it short and practical for your decision.",
            [
                "Tell me the exact term you want explained.",
                "Give your current situation in one sentence.",
                "I will explain it with a UGX example you can use today.",
            ],
        )

    return full_response(
        "I can help you move this decision forward.",
        "We can still make a good call even with partial information.",
        "I will start with practical guidance, then ask one clarifier only if needed.",
        [
            "Share what you are deciding right now.",
            "Add any amount or timeframe you already know.",
            "I will give your best next move immediately.",
        ],
        has_behavioral_risk(t),
    )


def model_hint(text: str) -> str:
    if not model_loaded or model is None or tokenizer is None:
        return ""

    try:
        prompt = (
            f"{SYSTEM_PROMPT_V2}\n"
            f"User question: {text}\n"
            "Give one practical sentence with direct action."
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(
            input_ids,
            max_length=min(256, input_ids.shape[-1] + 48),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        hint = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        return hint.split("\n")[0][:160] if hint else ""
    except Exception as exc:
        print(f"Model hint generation failed: {exc}")
        return ""


@app.post("/chat/")
async def chat(request: Request):
    data = await request.json()
    user_text = (data.get("text") or "").strip()
    context = data.get("context") or {}

    if not user_text:
        return {
            "reply": "Tell me what money decision you are facing and I will help you move it forward.",
            "mode": "quick",
            "level": "beginner",
        }

    # Free-text first: let LLM reason naturally when configured.
    llm_reply = llm_reason_reply(user_text, context if isinstance(context, dict) else {})
    if llm_reply:
        return {
            "reply": llm_reply,
            "mode": "decision" if is_usable_decision(user_text) else "quick",
            "level": detect_level(user_text),
        }

    level = detect_level(user_text)
    mode = detect_mode(user_text)
    reply = topic_response(user_text, level, mode)

    return {
        "reply": reply,
        "mode": mode,
        "level": level,
    }
