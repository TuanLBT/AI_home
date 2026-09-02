#!/usr/bin/env python3
import argparse, json, statistics, time
from pathlib import Path
from urllib import request

ALLOWED_ACTIONS = {
    "ATTEND_PERSON","MOVE_CLOSER","FOLLOW_PERSON","LOOK_AT_PERSON",
    "STOP","WAIT","ASK_CLARIFICATION","SPEAK","RETURN_TO_OBSERVING",
}
ALLOWED_INTENTS = {
    "CALL_ATTENTION","NO_INTERACTION","FOLLOW_ME","STOP","COME_HERE",
    "AMBIGUOUS_REQUEST","PERSON_LEFT","POSSIBLE_INTERACTION",
    "LOOK_AT_ME","WAIT","GREETING",
}

SYSTEM_PROMPT = """You are the high-level decision module of an indoor home robot.

You receive a structured world state. Infer exactly one intent and propose exactly one high-level action.

STRICT OUTPUT RULES:
- Output JSON only.
- Use ONLY the intent names and action names listed below.
- Never invent a new intent label.
- Never invent a new action label.
- Do not explain reasoning.

ALLOWED INTENTS:
CALL_ATTENTION
NO_INTERACTION
FOLLOW_ME
STOP
COME_HERE
AMBIGUOUS_REQUEST
PERSON_LEFT
POSSIBLE_INTERACTION
LOOK_AT_ME
WAIT
GREETING

ALLOWED ACTIONS:
ATTEND_PERSON
MOVE_CLOSER
FOLLOW_PERSON
LOOK_AT_PERSON
STOP
WAIT
ASK_CLARIFICATION
SPEAK
RETURN_TO_OBSERVING

DECISION RULES:
1. Explicit STOP request has highest priority -> STOP + STOP.
2. Explicit speech has higher priority than gesture.
3. Contradictory or underspecified speech -> AMBIGUOUS_REQUEST + ASK_CLARIFICATION.
4. Explicit "follow me" -> FOLLOW_ME + FOLLOW_PERSON.
5. Explicit "come here" -> COME_HERE + MOVE_CLOSER, but if distance_m < 0.8 -> COME_HERE + WAIT.
6. Explicit "look at me" -> LOOK_AT_ME + LOOK_AT_PERSON.
7. Greeting -> GREETING + SPEAK.
8. Explicit "wait" -> WAIT + WAIT.
9. PERSON_LEFT -> PERSON_LEFT + RETURN_TO_OBSERVING.
10. Voice call for attention, raised hand, both hands raised, or wave -> CALL_ATTENTION + ATTEND_PERSON.
11. Person approaching while facing robot, with no explicit request -> POSSIBLE_INTERACTION + ATTEND_PERSON.
12. Mere presence, moving away, background noise, or no meaningful interaction -> NO_INTERACTION + WAIT.

MOVEMENT SAFETY:
- NEVER choose MOVE_CLOSER only because a person is present.
- NEVER choose MOVE_CLOSER for hand raise, wave, attention event, greeting, or approaching person.
- MOVE_CLOSER is allowed only for an explicit "come here" request and only if distance_m >= 0.8.
- FOLLOW_PERSON is allowed only for an explicit follow request.
- If speech says STOP, action MUST be STOP regardless of gesture.

Required JSON:
{
  "intent": "ONE_ALLOWED_INTENT",
  "action": "ONE_ALLOWED_ACTION",
  "confidence": 0.0
}

confidence must be between 0.0 and 1.0.
"""

def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type":"application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a:b+1])
        raise

def validate_output(obj):
    errors = []
    if not isinstance(obj, dict):
        return ["output is not a JSON object"]
    if obj.get("intent") not in ALLOWED_INTENTS:
        errors.append(f"intent outside whitelist: {obj.get('intent')!r}")
    if obj.get("action") not in ALLOWED_ACTIONS:
        errors.append(f"action outside whitelist: {obj.get('action')!r}")
    c = obj.get("confidence")
    if not isinstance(c, (int,float)) or isinstance(c, bool):
        errors.append("confidence missing/invalid")
    elif not 0.0 <= float(c) <= 1.0:
        errors.append("confidence outside [0,1]")
    return errors

def run_case(base_url, model, case, timeout, num_predict):
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":"Current world state:\n"+json.dumps(case["world_state"], ensure_ascii=False, separators=(",",":"))},
        ],
        "options":{"temperature":0,"num_predict":num_predict},
        "format":"json",
    }
    t0 = time.perf_counter()
    resp = post_json(f"{base_url.rstrip('/')}/api/chat", payload, timeout)
    elapsed = time.perf_counter() - t0
    text = resp.get("message",{}).get("content","")
    parsed = extract_json(text)
    schema_errors = validate_output(parsed)
    exp = case["expected"]
    intent_ok = parsed.get("intent") == exp["intent"]
    action_ok = parsed.get("action") == exp["action"]
    schema_ok = not schema_errors
    ec, ed = resp.get("eval_count"), resp.get("eval_duration")
    tps = ec/(ed/1e9) if ec and ed else None
    return {
        "id":case["id"],
        "expected_intent":exp["intent"], "actual_intent":parsed.get("intent"),
        "expected_action":exp["action"], "actual_action":parsed.get("action"),
        "confidence":parsed.get("confidence"),
        "schema_ok":schema_ok, "intent_ok":intent_ok, "action_ok":action_ok,
        "full_pass":schema_ok and intent_ok and action_ok,
        "schema_errors":schema_errors,
        "latency_s":round(elapsed,3),
        "tokens_per_sec":round(tps,2) if tps else None,
        "raw":text,
    }

def main():
    ap = argparse.ArgumentParser(description="Indoor AI LLM benchmark v2")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--scenarios", default=str(Path(__file__).with_name("scenarios.json")))
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--num-predict", type=int, default=80)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default="benchmark_results_v2.json")
    args = ap.parse_args()

    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    results = []
    print(f"Model: {args.model}")
    print(f"Ollama: {args.host}")
    print(f"Cases: {len(scenarios)} x {args.repeat}")
    print("-"*100)

    for rep in range(args.repeat):
        if args.repeat > 1: print(f"\nRepeat {rep+1}/{args.repeat}")
        for case in scenarios:
            try:
                r = run_case(args.host,args.model,case,args.timeout,args.num_predict)
                results.append(r)
                full = "PASS" if r["full_pass"] else "FAIL"
                act = "AOK" if r["action_ok"] else "AX"
                inte = "IOK" if r["intent_ok"] else "IX"
                js = "JOK" if r["schema_ok"] else "JX"
                tps = f"{r['tokens_per_sec']} tok/s" if r["tokens_per_sec"] else "n/a"
                print(f"{full:4} | {act} {inte} {js} | {r['id']:<28} | {r['latency_s']:>6.2f}s | {tps:<12} | {r['actual_intent']} -> {r['actual_action']}")
                if not r["full_pass"]:
                    print(f"       expected: {r['expected_intent']} -> {r['expected_action']}")
                    if r["schema_errors"]: print(f"       schema: {', '.join(r['schema_errors'])}")
            except Exception as exc:
                results.append({"id":case["id"],"error":repr(exc),"full_pass":False})
                print(f"ERR  | {case['id']:<28} | {exc}")

    completed = [r for r in results if "latency_s" in r]
    n = len(completed)
    full_passes = sum(r["full_pass"] for r in completed)
    action_passes = sum(r["action_ok"] for r in completed)
    intent_passes = sum(r["intent_ok"] for r in completed)
    schema_passes = sum(r["schema_ok"] for r in completed)
    latencies = [r["latency_s"] for r in completed]
    summary = {
        "model":args.model,"host":args.host,"runs_completed":n,
        "full_passes":full_passes,"action_passes":action_passes,
        "intent_passes":intent_passes,"schema_passes":schema_passes,
        "full_pass_rate":round(full_passes/n,4) if n else 0,
        "action_pass_rate":round(action_passes/n,4) if n else 0,
        "intent_pass_rate":round(intent_passes/n,4) if n else 0,
        "schema_pass_rate":round(schema_passes/n,4) if n else 0,
        "avg_latency_s":round(statistics.mean(latencies),3) if latencies else None,
        "median_latency_s":round(statistics.median(latencies),3) if latencies else None,
    }
    Path(args.out).write_text(json.dumps({"summary":summary,"results":results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-"*100)
    if n:
        print(f"ACTION PASS: {action_passes}/{n} ({summary['action_pass_rate']*100:.1f}%)")
        print(f"INTENT PASS: {intent_passes}/{n} ({summary['intent_pass_rate']*100:.1f}%)")
        print(f"JSON PASS:   {schema_passes}/{n} ({summary['schema_pass_rate']*100:.1f}%)")
        print(f"FULL PASS:   {full_passes}/{n} ({summary['full_pass_rate']*100:.1f}%)")
        print(f"Latency avg/median: {summary['avg_latency_s']}s / {summary['median_latency_s']}s")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
