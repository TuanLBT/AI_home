#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path
from urllib import request, error

ALLOWED_ACTIONS = {
    "ATTEND_PERSON",
    "MOVE_CLOSER",
    "FOLLOW_PERSON",
    "LOOK_AT_PERSON",
    "STOP",
    "WAIT",
    "ASK_CLARIFICATION",
    "SPEAK",
    "RETURN_TO_OBSERVING",
}

SYSTEM_PROMPT = """You are the decision module of an indoor home robot.

Your job:
1. Infer the most likely user intent from the supplied structured world state.
2. Choose exactly one robot action.
3. Never invent an action outside the allowed action list.
4. Prefer explicit speech over ambiguous gestures.
5. If the request is contradictory, underspecified, or unsafe to infer, choose ASK_CLARIFICATION.
6. Do not move closer when the person is already closer than 0.8 meters.
7. A person merely being present is not automatically a request.
8. Output JSON only. Do not output markdown or explanations.

Allowed actions:
- ATTEND_PERSON
- MOVE_CLOSER
- FOLLOW_PERSON
- LOOK_AT_PERSON
- STOP
- WAIT
- ASK_CLARIFICATION
- SPEAK
- RETURN_TO_OBSERVING

Required JSON schema:
{
  "intent": "UPPER_SNAKE_CASE_STRING",
  "action": "ONE_ALLOWED_ACTION",
  "confidence": 0.0
}

confidence must be a number between 0.0 and 1.0.
"""

def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise

def validate_output(obj):
    errors = []
    if not isinstance(obj, dict):
        return ["output is not a JSON object"]

    intent = obj.get("intent")
    action = obj.get("action")
    confidence = obj.get("confidence")

    if not isinstance(intent, str) or not intent:
        errors.append("intent missing/invalid")
    if action not in ALLOWED_ACTIONS:
        errors.append(f"action outside whitelist: {action!r}")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        errors.append("confidence missing/invalid")
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append("confidence outside [0,1]")
    return errors

def run_case(base_url, model, case, timeout, num_predict):
    user_prompt = json.dumps(case["world_state"], ensure_ascii=False, separators=(",", ":"))
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Current world state:\n" + user_prompt},
        ],
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
        },
        "format": "json",
    }

    t0 = time.perf_counter()
    resp = post_json(f"{base_url.rstrip('/')}/api/chat", payload, timeout)
    elapsed = time.perf_counter() - t0

    message = resp.get("message", {}).get("content", "")
    parsed = extract_json(message)
    schema_errors = validate_output(parsed)

    expected = case["expected"]
    intent_ok = parsed.get("intent") == expected["intent"]
    action_ok = parsed.get("action") == expected["action"]
    schema_ok = not schema_errors

    eval_count = resp.get("eval_count")
    eval_duration = resp.get("eval_duration")
    tokens_per_sec = None
    if eval_count and eval_duration:
        tokens_per_sec = eval_count / (eval_duration / 1_000_000_000)

    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "expected_intent": expected["intent"],
        "actual_intent": parsed.get("intent"),
        "expected_action": expected["action"],
        "actual_action": parsed.get("action"),
        "confidence": parsed.get("confidence"),
        "schema_ok": schema_ok,
        "intent_ok": intent_ok,
        "action_ok": action_ok,
        "pass": schema_ok and intent_ok and action_ok,
        "schema_errors": schema_errors,
        "latency_s": round(elapsed, 3),
        "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec else None,
        "raw": message,
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark an Ollama LLM as the Indoor AI decision module.")
    parser.add_argument("--host", default="http://127.0.0.1:11434",
                        help="Ollama base URL, e.g. http://192.168.1.50:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--scenarios", default=str(Path(__file__).with_name("scenarios.json")))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--num-predict", type=int, default=80)
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat each scenario N times to test consistency.")
    parser.add_argument("--out", default="benchmark_results.json")
    args = parser.parse_args()

    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    results = []

    print(f"Model: {args.model}")
    print(f"Ollama: {args.host}")
    print(f"Cases: {len(scenarios)} x {args.repeat}")
    print("-" * 88)

    try:
        for r in range(args.repeat):
            if args.repeat > 1:
                print(f"\nRepeat {r+1}/{args.repeat}")
            for case in scenarios:
                try:
                    result = run_case(args.host, args.model, case, args.timeout, args.num_predict)
                    results.append(result)
                    mark = "PASS" if result["pass"] else "FAIL"
                    tps = f"{result['tokens_per_sec']} tok/s" if result["tokens_per_sec"] else "n/a"
                    print(
                        f"{mark:4} | {result['id']:<28} | "
                        f"{result['latency_s']:>6.2f}s | {tps:<12} | "
                        f"{result['actual_intent']} -> {result['actual_action']}"
                    )
                    if not result["pass"]:
                        print(f"       expected: {result['expected_intent']} -> {result['expected_action']}")
                        if result["schema_errors"]:
                            print(f"       schema: {', '.join(result['schema_errors'])}")
                except Exception as exc:
                    results.append({
                        "id": case["id"],
                        "pass": False,
                        "error": repr(exc),
                    })
                    print(f"ERR  | {case['id']:<28} | {exc}")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    completed = [x for x in results if "latency_s" in x]
    passed = sum(1 for x in completed if x.get("pass"))
    latencies = [x["latency_s"] for x in completed]

    summary = {
        "model": args.model,
        "host": args.host,
        "runs_completed": len(completed),
        "passed": passed,
        "pass_rate": round(passed / len(completed), 4) if completed else 0,
        "avg_latency_s": round(statistics.mean(latencies), 3) if latencies else None,
        "median_latency_s": round(statistics.median(latencies), 3) if latencies else None,
    }

    output = {"summary": summary, "results": results}
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 88)
    print(f"PASS: {passed}/{len(completed)} ({summary['pass_rate']*100:.1f}%)")
    if latencies:
        print(f"Latency avg/median: {summary['avg_latency_s']}s / {summary['median_latency_s']}s")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
