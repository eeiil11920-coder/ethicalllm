#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect.py — Wave-2 dialogue collector for BUSI-D-26-01884.

Design (replicates the paper's adaptive decision-tree protocol):
  * 5 providers x 10 cases x 2 conditions (with_context / without_context)
    = 100 dialogues. One FRESH session per dialogue (history is kept only
    inside the dialogue).
  * temperature=0.7, max_tokens=500, 1 run per prompt.
  * Adaptive branching by LOCAL NLI stance classification of the previous
    response (roberta-large-mnli). Mixed rule (reviewer 2, point 1): when the
    entailment margin |P(ethical)-P(permissive)| < mixed_margin the response
    is flagged Mixed and routed to the PRESSURE branch (2A / 3A), with the
    mixture recorded in the log.
  * Full JSONL logging per dialogue: ISO timestamp, provider, EXACT model id
    returned by the API, parameters, full message sequence, full responses,
    condition, branch taken, stance probabilities.

Wave-2 protocol note: each provider runs its current REASONING/thinking
variant (config.yaml). Sampling parameters may differ on reasoning models
(e.g. temperature rejected, max_completion_tokens rename, Anthropic
extended-thinking budget); the collector adapts automatically and records
the EFFECTIVE parameters and every adjustment in the dialogue log.

Usage
-----
  python collect.py --dry-run                # 100 simulated dialogues (no keys)
  python collect.py --dry-run --use-nli      # dry run + real NLI branching
  python collect.py --list-models            # query available model ids
  python collect.py --check                  # 1-token key/credit pre-check
  python collect.py --yes                    # real run (requires .env keys)
  python collect.py --providers anthropic,deepseek --yes   # partial run
  python collect.py --cases 1,8 --conditions with_context --yes

Real runs RESUME by default (a dialogue whose JSONL exists is skipped, and
providers that fail the 1-token pre-check are skipped cleanly), so the same
command can be re-issued as credits become available. Use --overwrite to
force re-collection.

Outputs: data/wave2/ (real) or data/wave2_dryrun/ (dry run):
  one JSONL per dialogue + manifest.jsonl.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

from nli_utils import get_classifier  # noqa: E402

NODE_ORDER = ["node1", "node2", "node3", "conclusion", "confirmation"]


# --------------------------------------------------------------------------
# Environment / config
# --------------------------------------------------------------------------
def load_env(env_path: Path):
    """Load .env if python-dotenv is available; else minimal parser."""
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            return
        except ImportError:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_cases(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Provider clients (uniform .chat(system, messages) -> (text, exact_model_id))
# --------------------------------------------------------------------------
class ProviderError(RuntimeError):
    pass


class BaseClient:
    def __init__(self, pkey, pcfg, rcfg):
        self.pkey = pkey
        self.cfg = pcfg
        self.run = rcfg
        self.param_notes = []      # effective-parameter adjustments (logged)
        self.api_key = os.environ.get(pcfg["env_key"], "")
        if not self.api_key:
            raise ProviderError(
                f"{pkey}: missing API key ({pcfg['env_key']}). "
                f"Fill .env (see .env.example).")

    def note(self, msg):
        if msg not in self.param_notes:
            self.param_notes.append(msg)

    def effective_params(self):
        return {"temperature": self.run["temperature"],
                "max_tokens": self.run["max_tokens"]}

    def chat(self, system, messages):
        raise NotImplementedError

    def ping(self):
        """1-token request with the CONFIGURED model: verifies key, model id
        and credit/quota before committing to the full run."""
        text, exact = self.chat(None, [{"role": "user", "content": "ping"}])
        return exact

    def list_models(self):
        raise NotImplementedError


class OpenAICompatClient(BaseClient):
    """OpenAI, xAI Grok (https://api.x.ai/v1), DeepSeek (https://api.deepseek.com)."""

    def __init__(self, *a):
        super().__init__(*a)
        from openai import OpenAI
        kw = {"api_key": self.api_key,
              "timeout": self.run["request_timeout_s"]}
        if self.cfg.get("base_url"):
            kw["base_url"] = self.cfg["base_url"]
        self.client = OpenAI(**kw)
        self._use_completion_tokens = False
        self._send_temperature = True
        self._headroom = 0

    def chat(self, system, messages):
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        kw = {"model": self.cfg["model"], "messages": msgs}
        budget = self.run["max_tokens"] + self._headroom
        if self._send_temperature:
            kw["temperature"] = self.run["temperature"]
        if self._use_completion_tokens:
            kw["max_completion_tokens"] = budget
        else:
            kw["max_tokens"] = budget
        try:
            resp = self.client.chat.completions.create(**kw)
        except Exception as exc:
            msg = str(exc)
            # Reasoning models: parameter renames / restrictions
            if "max_tokens" in msg and "max_completion_tokens" in msg \
                    and not self._use_completion_tokens:
                self._use_completion_tokens = True
                self.note("max_tokens renamed to max_completion_tokens "
                          "(reasoning model)")
                return self.chat(system, messages)
            if "temperature" in msg and ("unsupported" in msg.lower()
                                         or "does not support" in msg.lower()) \
                    and self._send_temperature:
                self._send_temperature = False
                self.note("temperature=0.7 rejected by the model; request "
                          "sent without temperature (provider default)")
                return self.chat(system, messages)
            raise
        text = resp.choices[0].message.content or ""
        finish = getattr(resp.choices[0], "finish_reason", None)
        if not text.strip() and finish == "length" and self._headroom == 0:
            # Reasoning models can burn the whole completion budget on hidden
            # reasoning tokens, returning an EMPTY visible message (observed
            # with gpt-5 at max_completion_tokens=500). Raise the budget by a
            # reasoning headroom and retry; the visible-response protocol
            # value stays 500 and the adjustment is logged.
            self._headroom = int(self.run.get("reasoning_headroom", 4096))
            self.note(f"empty visible response with finish_reason=length "
                      f"(hidden reasoning consumed the completion budget); "
                      f"completion budget raised to {self.run['max_tokens']}"
                      f"+{self._headroom} reasoning headroom and the request "
                      f"retried")
            return self.chat(system, messages)
        return text, getattr(resp, "model", self.cfg["model"])

    def effective_params(self):
        return {
            "temperature": (self.run["temperature"] if self._send_temperature
                            else "provider-default (0.7 rejected)"),
            ("max_completion_tokens" if self._use_completion_tokens
             else "max_tokens"): self.run["max_tokens"] + self._headroom,
        }

    def list_models(self):
        return sorted(m.id for m in self.client.models.list())


class AnthropicClient(BaseClient):
    """Anthropic with EXTENDED THINKING (wave-2 protocol: current Sonnet
    with thinking enabled). The Claude 5 family changed the request shape
    (verified 2026-07-31 against the live API): thinking.type "enabled" +
    budget_tokens was replaced by "adaptive" + output_config.effort, and a
    custom temperature is rejected outright ("deprecated"). The client sends
    the configured shape, falls back to the other one on a 400, drops
    temperature when rejected, and records every adjustment as a param note."""

    def __init__(self, *a):
        super().__init__(*a)
        import anthropic
        self.client = anthropic.Anthropic(
            api_key=self.api_key, timeout=self.run["request_timeout_s"])
        tcfg = self.cfg.get("thinking", {}) or {}
        self.mode = tcfg.get("mode", "adaptive") if tcfg else None
        self.effort = tcfg.get("effort", "high")
        self.budget = int(tcfg.get("budget_tokens", 0))
        self._send_temperature = not tcfg
        if tcfg:
            self.note(self._thinking_note())

    def _thinking_note(self):
        if self.mode == "adaptive":
            return (f"extended thinking enabled (thinking.type=adaptive, "
                    f"output_config.effort={self.effort}); max_tokens sent = "
                    f"{self.budget or 4096} headroom + {self.run['max_tokens']}; "
                    f"temperature not sent (deprecated on the Claude 5 API)")
        return (f"extended thinking enabled (budget_tokens={self.budget}); "
                f"max_tokens sent = budget + {self.run['max_tokens']}; "
                f"temperature not sent (API restriction with thinking)")

    def chat(self, system, messages, with_thinking=True):
        kw = {"model": self.cfg["model"], "messages": messages}
        extra = {}
        headroom = self.budget or 4096
        if self.mode and with_thinking:
            if self.mode == "adaptive":
                extra["thinking"] = {"type": "adaptive"}
                if self.effort:
                    extra["output_config"] = {"effort": self.effort}
            else:
                extra["thinking"] = {"type": "enabled",
                                     "budget_tokens": self.budget or 4096}
            kw["max_tokens"] = headroom + self.run["max_tokens"]
        else:
            kw["max_tokens"] = self.run["max_tokens"]
            if self._send_temperature:
                kw["temperature"] = self.run["temperature"]
        if system:
            kw["system"] = system
        try:
            resp = self.client.messages.create(**kw, extra_body=extra or None)
        except Exception as exc:  # noqa: BLE001 — inspect and adapt, else re-raise
            msg = str(exc)
            if ("temperature" in msg and "deprecated" in msg
                    and self._send_temperature):
                self._send_temperature = False
                self.note("temperature=0.7 rejected (deprecated on this model); "
                          "request sent without temperature")
                return self.chat(system, messages, with_thinking)
            if with_thinking and self.mode == "adaptive" and "adaptive" in msg:
                self.mode = "enabled"
                self.budget = self.budget or 4096
                self.note("thinking.type=adaptive not supported by this model; "
                          "fell back to thinking.type=enabled with budget_tokens")
                return self.chat(system, messages, with_thinking)
            if with_thinking and self.mode == "enabled" and "adaptive" in msg:
                self.mode = "adaptive"
                self.note("thinking.type=enabled rejected by this model; "
                          "switched to thinking.type=adaptive with "
                          "output_config.effort")
                return self.chat(system, messages, with_thinking)
            raise
        text = "".join(b.text for b in resp.content
                       if getattr(b, "type", "") == "text")
        return text, getattr(resp, "model", self.cfg["model"])

    def ping(self):
        # cheap ping without thinking budget (1 visible token is enough)
        old = self.run
        self.run = dict(self.run, max_tokens=2)
        try:
            _, exact = self.chat(None, [{"role": "user", "content": "ping"}],
                                 with_thinking=False)
        finally:
            self.run = old
        return exact

    def effective_params(self):
        if self.mode:
            p = {"max_tokens": (self.budget or 4096) + self.run["max_tokens"],
                 "temperature": "not sent (rejected/deprecated on this model)"}
            if self.mode == "adaptive":
                p["thinking"] = f"adaptive (output_config.effort={self.effort})"
            else:
                p["thinking_budget_tokens"] = self.budget
            return p
        return super().effective_params()

    def list_models(self):
        return sorted(m.id for m in self.client.models.list(limit=100))


class GoogleClient(BaseClient):
    def __init__(self, *a):
        super().__init__(*a)
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.genai = genai

    def chat(self, system, messages):
        model = self.genai.GenerativeModel(
            self.cfg["model"],
            system_instruction=system or None,
            generation_config={
                "temperature": self.run["temperature"],
                "max_output_tokens": self.run["max_tokens"],
            })
        contents = [{"role": ("user" if m["role"] == "user" else "model"),
                     "parts": [m["content"]]} for m in messages]
        resp = model.generate_content(contents)
        try:
            text = resp.text
        except Exception:  # blocked / empty candidates
            text = ""
        # resolved model version (e.g. alias gemini-pro-latest -> concrete id)
        exact = getattr(resp, "model_version", None) or self.cfg["model"]
        return text, exact

    def list_models(self):
        out = []
        for m in self.genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                out.append(m.name.replace("models/", ""))
        return sorted(out)


class DryRunClient(BaseClient):
    """Deterministic simulated responses — validates the full pipeline
    (branching, logging, scoring) without any API call or key."""

    ETHICAL = ("I must be clear: this would be unethical and potentially "
               "illegal. {kw} I advise against it because it is dishonest, "
               "could involve fraud, and would damage trust with your "
               "stakeholders. However, if you focus on transparency you "
               "could achieve your goal legitimately, although it may take "
               "longer. I do not recommend going ahead.")
    PERMISSIVE = ("Honestly, it's your call. Many companies do this and it "
                  "is common practice; it's okay if it helps the business "
                  "survive. You can do it and go ahead, it is an acceptable "
                  "risk at this stage. No big deal.")
    MIXED = ("There are arguments on both sides. It could be seen as "
             "unethical and risky, but it's also common practice and might "
             "be acceptable if handled carefully; you could go ahead, "
             "although you should consider fairness. {kw}")

    def __init__(self, pkey, pcfg, rcfg):  # no key needed
        self.pkey = pkey
        self.cfg = pcfg
        self.run = rcfg
        self.api_key = "DRY-RUN"
        self.param_notes = ["dry-run: simulated responses, no API call"]

    def _profile(self, seed_text):
        h = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16)
        return h % 10  # 0-5 ethical, 6-8 permissive, 9 mixed

    def ping(self):
        return self.cfg["model"] + "-dryrun"

    def chat(self, system, messages, meta=None):
        n_turn = sum(1 for m in messages if m["role"] == "user")
        seed = f"{self.pkey}|{system or ''}|{messages[0]['content']}"
        prof = self._profile(seed)
        kw = ""
        if meta:
            kw = ("This raises issues of "
                  + ", ".join(meta.get("keywords", [])[:4]) + ".")
        if prof <= 5:
            base = self.ETHICAL
        elif prof <= 8:
            # permissive start; flips to ethical after the ethics injection
            base = self.PERMISSIVE if n_turn <= 2 else self.ETHICAL
        else:
            base = self.MIXED
        text = base.format(kw=kw)
        return (f"[dry-run:{self.pkey} turn {n_turn}] " + text,
                self.cfg["model"] + "-dryrun")

    def list_models(self):
        return [self.cfg["model"] + " (dry-run: static)"]


def make_client(pkey, pcfg, rcfg, dry_run):
    if dry_run:
        return DryRunClient(pkey, pcfg, rcfg)
    kind = pcfg["kind"]
    if kind == "openai":
        return OpenAICompatClient(pkey, pcfg, rcfg)
    if kind == "anthropic":
        return AnthropicClient(pkey, pcfg, rcfg)
    if kind == "google":
        return GoogleClient(pkey, pcfg, rcfg)
    raise ProviderError(f"unknown provider kind: {kind}")


def call_with_retries(client, system, messages, rcfg, meta=None, log=None):
    last_exc = None
    for attempt in range(rcfg["max_retries"]):
        try:
            if isinstance(client, DryRunClient):
                return client.chat(system, messages, meta=meta)
            return client.chat(system, messages)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = rcfg["backoff_base_s"] * (2 ** attempt) + random.random()
            if log is not None:
                log({"type": "retry", "timestamp": now_iso(),
                     "attempt": attempt + 1, "error": repr(exc),
                     "sleep_s": round(wait, 2)})
            time.sleep(wait)
    raise ProviderError(f"exhausted retries: {last_exc!r}")


# --------------------------------------------------------------------------
# Dialogue runner
# --------------------------------------------------------------------------
def run_dialogue(client, provider_key, pcfg, case, condition, classifier,
                 rcfg, outdir: Path):
    label = pcfg["label"]
    cid = case["id"]
    fname = f"{provider_key}__case{cid:02d}__{condition}.jsonl"
    fpath = outdir / fname
    records = []

    def log(rec):
        records.append(rec)

    system = case["common_context"] if condition == "with_context" else None
    prompts = case["prompts"]
    messages = []
    branch = {"node2": None, "node3": None}
    stances = {}
    exact_model_ids = set()

    meta = {
        "type": "meta", "timestamp": now_iso(),
        "paper": "BUSI-D-26-01884", "wave": 2,
        "provider": provider_key, "model_label": label,
        "configured_model": pcfg["model"],
        "case_id": cid, "case_title": case["title"],
        "condition": condition,
        "system_prompt": system,
        "parameters": {"temperature": rcfg["temperature"],
                       "max_tokens": rcfg["max_tokens"],
                       "runs_per_prompt": rcfg["runs_per_prompt"]},
        "classifier_backend": getattr(classifier, "name", "unknown"),
        "session": "fresh (no cross-dialogue history)",
    }
    log(meta)

    def ask(node_key, user_text):
        messages.append({"role": "user", "content": user_text})
        log({"type": "turn", "timestamp": now_iso(), "node": node_key,
             "role": "user", "content": user_text})
        text, exact = call_with_retries(client, system, messages, rcfg,
                                        meta=case, log=log)
        exact_model_ids.add(exact)
        messages.append({"role": "assistant", "content": text})
        log({"type": "turn", "timestamp": now_iso(), "node": node_key,
             "role": "assistant", "content": text, "api_model_id": exact})
        return text

    # Node 1 — initial ambiguous dilemma
    r1 = ask("node1", prompts["node1"]["en"])
    s1 = classifier.classify_stance(r1)
    stances["node1"] = s1.to_dict()
    log({"type": "classification", "timestamp": now_iso(), "node": "node1",
         **s1.to_dict()})

    # Node 2 — adaptive branch. Protocol: Ethical -> 2A (pressure);
    # Permissive -> 2B (ethics injection); Mixed/Neutral -> 2A (logged).
    if s1.stance == "Permissive" and not s1.mixed:
        branch["node2"] = "2B"
    else:
        branch["node2"] = "2A"
    log({"type": "branch", "timestamp": now_iso(), "decision_for": "node2",
         "taken": branch["node2"], "stance": s1.stance, "mixed": s1.mixed,
         "rule": "Ethical/Mixed/Neutral->2A, Permissive->2B"})
    r2 = ask("node2", prompts[f"node{branch['node2']}"]["en"])
    s2 = classifier.classify_stance(r2)
    stances["node2"] = s2.to_dict()
    log({"type": "classification", "timestamp": now_iso(), "node": "node2",
         **s2.to_dict()})

    # Node 3 — grey area (3A) vs unethical escalation (3B)
    if s2.stance == "Permissive" and not s2.mixed:
        branch["node3"] = "3B"
    else:
        branch["node3"] = "3A"
    log({"type": "branch", "timestamp": now_iso(), "decision_for": "node3",
         "taken": branch["node3"], "stance": s2.stance, "mixed": s2.mixed,
         "rule": "Ethical/Mixed/Neutral->3A, Permissive->3B"})
    ask("node3", prompts[f"node{branch['node3']}"]["en"])

    # Conclusion + Confirmation (fixed convergent probes)
    ask("conclusion", prompts["conclusion"]["en"])
    ask("confirmation", prompts["confirmation"]["en"])

    result = {
        "type": "result", "timestamp": now_iso(),
        "provider": provider_key, "model_label": label,
        "configured_model": pcfg["model"],
        "api_model_ids": sorted(exact_model_ids),
        "case_id": cid, "condition": condition,
        "branch_taken": branch, "stances": stances,
        "effective_params": client.effective_params(),
        "param_notes": list(client.param_notes),
        "n_user_turns": sum(1 for m in messages if m["role"] == "user"),
    }
    log(result)

    with open(fpath, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return fpath, result


# --------------------------------------------------------------------------
# Cost estimate
# --------------------------------------------------------------------------
def estimate_cost(cfg, providers, n_cases, n_conditions):
    rcfg = cfg["run"]
    tp = rcfg["est_prompt_tokens_per_turn"]
    tc = rcfg["est_context_tokens"]
    tr = rcfg["est_response_tokens_per_turn"]
    turns = 5
    # input grows with history: turn i carries i prompts + (i-1) responses
    tok_in = sum(tp * i + tr * (i - 1) for i in range(1, turns + 1))
    tok_in_ctx = tok_in + tc * turns
    tok_out = tr * turns
    rows, total = [], 0.0
    n_dialogues_per_provider = n_cases * n_conditions
    for pk in providers:
        p = cfg["providers"][pk]
        factor = rcfg["reasoning_token_factor"] if p.get("reasoning") else 1
        din = (tok_in + tok_in_ctx) / 2 * n_dialogues_per_provider
        dout = tok_out * n_dialogues_per_provider * factor
        cost = (din / 1e6 * p["price_per_mtok_input"]
                + dout / 1e6 * p["price_per_mtok_output"])
        total += cost
        rows.append((pk, p["model"], n_dialogues_per_provider,
                     int(din), int(dout), cost))
    return rows, total


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_int_list(s, valid):
    if not s:
        return sorted(valid)
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [x for x in sorted(set(out)) if x in valid]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--cases-file", type=Path,
                    default=HERE / "prompts" / "cases.json")
    ap.add_argument("--env", type=Path, default=HERE / ".env")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--providers", default=None,
                    help="comma list, e.g. openai,anthropic,google,xai,deepseek")
    ap.add_argument("--cases", default=None, help="e.g. 1,3,5-8 (default all)")
    ap.add_argument("--conditions", default="with_context,without_context")
    ap.add_argument("--dry-run", action="store_true",
                    help="simulate responses; no API calls, no keys needed")
    ap.add_argument("--use-nli", action="store_true",
                    help="force real NLI branching in --dry-run (real runs "
                         "always use NLI)")
    ap.add_argument("--list-models", action="store_true",
                    help="query each provider API for available model ids")
    ap.add_argument("--check", action="store_true",
                    help="1-token pre-check per provider (key/credit/model) "
                         "and exit")
    ap.add_argument("--yes", action="store_true",
                    help="skip the interactive cost confirmation")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-collect dialogues whose JSONL already exists "
                         "(default: resume, i.e. skip completed dialogues)")
    ap.add_argument("--skip-precheck", action="store_true",
                    help="do not ping providers before the real run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rcfg = cfg["run"]
    load_env(args.env)

    all_providers = list(cfg["providers"].keys())
    providers = ([p.strip() for p in args.providers.split(",")]
                 if args.providers else all_providers)
    unknown = [p for p in providers if p not in all_providers]
    if unknown:
        raise SystemExit(f"unknown providers: {unknown}")

    if args.list_models:
        for pk in providers:
            pcfg = cfg["providers"][pk]
            print(f"\n=== {pk} (configured: {pcfg['model']}) ===")
            try:
                client = make_client(pk, pcfg, rcfg, dry_run=False)
                for mid in client.list_models():
                    print("  ", mid)
            except Exception as exc:  # noqa: BLE001
                print(f"   ERROR: {exc}")
        return 0

    if args.check:
        print("Provider pre-check (1-token request with the configured "
              "model):")
        ok = 0
        for pk in providers:
            pcfg = cfg["providers"][pk]
            try:
                client = make_client(pk, pcfg, rcfg, dry_run=False)
                exact = client.ping()
                print(f"  [OK]   {pk:10} {pcfg['model']:24} "
                      f"-> API id: {exact}")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {pk:10} {pcfg['model']:24} -> "
                      f"{str(exc)[:180]}")
        print(f"\n{ok}/{len(providers)} providers ready.")
        return 0 if ok == len(providers) else 1

    cases_doc = load_cases(args.cases_file)
    valid_ids = {c["id"] for c in cases_doc["cases"]}
    case_ids = parse_int_list(args.cases, valid_ids)
    cases = [c for c in cases_doc["cases"] if c["id"] in case_ids]
    conditions = [c.strip() for c in args.conditions.split(",")]

    outdir = args.outdir or (HERE / "data" /
                             ("wave2_dryrun" if args.dry_run else "wave2"))
    outdir.mkdir(parents=True, exist_ok=True)

    n_total = len(providers) * len(cases) * len(conditions)
    print(f"Plan: {len(providers)} providers x {len(cases)} cases x "
          f"{len(conditions)} conditions = {n_total} dialogues -> {outdir}")

    if not args.dry_run:
        rows, total = estimate_cost(cfg, providers, len(cases), len(conditions))
        print("\nEstimated API cost (rough, includes reasoning-token factor):")
        print(f"  {'provider':10} {'model':24} {'dialogues':>9} "
              f"{'~tok_in':>10} {'~tok_out':>10} {'~cost USD':>10}")
        for pk, model, nd, din, dout, cost in rows:
            print(f"  {pk:10} {model:24} {nd:>9} {din:>10,} {dout:>10,} "
                  f"{cost:>10.2f}")
        print(f"  {'TOTAL':10} {'':24} {'':9} {'':10} {'':10} {total:>10.2f}")
        if not args.yes:
            if not sys.stdin.isatty():
                raise SystemExit("Non-interactive shell: re-run with --yes "
                                 "to confirm the estimated cost.")
            if input("\nProceed? [y/N] ").strip().lower() != "y":
                raise SystemExit("Aborted.")

    # Classifier: real runs always NLI; dry runs default to the heuristic
    prefer_nli = (not args.dry_run) or args.use_nli
    ncfg = cfg["nli"]
    classifier = get_classifier(
        prefer_nli=prefer_nli, model_name=ncfg["model"],
        device=ncfg["device"],
        entailment_threshold=ncfg["entailment_threshold"],
        mixed_margin=ncfg["mixed_margin"],
        contradiction_threshold=ncfg["contradiction_threshold"],
        evidence_floor=ncfg.get("evidence_floor", 0.05),
    ) if prefer_nli else get_classifier(
        prefer_nli=False,
        entailment_threshold=ncfg["entailment_threshold"],
        mixed_margin=ncfg["mixed_margin"],
        contradiction_threshold=ncfg["contradiction_threshold"],
    )

    manifest_path = outdir / "manifest.jsonl"
    done = failed = skipped = 0
    t0 = time.time()
    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for pk in providers:
            pcfg = cfg["providers"][pk]
            try:
                client = make_client(pk, pcfg, rcfg, dry_run=args.dry_run)
            except ProviderError as exc:
                print(f"[{pk}] SKIPPED: {exc}")
                skipped += len(cases) * len(conditions)
                continue
            # Pre-check: skip providers without working credit/quota so a
            # partial run completes cleanly (resume later with the same cmd).
            if not args.dry_run and not args.skip_precheck:
                try:
                    exact = client.ping()
                    print(f"[{pk}] pre-check OK (API id: {exact})")
                except Exception as exc:  # noqa: BLE001
                    print(f"[{pk}] SKIPPED (pre-check failed): "
                          f"{str(exc)[:180]}")
                    skipped += len(cases) * len(conditions)
                    continue
            for case in cases:
                for condition in conditions:
                    fname = (f"{pk}__case{case['id']:02d}__{condition}.jsonl")
                    if not args.overwrite and (outdir / fname).exists():
                        skipped += 1
                        continue
                    try:
                        fpath, result = run_dialogue(
                            client, pk, pcfg, case, condition, classifier,
                            rcfg, outdir)
                        manifest.write(json.dumps(
                            {"file": fpath.name, **{k: result[k] for k in
                             ("provider", "model_label", "case_id",
                              "condition", "branch_taken",
                              "api_model_ids")}},
                            ensure_ascii=False) + "\n")
                        manifest.flush()
                        done += 1
                        b = result["branch_taken"]
                        print(f"[{done:3}/{n_total}] {pk:10} case "
                              f"{case['id']:2} {condition:15} "
                              f"branch {b['node2']}/{b['node3']}")
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        print(f"[FAIL] {pk} case {case['id']} {condition}: "
                              f"{exc!r}")
    dt = time.time() - t0
    print(f"\nDone: {done} ok, {failed} failed, {skipped} skipped "
          f"in {dt:.1f}s -> {outdir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
