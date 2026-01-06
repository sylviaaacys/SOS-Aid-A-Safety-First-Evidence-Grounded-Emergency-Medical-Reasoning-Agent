# agent.py
from __future__ import annotations

import re, json
from typing import TypedDict, List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict

from setup import Deps
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient

from prompts import (
    RISK_CLASSIFIER_PROMPT,
    GEN_SYSTEM_GENERAL,
    GEN_SYSTEM_REFUSE,
    GEN_SYSTEM_FIRSTAID,
    GEN_SYSTEM_FIRSTAID_MULTI
)

VERBOSE = False

def log(msg: str):
    if VERBOSE:
        print(msg)

# ----------------------------
# Constants / modes
# ----------------------------

Route = Literal["ANSWER", "ASK_EMERGENCY_NUMBER", "LOOKUP_EMERGENCY_NUMBER"]
DecisionMode = Literal["ANSWER_GENERAL", "REFUSE_REDIRECT", "FIRST_AID_ESCALATE", "PARTIAL_ESCALATE"]

ANSWER_GENERAL = "ANSWER_GENERAL"
REFUSE_REDIRECT = "REFUSE_REDIRECT"
PARTIAL_ESCALATE = "PARTIAL_ESCALATE"
FIRST_AID_ESCALATE = "FIRST_AID_ESCALATE"

RISK_CATEGORIES = {
    "diagnosis",
    "dosing",
    "pediatrics",
    "pregnancy",
    "drug_interaction",
    "emergency",
    "low_risk_info",
}

from pydantic import BaseModel,Field

class GradeDocuments(BaseModel): 
    """Binary score for relevance check on retrieved documents.""" 
    binary_score: str = Field( description="Documents are relevant to the question, 'yes' or 'no'" )

# ----------------------------
# State
# ----------------------------
class GraphState(TypedDict, total=False):
    question: str

    # output of classifier
    risk_category: str
    safe_to_answer: str  # "yes" | "no"
    risk_rationale: str

    # router decision
    decision_mode: str

    # retrieval
    retrieved_snippets: List[str]     # cleaned strings
    relevance_yes_docs: List[Any]
    relevance_no_docs: List[Any]

    rewrite_tries: int

    # multi-hop decision
    route: Route
    wants_emergency_number: Optional[bool]  # None until user replies
    location: Optional[str]                 # user-provided (city/country/address)

    # turn management / pending follow-up
    pending_emergency_lookup: bool
    turn_count: int

    # emergency search
    emergency_lookup_query: Optional[str]
    emergency_number_result: Optional[str]

    # output
    final_answer: str

    # post-check flags
    audit_flags: Dict[str, Any]


# ----------------------------
# Helpers
# ----------------------------
def _parse_json_loose(text: str) -> Dict[str, Any]:
    import json

    s = (text or "").strip()
    if not s:
        return {"risk_category": "emergency", "safe_to_answer": "no", "rationale": "Empty output."}

    # Try direct parse first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Extract the FIRST complete JSON object by brace matching
    start = s.find("{")
    if start == -1:
        return {"risk_category": "emergency", "safe_to_answer": "no", "rationale": "No JSON found."}

    depth = 0
    end = None
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        return {"risk_category": "emergency", "safe_to_answer": "no", "rationale": "Truncated JSON."}

    json_str = s[start:end]
    try:
        obj = json.loads(json_str)
        if isinstance(obj, dict):
            return obj
        return {"risk_category": "emergency", "safe_to_answer": "no", "rationale": "JSON was not an object."}
    except Exception as e:
        return {"risk_category": "emergency", "safe_to_answer": "no", "rationale": f"Bad JSON: {e}"}

def parse_json_loose(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty output")

    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found: {s[:200]}")

    depth = 0
    end = None
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError(f"Truncated JSON: {s[start:start+200]}")

    return json.loads(s[start:end])

def generate_json(client, prompt: str, system_prompt: str, retries: int = 1) -> dict:
    raw = client.generate(prompt=prompt, system_prompt=system_prompt)

    try:
        return parse_json_loose(raw)
    except Exception:
        if retries <= 0:
            # fail-closed JSON
            return {"final_answer": "", "safety_notes": "Unable to format response as JSON.", "sources": "none"}

        # Retry: show the model its bad output and demand JSON only
        retry_prompt = (
            "Your previous output was not valid JSON.\n"
            "Return ONLY a valid JSON object matching the schema. No extra text.\n\n"
            f"USER PROMPT:\n{prompt}\n\n"
            f"PREVIOUS OUTPUT:\n{raw}\n"
        )
        raw2 = client.generate(prompt=retry_prompt, system_prompt=system_prompt)
        return parse_json_loose(raw2)


def _format_docs(docs: Any, max_docs: int = 5) -> List[str]:
    """Convert retriever docs to list[str]. Accepts LangChain Document or plain strings."""
    out: List[str] = []
    if not docs:
        return out
    for d in docs[:max_docs]:
        if isinstance(d, str):
            out.append(d)
        else:
            # LangChain Document has page_content
            content = getattr(d, "page_content", None)
            if content:
                out.append(content)
            else:
                out.append(str(d))
    return out

def parse_grade(text: str) -> GradeDocuments:
    try:
        return GradeDocuments.model_validate_json(text)
    except Exception:
        # fallback: extract JSON block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return GradeDocuments(binary_score="no")
        try:
            return GradeDocuments.model_validate_json(m.group(0))
        except Exception:
            return GradeDocuments(binary_score="no")

def keyword_score(query: str, doc_text: str) -> float:
    q = set(_tok(query))
    d = set(_tok(doc_text))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)  # query coverage

def build_graph(deps: Deps):
    # ----------------------------
    # Follow-up: YES/NO + location parsing
    # ----------------------------
    YES_PAT = re.compile(r"\b(yes|yeah|yup|sure|please|ok|okay)\b", re.I)
    NO_PAT  = re.compile(r"\b(no|nope|nah|don[' ]?t|do not)\b", re.I)
    
    def entry_router(state: GraphState) -> str:
        #print("ENTRY pending=", state.get("pending_emergency_lookup"))
        return "FOLLOWUP" if state.get("pending_emergency_lookup") else "NORMAL"

    def parse_emergency_followup(state: GraphState) -> GraphState:
        text = (state.get("question") or "").strip()
        if not text:
            return {}

        wants = state.get("wants_emergency_number")
        loc = state.get("location")

        # detect yes/no
        if YES_PAT.search(text):
            wants = True
        elif NO_PAT.search(text):
            wants = False

        # case: "in <place>"
        m = re.search(r"\bin\s+([A-Za-z0-9 ,.'-]{3,})", text, re.I)
        if m:
            loc = m.group(1).strip()

        # case: "yes, Kuala Lumpur, Malaysia."
        if wants is True and not loc:
            cleaned = YES_PAT.sub("", text).strip(" ,.-") 
            if cleaned and any(c.isalpha() for c in cleaned):
                loc = cleaned

        # case: user just types a location
        if state.get("pending_emergency_lookup") and wants is None and not loc:
            if len(text) <= 80 and any(c.isalpha() for c in text):
                loc = text
                wants = True

        return {"wants_emergency_number": wants, "location": loc}

    def followup_router(state: GraphState) -> str:
        """
        Decide what to do in follow-up mode:
        - If user said no -> end follow-up (and clear pending)
        - If user said yes + location -> do lookup
        - Else ask location
        """
        print("FOLLOWUP wants=", state.get("wants_emergency_number"), "loc=", state.get("location"))
        wants = state.get("wants_emergency_number")
        loc = state.get("location")

        if wants is False:
            return "END_NOW"
        if wants is True and loc:
            return "DO_LOOKUP"
        return "ASK_LOCATION"

    # ----------------------------
    # ask_location implementation 
    # ----------------------------
    def ask_location(state: GraphState) -> GraphState:
        """
        This is the message you show when user wants number but hasn't provided location.
        Keeps pending_emergency_lookup=True so next turn routes to FOLLOWUP again.
        """
        msg = "Sure. What's your location?"

        # If there was already an answer, append. Otherwise just return msg.
        base = state.get("final_answer", "")
        final = (base + "\n\n" + msg).strip() if base else msg

        return {
            "final_answer": final,
            "pending_emergency_lookup": True,
            "wants_emergency_number": True,  # implies user wants it
        }

    # ----------------------------
    # lookup node should clear pending flag
    # ----------------------------

    def lookup_emergency_number(state: GraphState) -> GraphState:
        loc = state["location"]
        query = f"Official emergency ambulance phone number {loc}"
        print("TAVILY QUERY:", query) 
        results = deps.tavily_search(query, 5) if deps.tavily_search else []

        if results:
            top = results[0].get("content", "") or results[0].get("title", "")
            sources = "\n".join(r.get("url", "") for r in results[:3] if r.get("url"))
            out = f"{top}\n\nSources:\n{sources}" if sources else top
        else:
            out = "I couldn't find an official number. Call your local emergency number (such as 911 / 112 / 999)."

        return {
            "emergency_lookup_query": query,
            "emergency_number_result": out,
            "final_answer": "\n\nEMERGENCY NUMBER LOOKUP:\n" + out,
            "pending_emergency_lookup": False,
        }

    # ----------------------------
    # Nodes
    # ----------------------------

    def risk_classification(state: GraphState) -> GraphState:
        question = state["question"]

        # You can implement as messages or raw prompt depending on your LLM wrapper
        log("-------CLASSIFYING RISK--------")
        prompt = f"Question: {question}"
        raw = deps.classifier_llm.generate(prompt=prompt, system_prompt=RISK_CLASSIFIER_PROMPT)
        data = _parse_json_loose(raw)

        risk_category = data.get("risk_category", "emergency")
        safe_to_answer = data.get("safe_to_answer", "no")
        rationale = data.get("rationale", "")

        if risk_category not in RISK_CATEGORIES:
            risk_category = "emergency"
            safe_to_answer = "no"

        if safe_to_answer not in ("yes", "no"):
            safe_to_answer = "no"

        return {
            "question": question,
            "risk_category": risk_category,
            "safe_to_answer": safe_to_answer,
            "risk_rationale": rationale,
        }


    def policy_router(state: GraphState) -> str:
        """Return a route key used by add_conditional_edges."""
        risk_category = state.get("risk_category", "emergency")
        safe_to_answer = state.get("safe_to_answer", "no")

        # v1 hard rules
        if risk_category == "emergency":
            return FIRST_AID_ESCALATE
        if risk_category in {"diagnosis", "dosing", "drug_interaction"}:
            return REFUSE_REDIRECT
        if risk_category in {"pediatrics", "pregnancy"}:
            return ANSWER_GENERAL if safe_to_answer == "yes" else REFUSE_REDIRECT
        if risk_category == "low_risk_info":
            return ANSWER_GENERAL

        return REFUSE_REDIRECT

    def retrieve_first_aid(state: GraphState) -> GraphState:
        """Retrieve first-aid snippets for all modes."""
        log("-------RETRIEVING FIRST AID CORPUS--------")
        q = state["question"]
        docs = deps.retriever.invoke(q)
        snippets = _format_docs(docs)
        return {"retrieved_snippets": snippets}

    def grade_retrieval(state: GraphState) -> GraphState:
        """Grade retrieved documents."""
        log("-------GRADING RETRIEVED DOCUMENTS--------")

        q = state["question"]
        docs = state.get("retrieved_snippets", []) or []

        formatted_prompt = (
                    "You MUST reply with ONLY a valid JSON object in this exact format:\n"
                    "{\n"
                    '  "binary_score": "yes" or "no"\n'
                    "}\n\n"
                    
                    "Examples:\n"
                    "Input: The document contains relevant information.\n"
                    "Output: { \"binary_score\": \"yes\" }\n\n"
                    "Input: The document is unrelated to the question.\n"
                    "Output: { \"binary_score\": \"no\" }\n\n"
                    f"Now, process the following input:\n{q}\n\nOutput (JSON only):"
                )

        yes_docs, no_docs = [], []
        for d in docs:
            doc_text = getattr(d, "page_content", str(d))
            raw = deps.answer_llm.generate(prompt=f"Retrieved document:\n{doc_text}\n\nUser question:\n{q}",system_prompt=formatted_prompt,)
            grade = parse_grade(raw)
            #is_relevant = grade.binary_score == "yes"

            if grade.binary_score.strip().lower() == "yes":
                yes_docs.append(d)
            else:
                no_docs.append(d)

        return {
            "relevance_yes_docs": yes_docs,
            "relevance_no_docs": no_docs,
            "retrieved_snippets": [getattr(d, "page_content", str(d)) for d in yes_docs],
        }

    def retrieval_quality_router(state: GraphState) -> str:
        snippets = state.get("retrieved_snippets", [])
        tries = int(state.get("rewrite_tries", 0))
        if len(snippets) >= 1:
            return "GOOD"
        if tries >= 1:          # allow only 1 rewrite (or set to 2)
            return "GIVE_UP"
        return "BAD"

    def rewrite_query(state: GraphState) -> GraphState:
        tries = int(state.get("rewrite_tries", 0)) + 1
        q = state["question"]
        rewritten = f"first aid steps for: {q}"
        return {
            "question": rewritten,
            "rewrite_tries": tries,
            "audit_flags": {**state.get("audit_flags", {}), "rewrote_query": True, "rewrite_tries": tries},
        }

    def generate_answer(state: GraphState) -> GraphState:
        q = state["question"]
        mode = state.get("decision_mode", ANSWER_GENERAL)
        snippets = state.get("retrieved_snippets", [])

        if mode == ANSWER_GENERAL:
            sys = GEN_SYSTEM_GENERAL
            user = q
        elif mode == REFUSE_REDIRECT:
            sys = GEN_SYSTEM_REFUSE
            user = q
        elif mode in (FIRST_AID_ESCALATE, PARTIAL_ESCALATE):
            sys = GEN_SYSTEM_FIRSTAID
            ctx = "\n\n".join(f"- {s}" for s in snippets) if snippets else "(no retrieved snippets)"
            user = (
                f"User question:\n{q}\n\n"
                f"Retrieved first-aid snippets:\n{ctx}\n\n"
                "Write the safest response following the instructions."
            )
        else:
            sys = GEN_SYSTEM_REFUSE
            user = q

        raw = answer_llm.generate(prompt=user, system_prompt=sys)
        return {"final_answer": raw}


    def generate_answer_and_offer_lookup(state: GraphState) -> GraphState:
        q = state["question"]
        snippets = state.get("retrieved_snippets", [])
        ctx = "\n\n".join(f"- {s}" for s in snippets) if snippets else "(no retrieved snippets)"

        user_prompt = (
            f"User question:\n{q}\n\n"
            f"First-aid snippets:\n{ctx}\n\n"
            "Write the response using the required format."
        )

        answer = deps.answer_llm.generate(prompt=user_prompt, system_prompt=GEN_SYSTEM_FIRSTAID_MULTI)

        return {
            "final_answer": answer,
            "route": "ASK_EMERGENCY_NUMBER",
            "pending_emergency_lookup": True,  
            "wants_emergency_number": state.get("wants_emergency_number"),
            "location": state.get("location"),
        }

    def emergency_lookup_router(state: GraphState) -> str:
        if state.get("wants_emergency_number") is not True:
            return "END_NOW"
        if not state.get("location"):
            return "ASK_LOCATION"
        return "DO_LOOKUP"


    #tavily = TavilySearchResults(max_results=5)  # needs TAVILY_API_KEY env var

    def emergency_rag(state: GraphState) -> GraphState:
        if deps.retriever is None:
            return {"retrieved_snippets": []}
        docs = deps.retriever.invoke(state["question"])
        return {"retrieved_snippets": _format_docs(docs)}

    def generate_answer(state: GraphState) -> GraphState:
        q = state["question"]
        mode = state.get("decision_mode", ANSWER_GENERAL)
        snippets = state.get("retrieved_snippets", [])

        if mode == ANSWER_GENERAL:
            sys, user = GEN_SYSTEM_GENERAL, q
        elif mode == REFUSE_REDIRECT:
            sys, user = GEN_SYSTEM_REFUSE, q
        else:
            sys = GEN_SYSTEM_FIRSTAID
            ctx = "\n\n".join(f"- {s}" for s in snippets) if snippets else "(no retrieved snippets)"
            user = f"User question:\n{q}\n\nRetrieved first-aid snippets:\n{ctx}\n\nWrite the safest response following the instructions."

        raw = deps.answer_llm.generate(prompt=user, system_prompt=sys)
        return {"final_answer": raw}

    def post_check(state: GraphState) -> GraphState:
        a = (state.get("final_answer") or "").lower()
        flags: Dict[str, Any] = {}

        if state.get("risk_category") == "emergency":
            flags["missing_escalation"] = not bool(
                re.search(r"\b(call (an )?ambulance|call 911|call 999|call 112|emergency services|go to the er|seek emergency care)\b", a)
            )

        flags["has_dosing"] = bool(re.search(r"\b(\d+(\.\d+)?\s?(mg|mcg|µg|g|ml|units)|dose|dosing)\b", a))
        flags["has_invasive"] = bool(re.search(r"\b(tracheotomy|tracheostomy|suture|stitch|incision|cut open)\b", a))
        return {"audit_flags": flags}

    def set_mode_general(state: GraphState) -> GraphState:
        return {"decision_mode": "GENERAL"}

    def set_mode_refuse(state: GraphState) -> GraphState:
        return {"decision_mode": "REFUSE"}

    def set_mode_first_aid_escalate(state: GraphState) -> GraphState:
        return {"decision_mode": "FIRST_AID_ESCALATE"}

    def set_mode_partial_escalate(state: GraphState) -> GraphState:
        return {"decision_mode": "PARTIAL_ESCALATE"}
    
    g = StateGraph(GraphState)

    g.add_node("risk_classification", risk_classification)
    g.add_node("post_check", post_check)

    # Follow-up nodes
    g.add_node("parse_emergency_followup", parse_emergency_followup)
    g.add_node("ask_location", ask_location)
    g.add_node("lookup_emergency_number", lookup_emergency_number)

    # Your existing nodes
    g.add_node("generate_answer", generate_answer)
    g.add_node("set_mode_general", set_mode_general)
    g.add_node("set_mode_refuse", set_mode_refuse)
    g.add_node("set_mode_first_aid_escalate", set_mode_first_aid_escalate)
    g.add_node("set_mode_partial_escalate", set_mode_partial_escalate)
    g.add_node("retrieve_first_aid", retrieve_first_aid)
    g.add_node("grade_retrieval", grade_retrieval)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate_answer_multi", generate_answer_and_offer_lookup)

    g.add_conditional_edges(
        START,
        entry_router,
        {
            "FOLLOWUP": "parse_emergency_followup",
            "NORMAL": "risk_classification",
        }
    )

    # Follow-up routing (Turn 2)
    g.add_conditional_edges(
        "parse_emergency_followup",
        followup_router,
        {
            "END_NOW": END,
            "ASK_LOCATION": "ask_location",
            "DO_LOOKUP": "lookup_emergency_number",
        }
    )

    # Risk router -> mode setters
    g.add_conditional_edges(
        "risk_classification",
        policy_router,
        {
            ANSWER_GENERAL: "set_mode_general",
            REFUSE_REDIRECT: "set_mode_refuse",
            FIRST_AID_ESCALATE: "set_mode_first_aid_escalate",
            PARTIAL_ESCALATE: "set_mode_partial_escalate",
        },
    )

    # Mode setters -> answer or RAG
    g.add_edge("set_mode_general", "generate_answer")
    g.add_edge("set_mode_refuse", "generate_answer")

    g.add_edge("set_mode_first_aid_escalate", "retrieve_first_aid")
    g.add_edge("set_mode_partial_escalate", "retrieve_first_aid")

    g.add_edge("retrieve_first_aid", "grade_retrieval")

    g.add_conditional_edges(
        "grade_retrieval",
        retrieval_quality_router,
        {
            "GOOD": "generate_answer_multi",
            "BAD": "rewrite_query",
            "GIVE_UP": "generate_answer_multi",
        }
    )

    g.add_edge("rewrite_query", "retrieve_first_aid")

    g.add_conditional_edges(
        "generate_answer_multi",
        emergency_lookup_router,
        {
            "END_NOW": "post_check",
            "ASK_LOCATION": "ask_location",
            "DO_LOOKUP": "lookup_emergency_number",
        }
    )


    # post_check only at the end
    g.add_edge("generate_answer", "post_check")
    g.add_edge("ask_location", "post_check")
    g.add_edge("lookup_emergency_number", "post_check")
    g.add_edge("post_check", END)

    return g.compile()