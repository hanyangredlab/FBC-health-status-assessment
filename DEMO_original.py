from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from starlette.applications import Starlette
from starlette.routing import Mount, Host
from mcp.server.fastmcp import FastMCP
import contextlib
from starlette.middleware.cors import CORSMiddleware
import asyncio, csv, json, math, os, re, uuid, pathlib, time
from collections import Counter
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field
from pypdf import PdfReader
import uvicorn

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")  # 현재 작업 디렉터리 기준
print(os.getenv("OPENAI_API_KEY"))

mcp = FastMCP("mfix-bc", 
              host="127.0.0.1", 
              port=8000, 
            #   sse_path="/sse", 
            #   message_path="/messages",
              stateless_http=True
              )

@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(mcp.session_manager.run())
        yield
        
app = Starlette(
    routes=[
        Mount("/mcp", mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)
app = CORSMiddleware(
    app,
    allow_origins=["*"],  # Configure appropriately for production
    allow_methods=["GET", "POST", "DELETE"],  # MCP streamable HTTP methods
    expose_headers=["Mcp-Session-Id"],
)


# =========================
# Config
# =========================

MFX_ROOT = pathlib.Path(os.getenv("MFX_ROOT", "./refs_mfx")).resolve()
# KEYWORDS_PDF = pathlib.Path(os.getenv("KEYWORDS_PDF", "./Keywords.pdf")).resolve()
SERVER_LABEL = os.getenv("SERVER_LABEL", "mfix-bc-sse")
MFX_ROOT.mkdir(parents=True, exist_ok=True)
# KEYWORDS_JSON = pathlib.Path(os.getenv("KEYWORDS_JSON", "./Keywords.json")).resolve()
USEFUL_KEYWORDS_JSON = pathlib.Path(os.getenv("USEFUL_KEYWORDS_JSON", "./UsefulKeywords.json")).resolve()

# Optional 10,500 failure-scenario reference dataset.
# Default path matches Ablationstudy_leg/resources/fbc_scenario_generator_training_10500.json
# when this server file is placed at the project root. Override with:
#   export FAILURE_SCENARIOS_JSON=/path/to/fbc_scenario_generator_training_10500.json
FAILURE_SCENARIOS_JSON = pathlib.Path(
    os.getenv("FAILURE_SCENARIOS_JSON", os.getenv("FAILURE_SCENARIOS_PATH", "./resources/fbc_scenario_generator_training_10500.json"))
).resolve()


# =========================
# Utilities
# =========================

_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\([^\)]*\))?)\s*=\s*(.*?)\s*(?:!.*)?$")
_ASSIGN_KEYWORD_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\(([^)]*)\)\s*$")
# DIFF_SCHEMA = {
#     "type": "array",
#     "items": {
#         "type": "object",
#         "properties": {
#                 "location": { "type": "number"},
#                 "keyword": { "type": "string" },
#                 "value":   { "type": "number" }
#             },
#         "required": ["location","keyword","value"] 
#     }
# }

_keywords_json_cache = None
_keywords_json_mtime = None
_keywords_min = None
_keywords_max = None
_keywords_to_physical_json_cache = None
_keywords_to_physical_json_mtime = None
_mfx_parsed = None  # 캐시된 mfx 파싱 결과
_failure_scenarios_cache = None
_failure_scenarios_mtime = None
_failure_scenarios_metadata = None

# =========================
# LLM 프롬프트
# =========================

# # Initial prompt v1
# @mcp.prompt()
# def mfx_proposer_prompt(current_situation: str, goal_situation: str) -> str:
#     """
#     Do these tasks under the conditions.
#     """
    

#     return {
#         "You are a expert in fluidized bed combustor as well as a *.mfx modifier, given replication.mfx which is fluidized bed combustor with char and glass beads.\n"+
#         f"Given the current situation :{current_situation}\n"+
#         f"Explain what could happen, based on the keywords and their physical meanings by parsing :keywords_to_physical.json\n"
#         "Then, do the next step:\n\n"+
#         f"Based on the dictionary parsing :keywords_to_physical.json\n"+
#         f"Find which keyword is important to achieve the goal situation :{goal_situation}\n"+
#         f"Also, refer to the normal condition boiler mfx file by parsing :replication.mfx\n"+
#         "Find the appropriate keywords and propose modified values.\n"+
#         f"And ensure the value is over {_keywords_min}, under {_keywords_max}\n"+
#         f"Get the line number of the keyword found in the mfx file by parsing :replication.mfx\n"+
#         "Finally, make sure the keywords are exactly same with keywords in the keywords json file and the mfx file.\n"
#         "If you cannot find any appropriate keywords, return an empty list.\n"
#     }


# Refined prompt with 2 steps, v2
@mcp.prompt()
def analyze_and_identify_keywords(current_situation: str, goal_situation: str, keywords_info: dict, failure_scenarios_context: Any = None) -> dict:
    """
    You are an expert in fluidized bed combustors. Analyze the given situations and identify critical keywords.
    
    PLEASE NOTE: DON'T TRANSFORM THE KEYWORDS.
    """
    if failure_scenarios_context is None:
        try:
            # Keep the original Stage-1 prompt structure, but automatically attach
            # compact references from the 10,500 failure-scenario dataset when available.
            failure_scenarios_context = retrieve_relevant_failure_scenarios_impl(
                goal_situation, top_k=5, max_chars=5000
            )
        except Exception as exc:
            failure_scenarios_context = {"error": f"failure scenario retrieval failed: {exc}"}
    
    return {
        "must_keep": "keywords must match the format (*(*)), in the provided dictionary : res://keywords/json",
        "role": "structural engineer agent",
        "goals": "Analyze the current situation and identify keywords crucial for achieving the goal.",
        "response_format": {
            "analysis": "Brief explanation of the current and goal situation.",
            "important_keywords": ["keyword_1", "keyword_2", "etc."]
        },
        "dialogue": (
            f"Current Situation: {current_situation}\n"
            f"Goal Situation: {goal_situation}\n"
            f"Keywords Data: {keywords_info}\n"
            f"Failure Scenario References: {failure_scenarios_context if failure_scenarios_context else 'Not provided'}"
        )
    }
@mcp.prompt()
def propose_modifications(important_keywords: list, mfx_content_snippet: str, keywords_info: dict) -> dict:
    f"""
    You are a .mfx file modification expert. Based on the provided keywords and file content, propose modified values.

    PLESASE NOTE: If there are constraint keywords in the {keywords_info}, propose the value for them too.
    """
    
    return {
        "role": "appropriate modification agent",
        "goals": f"Propose new random values between {_keywords_min} and {_keywords_max} for the given keywords based on the mfx content.",
        "constraints": "Ensure proposed values are reasonable for a fluidized bed combustor system and clip them within the specified min and max ranges.",
        "response_format": {
            "modifications_proposal": [
                {
                    "keyword": "keyword_name",
                    "proposed_value": "new_value"
                }
            ]
        },
        "notes": f"Line number lookup will be handled by the system after this step.",

        "dialogue": f"Keywords to modify: {important_keywords}\nRelevant MFX content snippet: {mfx_content_snippet}\nKeywords Data: {keywords_info}"
    }

    

# =========================
# Parsers
# =========================
@mcp.tool()
def parse_mfx_and_get_linenumber(keyword_from_json: str) -> dict:
    """
    Args:
        keyword_from_json : The keyword from keywords_to_physical.json in the format KEY(ARGS)
        
    Parse the .mfx file at the given path relative to MFX_ROOT.
    
    Returns a dictionary with:
    - _mfx_parsed: { "keyword": str, "location": int }
    """
    path = "Resources/replication.mfx"
    mfx = parse_mfx_impl(path)
    linenumber=find_location(keyword_from_json, mfx)

    return {"keyword": keyword_from_json, "linenumber": linenumber}

def parse_keyword(key: str) -> Optional[Dict[str, Any]]:
    """
    Parse a keyword string into its base and arguments.
    
    Args:
        key (str): The keyword string to parse
    
    Returns:
        Optional[Dict[str, Any]]: A dictionary with 'base' and 'args' if parsing is successful, otherwise None.
    """
    m = _ASSIGN_KEYWORD_RE.match(key)
    if m:
        base = m.group(1)
        args = [arg.strip() for arg in m.group(2).split(",")]
        return {"base": base, "args": args}
    return None
# @mcp.tool()
# def parse_physical_key(key: str) -> Optional[Dict[str, Any]]:
#     """
#     Parse a physical keyword string into its base and arguments.
    
#     Args:
#         key (str): The physical keyword string to parse
    
#     Returns:
#         Optional[Dict[str, Any]]: A dictionary with 'base' and 'args' if parsing is successful, otherwise None.
#     """
#     return parse_keyword(key)

def find_location(keyword: str, mfx: str) -> Optional[int]:
    """
    After filling the arguments of keyword,
    Find the line number of the keyword in replication.mfx.
    """
    # if not _keywords_to_physical_json_cache:
    #     _ = load_keywords_to_physical_json()
    # if not _mfx_parsed:
    #     _ = parse_mfx_impl(MFX_ROOT / "Resources/replication.mfx")
    # if not _mfx_parsed:
    #     _ = load_keywords_json()

    if not keyword or not isinstance(keyword, str):
        return None
    entry = mfx.get(keyword)
    if entry:
        return entry["location"]
    return None


# =========================
# resources
# =========================

def parse_mfx_impl(path: str) -> dict:
    """
    Parse the .mfx file at the given path relative to MFX_ROOT.
    
    Returns a dictionary with:
    - _mfx_parsed: { "keyword": {"location": int, "value": str}, ...}
    """
    if not path or path.startswith(("/", "~")):
        raise ValueError("path must be a relative path under MFX_ROOT")
    fpath = (MFX_ROOT / path).resolve()
    if not str(fpath).startswith(str(MFX_ROOT)):
        raise ValueError("Path outside MFX_ROOT")
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {path}")

    
    try:
        text: str = read_text(fpath)
    except UnicodeDecodeError:
        # 혹시 모를 디코딩 이슈에 대한 폴백
        text = fpath.read_text(encoding="latin-1")

    raw_lines = text.splitlines()

    # parse every assignments
    global _mfx_parsed
    _mfx_parsed = {}
    for i, L in enumerate(raw_lines):
        m = _ASSIGN_RE.match(L)
        if m:
            _mfx_parsed[m.group(1).strip()] = {
                "location": i+1,
                "value": m.group(2).strip()
            }

    return _mfx_parsed

def read_text(path: pathlib.Path) -> str:
    """
    Read sample of replication.mfx files in the path.

    Args:
        path (pathlib.Path): The path to the replication.mfx files.

    Returns:
        str: The text content of the replication.mfx files as str type.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

# =========================
# Failure scenario dataset parser / retriever
# =========================

def _flatten_strings(obj: Any) -> List[str]:
    """Collect readable strings from nested dict/list scenario records."""
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            out.extend(_flatten_strings(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_flatten_strings(item))
    elif isinstance(obj, (str, int, float, bool)) and obj is not None:
        out.append(str(obj))
    return out


def _tokenize_for_retrieval(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_\.\-]+", str(text).lower())


def _scenario_record_to_text(record: Any) -> str:
    """Create compact searchable text from one failure-scenario item."""
    if isinstance(record, dict):
        rid = record.get("id") or record.get("scenario_id") or record.get("content_hash") or ""
        instruction = record.get("instruction", "")
        input_obj = record.get("input", {})
        output_obj = record.get("output", {})
        scenario = output_obj.get("scenario", output_obj) if isinstance(output_obj, dict) else output_obj
        parts = [str(rid), str(instruction)]
        if isinstance(input_obj, dict):
            parts.extend(_flatten_strings(input_obj))
        else:
            parts.append(str(input_obj))
        if isinstance(scenario, dict):
            preferred_keys = [
                "scenario_id", "scenario_name", "category", "subsystem", "failure_mode",
                "initiating_event", "operating_mode", "fuel_type", "causal_chain",
                "simulation_objective", "mfix_exa_simulation_controls", "observables",
                "trip_logic", "mitigation",
            ]
            for key in preferred_keys:
                if key in scenario:
                    parts.extend(_flatten_strings({key: scenario[key]}))
        else:
            parts.extend(_flatten_strings(scenario))
        return " | ".join(p for p in parts if p)
    return str(record)


def load_failure_scenarios() -> List[Dict[str, Any]]:
    """Load the 10,500 failure-scenario reference dataset with mtime cache."""
    global _failure_scenarios_cache, _failure_scenarios_mtime, _failure_scenarios_metadata

    path = FAILURE_SCENARIOS_JSON
    if not path.exists():
        _failure_scenarios_cache = []
        _failure_scenarios_metadata = {
            "path": str(path),
            "exists": False,
            "error": "Failure scenario dataset not found. Set FAILURE_SCENARIOS_JSON or FAILURE_SCENARIOS_PATH.",
        }
        return []

    mt = path.stat().st_mtime
    if _failure_scenarios_cache is not None and mt == _failure_scenarios_mtime:
        return _failure_scenarios_cache

    suffix = path.suffix.lower()
    records: List[Any] = []
    metadata: Dict[str, Any] = {"path": str(path), "exists": True, "suffix": suffix}

    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        else:
            obj = json.loads(path.read_text(encoding="utf-8"))
            metadata["dataset_name"] = obj.get("dataset_name") if isinstance(obj, dict) else None
            metadata["num_items_declared"] = obj.get("num_items") if isinstance(obj, dict) else None
            if isinstance(obj, dict) and isinstance(obj.get("items"), list):
                records = obj["items"]
            elif isinstance(obj, list):
                records = obj
            else:
                records = [obj]
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            records = [dict(row) for row in reader]
            metadata["columns"] = reader.fieldnames or []
    else:
        records = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]

    parsed: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        scenario_id = None
        if isinstance(record, dict):
            scenario_id = record.get("id")
            out = record.get("output")
            if not scenario_id and isinstance(out, dict) and isinstance(out.get("scenario"), dict):
                scenario_id = out["scenario"].get("scenario_id")
        parsed.append({
            "index": i,
            "id": scenario_id or f"scenario_{i:05d}",
            "text": _scenario_record_to_text(record),
            "raw": record,
        })

    metadata["num_items_loaded"] = len(parsed)
    _failure_scenarios_cache = parsed
    _failure_scenarios_mtime = mt
    _failure_scenarios_metadata = metadata
    return parsed


def _cosine_score(query_tokens: Counter, doc_tokens: Counter) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    keys = set(query_tokens) | set(doc_tokens)
    dot = sum(query_tokens.get(k, 0) * doc_tokens.get(k, 0) for k in keys)
    qnorm = math.sqrt(sum(v * v for v in query_tokens.values()))
    dnorm = math.sqrt(sum(v * v for v in doc_tokens.values()))
    if qnorm == 0 or dnorm == 0:
        return 0.0
    return dot / (qnorm * dnorm)


@mcp.resource("res://failure_scenarios/summary")
def res_failure_scenarios_summary() -> str:
    """Return metadata and a tiny sample of the 10,500 failure-scenario reference dataset."""
    scenarios = load_failure_scenarios()
    sample = [{"id": s["id"], "text": s["text"][:500]} for s in scenarios[:3]]
    return json.dumps({"metadata": _failure_scenarios_metadata, "sample": sample}, ensure_ascii=False)


@mcp.tool()
def parse_failure_scenarios(limit: int = 5) -> dict:
    """Parse the failure-scenario dataset and return metadata plus a small sample.

    Use this tool to verify that the 10,500 failure-scenario reference file is
    available before retrieving scenario examples. It intentionally returns only a
    small sample to avoid flooding the LLM context.
    """
    scenarios = load_failure_scenarios()
    return {
        "metadata": _failure_scenarios_metadata,
        "sample": [
            {"id": s["id"], "index": s["index"], "text": s["text"][:1000]}
            for s in scenarios[: max(0, min(int(limit), 20))]
        ],
    }


def retrieve_relevant_failure_scenarios_impl(goal_situation: str, top_k: int = 5, max_chars: int = 6000) -> dict:
    """Internal retrieval implementation shared by Stage-1 prompt and MCP tool."""
    scenarios = load_failure_scenarios()
    if not scenarios:
        return {"metadata": _failure_scenarios_metadata, "scenarios": []}

    q = Counter(_tokenize_for_retrieval(goal_situation))
    scored = []
    for s in scenarios:
        d = Counter(_tokenize_for_retrieval(s["text"]))
        score = _cosine_score(q, d)
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    used = 0
    for rank, (score, s) in enumerate(scored[: max(0, int(top_k))], start=1):
        text = s["text"]
        if used + len(text) > max_chars:
            remaining = max(0, max_chars - used)
            if remaining <= 200:
                break
            text = text[:remaining] + " ... [truncated]"
        out.append({
            "rank": rank,
            "score": round(float(score), 6),
            "id": s["id"],
            "index": s["index"],
            "text": text,
        })
        used += len(text)
    return {"metadata": _failure_scenarios_metadata, "query": goal_situation, "scenarios": out}


@mcp.tool()
def find_relevant_failure_scenarios(goal_situation: str, top_k: int = 5, max_chars: int = 6000) -> dict:
    """Retrieve relevant examples from the 10,500 failure-scenario reference dataset.

    Args:
        goal_situation: User's target fault/scenario description.
        top_k: Number of similar failure scenarios to return.
        max_chars: Approximate maximum total text budget for returned scenario texts.

    Returns:
        Dictionary with metadata and top matched simulation-only failure scenarios.
        These examples are references for scenario grounding, not direct plant instructions.
    """
    return retrieve_relevant_failure_scenarios_impl(goal_situation, top_k=top_k, max_chars=max_chars)

@mcp.resource("res://replication.mfx")
def res_replication_mfx() -> str:
    return parse_mfx_impl(MFX_ROOT / "Resources/replication.mfx")

def load_keywords_to_physical_json() -> dict:
    global _keywords_to_physical_json_cache, _keywords_to_physical_json_mtime
    if not USEFUL_KEYWORDS_JSON.exists():
        raise FileNotFoundError(f"Keywords.json not found at {USEFUL_KEYWORDS_JSON}")
    mt = USEFUL_KEYWORDS_JSON.stat().st_mtime
    if _keywords_to_physical_json_cache is None or mt != _keywords_to_physical_json_mtime:
        _keywords_to_physical_json_cache = json.loads(USEFUL_KEYWORDS_JSON.read_text(encoding="utf-8"))
        _keywords_to_physical_json_mtime = mt
    return _keywords_to_physical_json_cache

@mcp.resource("res://keywords/json")
def res_keywords_to_physical_json() -> str:
    data = load_keywords_to_physical_json()
    # 리소스는 문자열이 가장 호환성 좋음
    return json.dumps(data, ensure_ascii=False)

@mcp.tool()
def find_relevant_keywords(goal_situation: str) -> list:
    """
    Finds keywords from the keywords JSON file that are most relevant to the goal situation.
    This tool reduces token usage by only returning a filtered list of keywords.

    Args:
        goal_situation (str): The user's goal situation description.

    Returns:
        list: A list of relevant keywords in the format KEY(ARGS).
    """
    try:
        global _keywords_min, _keywords_max
        _keywords_min = {}
        _keywords_max = {}
        keywords_data = load_keywords_to_physical_json()
        relevant_keywords = {}
        
        # goal_situation을 분석하여 관련 키워드를 찾는 로직
        # 예시: 'goal_situation'에 특정 단어(예: 'pressure')가 포함되면 해당 키워드 검색
        search_terms = goal_situation.lower().split()
        
        for key, details in keywords_data.items():
            if 'description' in details:
                meaning_text = details['description'].lower()
                # 'goal_situation'의 키워드와 'description'을 비교하여 관련성 판단
                if any(term in meaning_text for term in search_terms):
                    relevant_keywords[key] = details
                    if 'min' in details or 'max' in details:
                        _keywords_min[key] = details.get("min", None)
                        _keywords_max[key] = details.get("max", None)

        return relevant_keywords

    except FileNotFoundError:
        return {"error": "Keywords JSON file not found."}



# =========================
# Server tools
# =========================

@mcp.tool()
async def long_running_task(task_name: str, ctx: Context[ServerSession, None], steps: int = 5) -> str:
    """Execute a task with progress updates."""
    await ctx.info(f"Starting: {task_name}")

    for i in range(steps):
        progress = (i + 1) / steps
        await ctx.report_progress(
            progress=progress,
            total=1.0,
            message=f"Step {i + 1}/{steps}",
        )
        await ctx.debug(f"Completed step {i + 1}")

    return f"Task '{task_name}' completed"




if __name__ == "__main__":
    
    mcp.run(transport="streamable-http")