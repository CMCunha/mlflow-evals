from pathlib import Path
import re, yaml, ast, subprocess, difflib, tempfile, json, numpy as np
import xml.etree.ElementTree as ET

# Optional semantic model
try:
    from sentence_transformers import SentenceTransformer, util
    _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    _embed_model = None
    print('Semantic model not available; install sentence-transformers for better matching.')


# =========================================================
# STEP EXTRACTION
# =========================================================
def load_manual_steps_from_record_inputs(record_inputs):
    tc = record_inputs.get('test_case', {})
    steps = tc.get("steps", [])
    return [s["step"] for s in steps if isinstance(s, dict) and "step" in s] or []

def load_generated_steps_from_code(path, language="python", framework="playwright"):
    """
    Extracts functional step-like calls for different language/framework combos.
    """
    txt = Path(path).read_text()
    steps = []

    language = (language or "").lower()
    framework = (framework or "").lower()

    # ---------- PLAYWRIGHT ----------
    if framework == "playwright":
        # Works for both Java & Python style
        for m in re.finditer(r"\bpage\.(\w+)\(['\"]([^'\"]*)['\"]\)", txt):
            steps.append(f"{m.group(1)} {m.group(2)}")
        for m in re.finditer(r"expect\(.*?\)\.(\w+)\(([^)]*)\)", txt):
            steps.append(f"expect {m.group(1)} {m.group(2).strip()}")

    # ---------- SELENIUM ----------
    elif framework == "selenium":
        # Python Selenium driver.get(), driver.find_element, click()
        for m in re.finditer(r"driver\.(get|find_element\w*|findElement\w*|click)\(([^)]*)\)", txt):
            action = m.group(1)
            target = m.group(2).strip().strip("'\"")
            steps.append(f"{action} {target}")
        # Java Selenium element actions: element.click(), element.sendKeys()
        for m in re.finditer(r"\w+\.(click|sendKeys)\(([^)]*)\)", txt):
            steps.append(f"{m.group(1)} {m.group(2).strip()}")

    # Add fallback extraction for assertions
    for m in re.finditer(r"assert\s*(.+)", txt):
        steps.append(f"assert {m.group(1).strip()}")

    return steps


# =========================================================
# MULTI-LANGUAGE SYNTAX & EXECUTION
# =========================================================
def check_syntax_and_execution(language, path):
    """
    Run syntax and execution checks depending on language.
    For Java: uses 'javac' to verify compilation.
    For Python: uses AST and python execution.
    """
    txt_path = Path(path)
    if not txt_path.exists():
        return {'status': '❌ File not found', 'score': 0.0}

    language = (language or "").lower()

    # ---- PYTHON ----
    if language == "python":
        try:
            ast.parse(txt_path.read_text())
            syntax_ok = True
        except SyntaxError as e:
            return {'status': f'❌ Python syntax error: {e}', 'score': 0.0}

        try:
            proc = subprocess.run(['python', str(path)],
                                  capture_output=True, text=True, timeout=10)
            if proc.returncode == 0:
                return {'status': '✅ Python runs to completion', 'score': 1.0, 'log': proc.stdout[:500]}
            return {'status': '⚠️ Runtime error', 'score': 0.3, 'log': proc.stderr[:500]}
        except Exception as e:
            return {'status': f'❌ Execution failure: {e}', 'score': 0.0}

    # ---- JAVA ----
    elif language == "java":
        try:
            proc = subprocess.run(['javac', str(path)],
                                  capture_output=True, text=True, timeout=10)
            if proc.returncode == 0:
                return {'status': '✅ Java compiles successfully', 'score': 1.0, 'log': proc.stdout[:500]}
            return {'status': '⚠️ Compilation error', 'score': 0.3, 'log': proc.stderr[:500]}
        except Exception as e:
            return {'status': f'❌ Java check failed: {e}', 'score': 0.0}

    # ---- UNKNOWN ----
    else:
        return {'status': f'⚠️ Unsupported language: {language}', 'score': 0.0}

# =========================================================
# BASIC VALIDATION
# =========================================================
def check_python_syntax(path):
    try:
        ast.parse(Path(path).read_text())
        return {'status': '✅ Valid Python', 'score': 1.0}
    except SyntaxError as e:
        return {'status': '❌ Python syntax error', 'score': 0.0, 'error': str(e)}


def check_python_execution(path):
    try:
        proc = subprocess.run(
            ['python', str(path)],
            capture_output=True,
            text=True,
            timeout=10
        )
        if proc.returncode == 0:
            return {'status': '✅ Runs to completion', 'score': 1.0, 'log': proc.stdout[:1000]}
        else:
            return {'status': '⚠️ Runtime error', 'score': 0.3, 'log': proc.stderr[:1000]}
    except Exception as e:
        return {'status': f'❌ Execution failure: {e}', 'score': 0.0, 'log': str(e)}


def check_assertions(path):
    src = Path(path).read_text()
    count = len(re.findall(r'assert|client\.assert_', src))
    if count == 0:
        return {'status': '⚠️ Missing assertions', 'score': 0.0, 'count': 0}
    if count < 2:
        return {'status': '🟡 Basic assertions present', 'score': 0.7, 'count': count}
    return {'status': '🟢 Assertions adequate', 'score': 1.0, 'count': count}


def check_code_quality(path):
    src = Path(path).read_text()
    hardcoded = len(re.findall(r"client\.type\('.*',\s*'.+'\)", src))
    comments = len(re.findall(r'#', src))
    long_lines = sum(1 for line in src.splitlines() if len(line) > 100)

    score = 1.0
    issues = []

    if hardcoded > 3:
        issues.append("Too many hardcoded values")
        score -= 0.2
    if long_lines > 5:
        issues.append("Excessively long lines")
        score -= 0.2
    if comments < 1:
        issues.append("No comments (poor readability)")
        score -= 0.2

    return {
        'status': '🟢 Readable' if score >= 0.8 else '🟡 Needs cleanup',
        'score': round(score, 2),
        'issues': issues
    }


# =========================================================
# NEW IMPORT VALIDATION
# =========================================================
def check_imports(path, language="python"):
    """
    Validate import syntax for Python or Java.
    """
    txt = Path(path).read_text()
    language = (language or "").lower()

    if language == "java":
        valid = all(l.strip().startswith("import ") or l.strip() == "" for l in txt.splitlines())
        return {
            "status": "🟢 Java imports valid" if valid else "❌ Invalid Java imports",
            "score": 1.0 if valid else 0.0,
            "issues": [] if valid else ["Invalid Java import syntax"]
        }

    # Default: Python check
    import_lines = [l for l in txt.splitlines() if l.strip()]
    invalid = [l for l in import_lines if not (l.startswith("import ") or l.startswith("from "))]
    try:
        ast.parse(txt)
        syntax_ok = True
    except SyntaxError:
        syntax_ok = False

    score = 1.0
    issues = []
    if invalid:
        score -= 0.5
        issues.append("Invalid import lines present")
    if not syntax_ok:
        score = 0.0
        issues.append("Python syntax error in imports")

    return {"status": "🟢 Imports valid" if score > 0.8 else "❌ Invalid imports",
            "score": round(score, 2),
            "issues": issues}



# =========================================================
# NEW CONFIG VALIDATION
# =========================================================
def check_config_file(path, language=None, framework=None):
    """
    Validate configuration files dynamically based on content type.
    Supports:
      - XML
      - JSON
      - YAML
      - Java/Python .properties
      - .env (KEY=VALUE)
    Returns a dict with {status, score, type, error?}
    """
    p = Path(path)
    if not p.exists():
        return {"status": "❌ Config file not found", "score": 0.0, "type": "missing"}

    txt = p.read_text().strip()
    if not txt:
        return {"status": "⚠️ Empty config file", "score": 0.0, "type": "empty"}

    ext = p.suffix.lower()

    # Heuristic detection if extension is missing or ambiguous
    def looks_like_xml(t): return t.lstrip().startswith("<")
    def looks_like_json(t): return t.lstrip().startswith("{")
    def looks_like_yaml(t): return ":" in t and not looks_like_json(t)
    def looks_like_props(t): return re.match(r"^[A-Za-z0-9_.-]+=.*", t)

    # ---------- XML ----------
    if ext == ".xml" or looks_like_xml(txt):
        try:
            ET.fromstring(txt)
            return {"status": "🟢 Valid XML", "score": 1.0, "type": "xml"}
        except ET.ParseError as e:
            return {"status": "❌ Invalid XML", "score": 0.0, "type": "xml", "error": str(e)}

    # ---------- JSON ----------
    elif ext == ".json" or looks_like_json(txt):
        try:
            json.loads(txt)
            return {"status": "🟢 Valid JSON", "score": 1.0, "type": "json"}
        except json.JSONDecodeError as e:
            return {"status": "❌ Invalid JSON", "score": 0.0, "type": "json", "error": str(e)}

    # ---------- YAML ----------
    elif ext in [".yaml", ".yml"] or looks_like_yaml(txt):
        try:
            yaml.safe_load(txt)
            return {"status": "🟢 Valid YAML", "score": 1.0, "type": "yaml"}
        except yaml.YAMLError as e:
            return {"status": "❌ Invalid YAML", "score": 0.0, "type": "yaml", "error": str(e)}

    # ---------- PROPERTIES / INI ----------
    elif ext in [".properties", ".ini"] or looks_like_props(txt):
        try:
            config = configparser.ConfigParser()
            # Allow .properties files without section headers
            if not re.match(r"\[.*\]", txt):
                txt = "[default]\n" + txt
            config.read_string(txt)
            return {"status": "🟢 Valid properties/INI", "score": 1.0, "type": "properties"}
        except Exception as e:
            return {"status": "❌ Invalid properties/INI", "score": 0.0, "type": "properties", "error": str(e)}

    # ---------- .ENV ----------
    elif "=" in txt and all("=" in line for line in txt.splitlines() if line.strip()):
        invalid = [line for line in txt.splitlines() if "=" not in line]
        if invalid:
            return {"status": "⚠️ Invalid .env entries", "score": 0.5, "type": "env", "error": f"Invalid lines: {invalid[:2]}"}
        return {"status": "🟢 Valid .env", "score": 1.0, "type": "env"}

    # ---------- UNKNOWN ----------
    return {"status": "⚠️ Unknown config format", "score": 0.2, "type": "unknown"}

# =========================================================
# SEMANTIC MATCHING
# =========================================================
def semantic_matrix(list_a, list_b):
    if _embed_model is None or not list_a or not list_b:
        return np.zeros((len(list_a), len(list_b)))
    a_emb = _embed_model.encode(list_a, convert_to_tensor=True)
    b_emb = _embed_model.encode(list_b, convert_to_tensor=True)
    return util.cos_sim(a_emb, b_emb).cpu().numpy()

def semantic_diff(manual_steps, gen_steps, threshold_match=0.80, threshold_partial=0.4):
    if not manual_steps:
        return {'matched': [], 'partial': [], 'missing': manual_steps}

    sim_matrix = semantic_matrix(manual_steps, gen_steps)

    matched, partial, missing = [], [], []

    for i, step in enumerate(manual_steps):
        if not gen_steps:
            missing.append(step)
            continue

        best_idx = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_idx])

        if best_score >= threshold_match:
            matched.append((step, gen_steps[best_idx], round(best_score, 3)))
        elif best_score >= threshold_partial:
            partial.append((step, gen_steps[best_idx], round(best_score, 3)))
        else:
            missing.append(step)

    return {'matched': matched, 'partial': partial, 'missing': missing}


def check_functional(record_inputs, code_path):
    language = record_inputs.get("language", "python")
    framework = record_inputs.get("testing_framework", "playwright")

    manual = load_manual_steps_from_record_inputs(record_inputs)
    gen = load_generated_steps_from_code(code_path, language, framework)
    diff = semantic_diff(manual, gen)

    matched_scores = [x[2] for x in diff['matched'] + diff['partial']]
    score = float(np.mean(matched_scores)) if matched_scores else 0.0

    status = (
        "🟢 Matches expected steps" if score > 0.85 else
        "🟡 Partially correct" if score > 0.5 else
        "❌ Incorrect automation"
    )

    return {"status": status, "score": round(score, 3), "semantic_diff": diff}



# =========================================================
# MULTI-FILE DIFF
# =========================================================
def compute_diff(path_a, path_b, label):
    if not path_a or not Path(path_a).exists(): return ""
    if not path_b or not Path(path_b).exists(): return ""

    a = Path(path_a).read_text().splitlines()
    b = Path(path_b).read_text().splitlines()

    return "\n".join(difflib.unified_diff(
        a, b,
        fromfile=f"baseline_{label}",
        tofile=f"generated_{label}",
        lineterm=""
    ))


# =========================================================
# MAIN ENTRY
# =========================================================
def evaluate(
    inputs,
    generated_code,
    generated_imports,
    generated_config,
    baseline_code=None,
    baseline_imports=None,
    baseline_config=None
):
    # Code checks
    language = inputs.get("language", "python")
    framework = inputs.get("testing_framework", "playwright")

    syntax_exec = check_syntax_and_execution(language, generated_code)
    imports_check = check_imports(generated_imports, language)
    functional = check_functional(inputs, generated_code)
    assertions = check_assertions(generated_code)
    quality = check_code_quality(generated_code)
    config_check = check_config_file(generated_config,language,framework)
    # Weighted scoring
    weights = {
        "code": 0.60,
        "imports": 0.20,
        "config": 0.20
    }

    code_score = (
        0.40 * syntax_exec['score'] +  # combined syntax + compilation/run
        0.30 * functional['score'] +
        0.20 * assertions['score'] +
        0.10 * quality['score']
    )

    overall_score = (
        weights["code"] * code_score +
        weights["imports"] * imports_check['score'] +
        weights["config"] * config_check['score']
    )

    # Diffs
    diff_code = compute_diff(baseline_code, generated_code, "code")
    diff_imports = compute_diff(baseline_imports, generated_imports, "imports")
    diff_config = compute_diff(baseline_config, generated_config, "config")

    # Human readable semantic diff
    semantic_text = []
    for s, g, sc in functional["semantic_diff"]["matched"]:
        semantic_text.append(f"MATCHED: '{s}' -> '{g}' (score={sc})")
    for s, g, sc in functional["semantic_diff"]["partial"]:
        semantic_text.append(f"PARTIAL: '{s}' -> '{g}' (score={sc})")
    for s in functional["semantic_diff"]["missing"]:
        semantic_text.append(f"MISSING: '{s}'")

    
    return {
        "overall_weighted_score": round(overall_score, 3),

        # Include detected language/framework context
        "context": {
            "language": language,
            "framework": framework
        },

        "code": {
            "syntax_and_execution": syntax_exec,      # unified syntax+run or compilation check
            "functional": functional,                 # semantic comparison between manual and code steps
            "assertions": assertions,                 # count of assertions or validation logic
            "quality": quality,                       # readability & hardcoded value check
            "semantic_diff_text": "\n".join(semantic_text)
        },

        "imports": imports_check,
        "config": config_check,

        "baseline_diffs": {
            "code": diff_code,
            "imports": diff_imports,
            "config": diff_config
        }
    }

