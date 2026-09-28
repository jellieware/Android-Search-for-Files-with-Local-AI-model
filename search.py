import os
import re
import sys
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Termux/Android note:
# MediaPipe GenAI is intentionally NOT used here.  It is not required.
# Natural-language parsing is done with the LiteRT-LM command-line runtime
# when available, with a local heuristic parser as a fallback.

HAS_LITERT_CLI = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DEFAULT_MODEL_PATH = os.getenv(
    "LITERT_MODEL_PATH",
    "/storage/emulated/0/gemma4.litertlm"
)

# You can set this to the full path of litert-lm / litert_lm_main.
# Examples:
#   export LITERT_CLI_PATH=$HOME/bin/litert-lm
#   export LITERT_CLI_PATH=$HOME/bin/litert_lm_main
LITERT_CLI_BINARY = os.getenv("LITERT_CLI_PATH", "litert-lm")



SYSTEM_PROMPT = """You are an intelligent Android file search assistant. 
Analyze the user's natural language request and convert it into a search filter JSON object.

Use your world knowledge to infer intent, synonyms, and Android file structures:
- "photos/pictures/shots" -> extensions: ["jpg", "jpeg", "png", "heic", "webp"]
- "videos/movies/clips" -> extensions: ["mp4", "mkv", "mov", "avi"]
- "documents/bills/receipts/notes" -> extensions: ["pdf", "docx", "txt", "xlsx"]
- "music/audio/voice" -> extensions: ["mp3", "m4a", "wav", "flac"]
- "apps/installers" -> extensions: ["apk", "xapk"]
- "archives/compressed" -> extensions: ["zip", "rar", "7z", "tar", "gz"]

JSON Schema Output Required:
{
  "explanation": "Short 1-sentence explanation of what you understood",
  "target_storage": "internal" | "external" | "all",
  "directories": ["folder1", "folder2"],
  "extensions": ["ext1", "ext2"],
  "keywords": ["word1", "word2"],
  "max_days_old": integer or null,
  "min_days_old": integer or null
}

Output strictly valid JSON inside ```json ... ``` blocks. No extra conversation."""


def sanitize_storage_path(path: Path) -> Path:
    """Strips /Android/data/... off resolved paths to get clean drive roots."""
    path_str = str(path.resolve())
    if "/Android/data/" in path_str:
        path_str = path_str.split("/Android/data/")[0]
    return Path(path_str)


def detect_storage_paths() -> dict:
    """Automatically detects internal storage and all mounted external SD cards / USB drives."""
    found_storages = {}

    # 1. Default Internal Storage
    internal_path = sanitize_storage_path(Path("/sdcard"))
    found_storages["internal"] = internal_path

    # 2. Check Termux symlinks in ~/storage
    termux_storage_dir = Path.home() / "storage"
    if termux_storage_dir.exists():
        for link in termux_storage_dir.glob("external*"):
            try:
                real_path = sanitize_storage_path(link.resolve(strict=True))
                if real_path.exists() and real_path not in found_storages.values():
                    found_storages[f"external_{link.name}"] = real_path
            except (OSError, RuntimeError):
                pass

    # 3. Direct inspection of /storage directory (scanning for XXXX-XXXX volume IDs)
    storage_root = Path("/storage")
    if storage_root.exists():
        try:
            for entry in storage_root.iterdir():
                if entry.name in ["emulated", "self", "knox", "container"]:
                    continue
                
                if re.match(r'^[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}$', entry.name) and entry.is_dir():
                    clean_path = sanitize_storage_path(entry)
                    if clean_path not in found_storages.values():
                        found_storages[f"external_{entry.name}"] = clean_path
        except PermissionError:
            pass

    return found_storages


class LiteRTFileSearch:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.storages = detect_storage_paths()
        self.engine = None

        print("\n[+] Active Storage Drives Detected:")
        for name, path in self.storages.items():
            print(f"  └─ [{name.upper()}]: {path}")

        self.cli_binary = self._find_litert_cli()
        if self.cli_binary:
            print(f"[+] LiteRT-LM CLI: {self.cli_binary}")
        else:
            print("[!] LiteRT-LM CLI not found. Using offline natural-language parser.")


    def _find_litert_cli(self):
        """Find a LiteRT-LM executable without requiring MediaPipe."""
        candidates = [
            LITERT_CLI_BINARY,
            "litert-lm",
            "litert_lm_main",
            str(Path.home() / "bin" / "litert-lm"),
            str(Path.home() / "bin" / "litert_lm_main"),
        ]

        for candidate in candidates:
            try:
                if os.path.sep in candidate:
                    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                        return candidate
                else:
                    result = subprocess.run(
                        ["sh", "-c", f"command -v {candidate}"],
                        capture_output=True,
                        text=True,
                        timeout=3
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        return result.stdout.strip()
            except Exception:
                pass

        return None

    def _run_via_cli(self, prompt: str) -> str:
        """
        Run LiteRT-LM directly.

        Supports:
          1. `litert-lm run MODEL --prompt "..."`
          2. `litert_lm_main --backend=cpu --model_path=MODEL
             --input_prompt="..."`

        No MediaPipe/Python package is required.
        """
        if not self.cli_binary or not self.model_path:
            return ""

        commands = []

        cli_name = os.path.basename(self.cli_binary)

        if cli_name in ("litert-lm", "litert-lm.exe", "litert"):
            commands.append([
                self.cli_binary,
                "run",
                self.model_path,
                "--prompt",
                prompt,
            ])

        # Official LiteRT-LM runtime binary syntax.
        commands.append([
            self.cli_binary,
            "--backend=cpu",
            f"--model_path={self.model_path}",
            f"--input_prompt={prompt}",
        ])

        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()

                if result.stderr.strip():
                    print(f"[!] LiteRT-LM notice: {result.stderr.strip()[:500]}")

            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                print("[!] LiteRT-LM inference timed out.")
                continue
            except Exception as e:
                print(f"[!] LiteRT-LM execution error: {e}")

        return ""

    def parse_query_with_llm(self, query: str) -> dict:
        prompt = (
            "<start_of_turn>system\n"
            + SYSTEM_PROMPT
            + "\n<end_of_turn>\n"
            + "<start_of_turn>user\n"
            + query
            + "\n<end_of_turn>\n"
            + "<start_of_turn>model\n"
        )

        raw_output = self._run_via_cli(prompt)

        if raw_output:
            # Models sometimes add markdown or explanatory text around JSON.
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                try:
                    cleaned_json = json_match.group(0)
                    cleaned_json = re.sub(r',\s*([}\]])', r'\1', cleaned_json)
                    parsed = json.loads(cleaned_json)

                    if isinstance(parsed, dict):
                        return self._normalize_params(parsed)
                except json.JSONDecodeError:
                    pass

        print("[+] Using offline natural-language parser.")
        return self._heuristic_fallback(query)

    def _normalize_params(self, params: dict) -> dict:
        """Make LLM output safe and predictable for the file-search engine."""
        params = dict(params)

        storage = str(params.get("target_storage", "all")).lower()
        if storage not in ("internal", "external", "all"):
            storage = "all"

        dirs = params.get("directories") or []
        exts = params.get("extensions") or []
        keywords = params.get("keywords") or []

        if isinstance(dirs, str):
            dirs = [dirs]
        if isinstance(exts, str):
            exts = [exts]
        if isinstance(keywords, str):
            keywords = [keywords]

        return {
            "explanation": str(
                params.get("explanation", "Natural-language search")
            ),
            "target_storage": storage,
            "directories": [str(x).strip() for x in dirs if str(x).strip()],
            "extensions": [
                str(x).lower().lstrip(".").strip()
                for x in exts if str(x).strip()
            ],
            "keywords": [
                str(x).lower().strip()
                for x in keywords if str(x).strip()
            ],
            "max_days_old": params.get("max_days_old"),
            "min_days_old": params.get("min_days_old"),
        }

    def _heuristic_fallback(self, query: str) -> dict:
        """
        Offline natural-language parser.

        This is deliberately independent of MediaPipe and works with only
        Python's standard library, which is suitable for Termux.
        """
        q = query.lower().strip()

        target_storage = "all"
        if re.search(r"\b(external|sd\s*card|memory\s*card)\b", q):
            target_storage = "external"
        elif re.search(r"\b(internal|phone|device)\b", q):
            target_storage = "internal"

        # Common Android file categories.
        category_map = {
            "photos": ["jpg", "jpeg", "png", "heic", "webp"],
            "photo": ["jpg", "jpeg", "png", "heic", "webp"],
            "pictures": ["jpg", "jpeg", "png", "heic", "webp"],
            "picture": ["jpg", "jpeg", "png", "heic", "webp"],
            "images": ["jpg", "jpeg", "png", "gif", "webp", "heic"],
            "image": ["jpg", "jpeg", "png", "gif", "webp", "heic"],
            "videos": ["mp4", "mkv", "mov", "avi", "webm", "3gp"],
            "video": ["mp4", "mkv", "mov", "avi", "webm", "3gp"],
            "movies": ["mp4", "mkv", "mov", "avi", "webm"],
            "music": ["mp3", "m4a", "wav", "flac", "ogg", "aac"],
            "songs": ["mp3", "m4a", "wav", "flac", "ogg", "aac"],
            "audio": ["mp3", "m4a", "wav", "flac", "ogg", "aac"],
            "documents": ["pdf", "doc", "docx", "txt", "rtf", "odt"],
            "document": ["pdf", "doc", "docx", "txt", "rtf", "odt"],
            "spreadsheets": ["xls", "xlsx", "csv", "ods"],
            "spreadsheet": ["xls", "xlsx", "csv", "ods"],
            "archives": ["zip", "rar", "7z", "tar", "gz", "bz2"],
            "archive": ["zip", "rar", "7z", "tar", "gz", "bz2"],
            "apps": ["apk", "xapk"],
            "installers": ["apk", "xapk"],
        }

        extensions = set()

        # Explicit extensions: "pdf files", ".jpg", "MP4s", etc.
        explicit_exts = re.findall(
            r"(?<![a-z0-9])\.?(pdf|jpg|jpeg|png|gif|heic|webp|"
            r"mp4|mkv|mov|avi|webm|3gp|mp3|m4a|wav|flac|ogg|aac|"
            r"txt|doc|docx|rtf|odt|xls|xlsx|csv|ods|zip|rar|7z|tar|"
            r"gz|bz2|apk|xapk|py|json|xml|html|css|js)(?![a-z0-9])",
            q
        )
        extensions.update(explicit_exts)

        for word, exts in category_map.items():
            if re.search(r"\b" + re.escape(word) + r"\b", q):
                extensions.update(exts)

        # Folder names mentioned by the user are NOT search restrictions.
        # The entire storage root is always scanned recursively.
        directories = []

        # Natural-language date filters.
        max_days_old = None
        min_days_old = None

        if re.search(r"\b(today|todays)\b", q):
            max_days_old = 1
        elif re.search(r"\byesterday\b", q):
            max_days_old = 2
            min_days_old = 1
        elif re.search(r"\b(this week|past week|last 7 days|recent)\b", q):
            max_days_old = 7
        elif re.search(r"\b(this month|past month|last 30 days)\b", q):
            max_days_old = 30
        elif re.search(r"\blast year\b", q):
            max_days_old = 365

        m = re.search(r"\b(?:last|past)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b", q)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            multiplier = 1
            if unit.startswith("week"):
                multiplier = 7
            elif unit.startswith("month"):
                multiplier = 30
            elif unit.startswith("year"):
                multiplier = 365
            max_days_old = n * multiplier

        # Words that describe the request rather than the filename.
        stopwords = {
            "find", "search", "show", "get", "give", "list", "locate",
            "where", "look", "looking", "for", "me", "my", "all", "any",
            "file", "files", "named", "name", "called", "with", "that",
            "have", "has", "containing", "contains", "inside", "under",
            "from", "in", "on", "at", "the", "a", "an", "of", "to",
            "and", "or", "please", "can", "you", "i", "want", "need",
            "internal", "external", "sd", "card", "storage", "phone",
            "device", "folder", "directory", "today", "todays", "yesterday",
            "recent", "recently", "this", "week", "month", "year", "past",
            "last", "days", "day", "weeks", "months", "years",
            "photos", "photo", "pictures", "picture", "images", "image",
            "videos", "video", "movies", "movie", "music", "songs", "song",
            "audio", "documents", "document", "spreadsheets", "spreadsheet",
            "archives", "archive", "apps", "installers",
            "download", "downloads", "dcim", "podcasts", "audiobooks",
            "pdf", "jpg", "jpeg", "png", "gif", "heic", "webp",
            "mp4", "mkv", "mov", "avi", "webm", "mp3", "m4a", "wav",
            "flac", "ogg", "aac", "txt", "doc", "docx", "rtf", "odt",
            "xls", "xlsx", "csv", "ods", "zip", "rar", "7z", "tar",
            "gz", "bz2", "apk", "xapk", "py", "json", "xml", "html",
            "css", "js",
        }

        words = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_.-]*\b", q)

        keywords = []
        for word in words:
            clean = word.strip("._-")
            if not clean or clean in stopwords:
                continue
            if clean in extensions:
                continue
            if clean.isdigit():
                continue
            if clean not in keywords:
                keywords.append(clean)

        # Don't turn common temporal phrases into filename keywords.
        keywords = [
            w for w in keywords
            if not re.fullmatch(r"(last|past)\d+", w)
        ]

        explanation_parts = []
        if extensions:
            explanation_parts.append(
                "file types: " + ", ".join(sorted(extensions))
            )
        # Directories intentionally omitted: search always starts at storage root.
        if keywords:
            explanation_parts.append(
                "filename words: " + ", ".join(keywords)
            )
        if max_days_old is not None:
            explanation_parts.append(
                f"modified within {max_days_old} day(s)"
            )

        explanation = (
            "Natural-language search"
            + (": " + "; ".join(explanation_parts) if explanation_parts else "")
        )

        return {
            "explanation": explanation,
            "target_storage": target_storage,
            "directories": directories,
            "extensions": sorted(extensions),
            "keywords": keywords,
            "max_days_old": max_days_old,
            "min_days_old": min_days_old,
        }

    def search_files(self, params: dict):
        target_type = params.get("target_storage", "all").lower()
        directories = params.get("directories", [])
        extensions = [e.lower().lstrip('.') for e in params.get("extensions", [])]
        keywords = [k.lower() for k in params.get("keywords", [])]
        
        max_days = params.get("max_days_old")
        min_days = params.get("min_days_old")

        now = datetime.now()
        max_cutoff = now - timedelta(days=int(max_days)) if max_days is not None else None
        min_cutoff = now - timedelta(days=int(min_days)) if min_days is not None else None

        # ALWAYS search recursively from the ROOT of each selected storage.
        # Natural-language queries never restrict the search to specialized
        # folders such as Download, DCIM, Music, etc.
        search_roots = []
        for name, root_path in self.storages.items():
            if target_type == "external" and "external" not in name:
                continue
            if target_type == "internal" and name != "internal":
                continue

            if root_path.exists():
                search_roots.append((name, root_path))

        matches = []
        searched_paths = set()

        for storage_name, search_path in search_roots:
            resolved_path = str(search_path.resolve())
            if resolved_path in searched_paths:
                continue
            searched_paths.add(resolved_path)

            print(f"\n[+] Searching ROOT recursively: [{storage_name.upper()}] {search_path}")
            for root, _, files in os.walk(search_path):
                # Skip Android system and internal app-data directories
                if "Android/data" in root or "Android/obb" in root:
                    continue

                for file in files:
                    file_lower = file.lower()
                    file_path = Path(root) / file

                    # Filter: Extensions
                    if extensions:
                        if not any(file_lower.endswith(f".{ext}") for ext in extensions):
                            continue

                    # Filter: Keywords in Filename
                    if keywords:
                        if not all(kw in file_lower for kw in keywords):
                            continue

                    # Filter: File modification age
                    if max_cutoff or min_cutoff:
                        try:
                            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if max_cutoff and mtime < max_cutoff:
                                continue
                            if min_cutoff and mtime > min_cutoff:
                                continue
                        except OSError:
                            continue

                    matches.append(file_path)

        return matches


def execute_search_job(searcher: LiteRTFileSearch, query: str):
    print(f"\n[?] Query: \"{query}\"")
    params = searcher.parse_query_with_llm(query)
    
    explanation = params.get("explanation", "Parsed search filters")
    print(f"\n🤖 AI Analysis: \"{explanation}\"")
    print(f"[+] Filters: {json.dumps(params, indent=2)}")
    
    results = searcher.search_files(params)

    print(f"\n--- Found {len(results)} matching file(s) ---")
    for filepath in results[:50]:
        print(f"  └─ {filepath}")

    if len(results) > 50:
        print(f"  ... and {len(results) - 50} more.")


def main():
    model_file = Path(DEFAULT_MODEL_PATH)
    if not model_file.exists():
        print(f"[!] Info: Model not found at '{DEFAULT_MODEL_PATH}'. Running on smart fallback search engine.")

    searcher = LiteRTFileSearch(model_path=str(model_file))

    # 1. Direct one-liner execution (e.g. python search.py find all files with name ibiza)
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        execute_search_job(searcher, query)
        return

    # 2. Interactive shell mode
    print("\n" + "=" * 55)
    print("      LiteRT-LM Natural Language File Assistant")
    print("      (Type 'exit', 'quit', or 'q' to stop)")
    print("=" * 55)

    while True:
        try:
            query = input("\nSearch > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting search.")
                break
            execute_search_job(searcher, query)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting search.")
            break


if __name__ == "__main__":
    main()

