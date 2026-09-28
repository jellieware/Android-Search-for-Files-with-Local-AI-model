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

# Maximum number of matching files to PRINT.
# None = unlimited (print every matching result).
# Set to an integer such as 100, 500, or 1000 to limit printed results.
MAX_PRINT_RESULTS = None

# Highlight matched filename keywords in red when printing results.
# Set to False to disable ANSI color output.
HIGHLIGHT_KEYWORDS_RED = True
RED = "\033[31m"
RESET_COLOR = "\033[0m"


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
  "directories": [],
  "extensions": ["ext1", "ext2"],
  "keywords": ["word1", "word2"],
  "max_days_old": integer or null,
  "min_days_old": integer or null
}

IMPORTANT: Directory searching is NOT supported. Always return "directories": [] and NEVER use folder names such as Download, DCIM, Music, Pictures, Documents, Movies, etc. as search filters. File categories must be represented ONLY by extensions.

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
        """
        Parse the user's request with the local LiteRT-LM model, then apply
        a deterministic local safety/type layer before searching.

        LiteRT-LM is used for natural-language intent (for example:
          "pictures of my dog from last week" -> keyword="dog" + recent date)

        The local extension/category map is authoritative for file types.
        This prevents the model from turning "photos" into an unrestricted
        search or inventing directory filters.
        """
        q = query.lower().strip()

        # First obtain the local deterministic interpretation.  This is also
        # the safety net if LiteRT-LM is unavailable or returns bad JSON.
        local = self._heuristic_fallback(query)

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
        model_params = None

        if raw_output:
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                try:
                    cleaned_json = re.sub(
                        r',\s*([}\]])', r'\1', json_match.group(0)
                    )
                    parsed = json.loads(cleaned_json)
                    if isinstance(parsed, dict):
                        model_params = self._normalize_params(parsed)
                except (json.JSONDecodeError, TypeError, ValueError):
                    model_params = None

        if model_params is None:
            print("[+] Using offline local natural-language parser.")
            return local

        # ------------------------------------------------------------------
        # Merge LOCAL AI intent with the deterministic file-type layer.
        # LiteRT-LM remains responsible for natural-language understanding,
        # while the local extension map decides what files a category means.
        # ------------------------------------------------------------------
        result = dict(local)
        result["directories"] = []

        # Preserve useful semantic information from the local model for
        # storage/date intent, but never allow model directories.
        result["target_storage"] = model_params.get(
            "target_storage", local.get("target_storage", "all")
        )

        if model_params.get("max_days_old") is not None:
            result["max_days_old"] = model_params["max_days_old"]
        if model_params.get("min_days_old") is not None:
            result["min_days_old"] = model_params["min_days_old"]

        # Filename-only searches are special: they intentionally search every
        # extension.  "name cook" must NOT become a photo/document search.
        filename_only = self._is_filename_only_query(query)

        if filename_only:
            result["extensions"] = []
            result["keywords"] = self._extract_filename_terms(query)
            result["explanation"] = (
                "Local AI natural-language search: all file types; "
                "filename words: " + ", ".join(result["keywords"])
                if result["keywords"]
                else "Local AI natural-language search: all file types"
            )
            return result

        # A category/type query MUST retain the complete local extension list.
        # Never replace it with model output.
        if local.get("extensions"):
            result["extensions"] = list(local["extensions"])

        # Use model keywords only when they are genuine filename terms.
        # Remove category/type vocabulary so "find photos" cannot become
        # keywords=["photos"], and do not combine every word in the sentence.
        local_keywords = set(local.get("keywords", []))
        model_keywords = []
        for word in model_params.get("keywords", []):
            word = str(word).lower().strip(" ._-")
            if not word or word in local_keywords:
                continue
            if word in {"photos", "photo", "pictures", "picture", "images",
                        "image", "videos", "video", "movies", "movie",
                        "audio", "music", "songs", "song", "documents",
                        "document", "docs", "doc", "archives", "archive",
                        "apps", "app", "files", "file"}:
                continue
            model_keywords.append(word)

        # The local parser is conservative.  If it found a meaningful filename
        # term, keep it. Otherwise accept only model-provided semantic keywords.
        if local_keywords:
            result["keywords"] = list(local.get("keywords", []))
        else:
            result["keywords"] = list(dict.fromkeys(model_keywords))

        result["directories"] = []
        result["explanation"] = (
            "Local AI natural-language search"
            + (": file types: " + ", ".join(result["extensions"])
               if result.get("extensions") else ": all file types")
            + ("; filename words: " + ", ".join(result["keywords"])
               if result.get("keywords") else "")
        )

        return result

    @staticmethod
    def _extract_filename_terms(query: str) -> list:
        """Extract only the actual filename/title terms from a filename query."""
        words = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_.-]*\b", query.lower())
        stop = {
            "all", "every", "file", "files", "with", "the", "name", "title",
            "named", "called", "titled", "find", "search", "for", "show",
            "get", "list", "me", "please", "by", "of"
        }
        return list(dict.fromkeys(
            w for w in words
            if w not in stop and not w.isdigit()
        ))

    @staticmethod
    def _is_filename_only_query(query: str) -> bool:
        """Return True when the user asks to search by filename/title only.

        A filename/title-only search intentionally has NO extension restriction,
        so it searches every known and unknown file extension.
        """
        q = query.lower().strip()

        # Examples:
        #   all files with name cook
        #   find all files with title cook
        #   files with title "cook"
        #   find all files named cook
        #   search for files called cook
        #   files titled cook
        file_phrase = r"\bfiles?\b"
        name_word = r"\b(?:name|title)\b"
        named_word = r"\b(?:named|called|titled)\b"

        # IMPORTANT: only treat the request as filename-only when the word
        # "file(s)" is explicitly part of the filename phrase.  A query such
        # as "find photos named cook" or "find videos with name movie" is a
        # CATEGORY + FILENAME search and must keep the photo/video extensions.
        return (
            bool(re.search(r"\b(?:all|every)\s+files?\b", q))
            and bool(re.search(name_word, q))
        ) or bool(
            re.search(file_phrase + r"\s+(?:with\s+)?(?:the\s+)?" + name_word, q)
        ) or bool(
            re.search(file_phrase + r"\s+" + named_word, q)
        )

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

        # A filename/title-only search (e.g. "all files with name cook" or "find all files with title cook")
        # searches every extension. It is intentionally NOT an extension search.
        filename_only_search = self._is_filename_only_query(query)

        # An empty quoted filename means "do not filter by filename".
        # This also handles queries such as:
        #   all files with name ""
        #   files named ""
        #   find all files with name ''
        #   search for all files
        empty_name_search = bool(
            re.search(r'\\\\bname\\s*(?:=|is|called|named)?\\s*(?:""|\'\'\\\\\'\\\\\')', q)
            or re.search(r'\\\\b(?:all|every)\\s+files?\\\\b', q)
        )

        target_storage = "all"
        if re.search(r"\b(external|sd\s*card|memory\s*card)\b", q):
            target_storage = "external"
        elif re.search(r"\b(internal|phone|device)\b", q):
            target_storage = "internal"

        # Common Android file categories.
        category_map = {
            # Images / photos
            "photos": ["jpg", "jpeg", "jpe", "jfif", "pjpeg", "pjp", "png",
                       "gif", "bmp", "dib", "webp", "heic", "heif", "avif",
                       "tif", "tiff", "raw", "dng", "cr2", "cr3", "nef",
                       "nrw", "arw", "orf", "rw2", "raf", "srw", "pef",
                       "ico", "svg", "svgz"],
            "photo": ["jpg", "jpeg", "jpe", "jfif", "pjpeg", "pjp", "png",
                      "gif", "bmp", "dib", "webp", "heic", "heif", "avif",
                      "tif", "tiff", "raw", "dng", "cr2", "cr3", "nef",
                      "nrw", "arw", "orf", "rw2", "raf", "srw", "pef",
                      "ico", "svg", "svgz"],
            "pictures": ["jpg", "jpeg", "jpe", "jfif", "pjpeg", "pjp", "png",
                         "gif", "bmp", "dib", "webp", "heic", "heif", "avif",
                         "tif", "tiff", "raw", "dng", "cr2", "cr3", "nef",
                         "nrw", "arw", "orf", "rw2", "raf", "srw", "pef",
                         "ico", "svg", "svgz"],
            "picture": ["jpg", "jpeg", "jpe", "jfif", "pjpeg", "pjp", "png",
                        "gif", "bmp", "dib", "webp", "heic", "heif", "avif",
                        "tif", "tiff", "raw", "dng", "cr2", "cr3", "nef",
                        "nrw", "arw", "orf", "rw2", "raf", "srw", "pef",
                        "ico", "svg", "svgz"],
            "images": ["jpg", "jpeg", "jpe", "jfif", "pjpeg", "pjp", "png",
                       "gif", "bmp", "dib", "webp", "heic", "heif", "avif",
                       "tif", "tiff", "raw", "dng", "cr2", "cr3", "nef",
                       "nrw", "arw", "orf", "rw2", "raf", "srw", "pef",
                       "ico", "svg", "svgz"],
            "image": ["jpg", "jpeg", "jpe", "jfif", "pjpeg", "pjp", "png",
                      "gif", "bmp", "dib", "webp", "heic", "heif", "avif",
                      "tif", "tiff", "raw", "dng", "cr2", "cr3", "nef",
                      "nrw", "arw", "orf", "rw2", "raf", "srw", "pef",
                      "ico", "svg", "svgz"],

            # Video
            "videos": ["mp4", "m4v", "mkv", "mov", "qt", "avi", "wmv",
                       "asf", "webm", "flv", "f4v", "3gp", "3g2", "mpeg",
                       "mpg", "mpe", "m1v", "m2v", "mts", "m2ts", "ts",
                       "vob", "ogv", "rm", "rmvb", "divx"],
            "video": ["mp4", "m4v", "mkv", "mov", "qt", "avi", "wmv",
                      "asf", "webm", "flv", "f4v", "3gp", "3g2", "mpeg",
                      "mpg", "mpe", "m1v", "m2v", "mts", "m2ts", "ts",
                      "vob", "ogv", "rm", "rmvb", "divx"],
            "movies": ["mp4", "m4v", "mkv", "mov", "qt", "avi", "wmv",
                       "asf", "webm", "flv", "f4v", "mpeg", "mpg", "mts",
                       "m2ts", "vob", "ogv", "3gp", "3g2"],
            "movie": ["mp4", "m4v", "mkv", "mov", "qt", "avi", "wmv",
                      "asf", "webm", "flv", "f4v", "mpeg", "mpg", "mts",
                      "m2ts", "vob", "ogv", "3gp", "3g2"],

            # Audio / music / recordings
            "music": ["mp3", "m4a", "m4b", "m4p", "aac", "wav", "wave",
                      "flac", "ogg", "oga", "opus", "wma", "aiff", "aif",
                      "aifc", "alac", "ape", "amr", "3ga", "mid", "midi",
                      "mka", "ac3", "dts", "ra", "ram"],
            "songs": ["mp3", "m4a", "m4b", "m4p", "aac", "wav", "wave",
                      "flac", "ogg", "oga", "opus", "wma", "aiff", "aif",
                      "aifc", "alac", "ape", "amr", "3ga", "mid", "midi",
                      "mka", "ac3", "dts", "ra", "ram"],
            "song": ["mp3", "m4a", "m4b", "m4p", "aac", "wav", "wave",
                     "flac", "ogg", "oga", "opus", "wma", "aiff", "aif",
                     "aifc", "alac", "ape", "amr", "3ga", "mid", "midi",
                     "mka", "ac3", "dts", "ra", "ram"],
            "audio": ["mp3", "m4a", "m4b", "m4p", "aac", "wav", "wave",
                      "flac", "ogg", "oga", "opus", "wma", "aiff", "aif",
                      "aifc", "alac", "ape", "amr", "3ga", "mid", "midi",
                      "mka", "ac3", "dts", "ra", "ram"],
            "recordings": ["mp3", "m4a", "aac", "wav", "flac", "ogg", "opus",
                           "amr", "3ga", "mka"],

            # Documents / text
            "documents": ["pdf", "doc", "docx", "docm", "dot", "dotx", "dotm",
                          "txt", "text", "rtf", "odt", "ott", "pages", "md",
                          "markdown", "tex", "latex", "epub", "mobi", "azw",
                          "azw3", "fb2", "djvu", "xps"],
            "document": ["pdf", "doc", "docx", "docm", "dot", "dotx", "dotm",
                         "txt", "text", "rtf", "odt", "ott", "pages", "md",
                         "markdown", "tex", "latex", "epub", "mobi", "azw",
                         "azw3", "fb2", "djvu", "xps"],
            "text": ["txt", "text", "log", "md", "markdown", "rst", "rtf",
                     "csv", "tsv", "tex", "latex"],

            # Spreadsheets / tabular data
            "spreadsheets": ["xls", "xlsx", "xlsm", "xlsb", "xlt", "xltx",
                             "xltm", "csv", "tsv", "ods", "ots", "numbers"],
            "spreadsheet": ["xls", "xlsx", "xlsm", "xlsb", "xlt", "xltx",
                            "xltm", "csv", "tsv", "ods", "ots", "numbers"],
            "tables": ["csv", "tsv", "xls", "xlsx", "xlsm", "xlsb", "ods"],

            # Presentations
            "presentations": ["ppt", "pptx", "pptm", "pps", "ppsx", "ppsm",
                              "pot", "potx", "potm", "odp", "otp", "key"],
            "presentation": ["ppt", "pptx", "pptm", "pps", "ppsx", "ppsm",
                             "pot", "potx", "potm", "odp", "otp", "key"],
            "slides": ["ppt", "pptx", "pptm", "pps", "ppsx", "ppsm",
                       "odp", "otp", "key"],

            # Archives / compressed files
            "archives": ["zip", "zipx", "rar", "7z", "tar", "gz", "tgz",
                         "bz", "bz2", "tbz", "tbz2", "xz", "txz", "z",
                         "lz", "lz4", "lzh", "cab", "arj", "ace", "iso",
                         "img", "dmg"],
            "archive": ["zip", "zipx", "rar", "7z", "tar", "gz", "tgz",
                        "bz", "bz2", "tbz", "tbz2", "xz", "txz", "z",
                        "lz", "lz4", "lzh", "cab", "arj", "ace", "iso",
                        "img", "dmg"],
            "compressed": ["zip", "zipx", "rar", "7z", "tar", "gz", "tgz",
                           "bz2", "xz", "lz", "lz4", "cab", "iso"],

            # Android apps / packages
            "apps": ["apk", "xapk", "apks", "apkx", "aab"],
            "app": ["apk", "xapk", "apks", "apkx", "aab"],
            "installers": ["apk", "xapk", "apks", "apkx", "aab", "exe",
                           "msi", "deb", "rpm", "dmg", "pkg"],
            "packages": ["apk", "xapk", "apks", "aab", "deb", "rpm", "pkg"],

            # Programming / source code
            "code": ["py", "pyw", "js", "jsx", "ts", "tsx", "java", "kt",
                     "kts", "scala", "groovy", "c", "h", "cpp", "cxx", "cc",
                     "hpp", "cs", "swift", "m", "mm", "go", "rs", "rb",
                     "php", "pl", "pm", "lua", "r", "dart", "ex", "exs",
                     "erl", "hrl", "fs", "fsx", "vb", "vbs", "sh", "bash",
                     "zsh", "fish", "ps1", "bat", "cmd", "sql", "asm",
                     "s", "sol", "clj", "cljs", "hs", "lhs", "jl"],
            "programming": ["py", "pyw", "js", "jsx", "ts", "tsx", "java",
                            "kt", "kts", "c", "h", "cpp", "cxx", "cc", "hpp",
                            "cs", "swift", "go", "rs", "rb", "php", "pl",
                            "lua", "r", "dart", "ex", "exs", "erl", "fs",
                            "fsx", "vb", "sh", "bash", "zsh", "ps1", "bat",
                            "cmd", "sql", "asm", "sol", "clj", "hs", "jl"],
            "scripts": ["py", "pyw", "js", "ts", "sh", "bash", "zsh", "fish",
                        "ps1", "bat", "cmd", "vbs", "pl", "rb", "lua"],

            # Web
            "web": ["html", "htm", "xhtml", "css", "scss", "sass", "less",
                    "js", "jsx", "ts", "tsx", "json", "xml", "svg", "wasm"],
            "website": ["html", "htm", "xhtml", "css", "scss", "sass", "less",
                        "js", "jsx", "ts", "tsx", "json", "xml", "svg"],
            "websites": ["html", "htm", "xhtml", "css", "scss", "sass", "less",
                         "js", "jsx", "ts", "tsx", "json", "xml", "svg"],

            # Data / configuration
            "data": ["json", "jsonl", "xml", "yaml", "yml", "toml", "ini",
                     "cfg", "conf", "config", "csv", "tsv", "db", "sqlite",
                     "sqlite3", "sql", "bak", "dat"],
            "database": ["db", "sqlite", "sqlite3", "db3", "mdb", "accdb",
                         "sql", "dump", "bak"],
            "databases": ["db", "sqlite", "sqlite3", "db3", "mdb", "accdb",
                          "sql", "dump", "bak"],
            "config": ["json", "xml", "yaml", "yml", "toml", "ini", "cfg",
                       "conf", "config", "properties", "plist", "env"],

            # Fonts
            "fonts": ["ttf", "otf", "woff", "woff2", "eot", "fon"],
            "font": ["ttf", "otf", "woff", "woff2", "eot", "fon"],

            # Subtitles / captions
            "subtitles": ["srt", "vtt", "ass", "ssa", "sub", "idx", "sup"],
            "subtitle": ["srt", "vtt", "ass", "ssa", "sub", "idx", "sup"],
            "captions": ["srt", "vtt", "ass", "ssa", "sub"],

            # Email
            "email": ["eml", "msg", "emlx", "mbox", "pst", "ost"],
            "emails": ["eml", "msg", "emlx", "mbox", "pst", "ost"],

            # Certificates / keys
            "certificates": ["pem", "crt", "cer", "der", "p7b", "p7c", "p12",
                             "pfx"],
            "certificates": ["pem", "crt", "cer", "der", "p7b", "p7c", "p12",
                             "pfx"],
            "keys": ["key", "pem", "pub", "ppk", "asc"],

            # 3D / CAD
            "3d": ["obj", "fbx", "stl", "dae", "gltf", "glb", "3ds", "blend",
                   "ply", "step", "stp", "iges", "igs"],
            "cad": ["dwg", "dxf", "step", "stp", "iges", "igs", "dgn"],
        }

        extensions = set()

        # Explicit extensions: "pdf files", ".jpg", "MP4s", etc.
        explicit_exts = re.findall(
            r"(?<![a-z0-9])\.?(pdf|doc|docx|docm|dot|dotx|dotm|txt|text|rtf|"
            r"odt|ott|pages|md|markdown|tex|epub|mobi|azw|azw3|fb2|djvu|xps|"
            r"xls|xlsx|xlsm|xlsb|xlt|xltx|xltm|csv|tsv|ods|ots|numbers|"
            r"ppt|pptx|pptm|pps|ppsx|ppsm|pot|potx|potm|odp|otp|key|"
            r"jpg|jpeg|jpe|jfif|pjpeg|pjp|png|gif|bmp|dib|webp|heic|heif|"
            r"avif|tif|tiff|raw|dng|cr2|cr3|nef|nrw|arw|orf|rw2|raf|srw|pef|"
            r"ico|svg|svgz|mp4|m4v|mkv|mov|qt|avi|wmv|asf|webm|flv|f4v|"
            r"3gp|3g2|mpeg|mpg|mpe|m1v|m2v|mts|m2ts|ts|vob|ogv|rm|rmvb|divx|"
            r"mp3|m4a|m4b|m4p|aac|wav|wave|flac|ogg|oga|opus|wma|aiff|aif|"
            r"aifc|alac|ape|amr|3ga|mid|midi|mka|ac3|dts|ra|ram|"
            r"zip|zipx|rar|7z|tar|gz|tgz|bz|bz2|tbz|tbz2|xz|txz|z|lz|lz4|"
            r"lzh|cab|arj|ace|iso|img|dmg|"
            r"apk|xapk|apks|apkx|aab|exe|msi|deb|rpm|pkg|"
            r"py|pyw|js|jsx|ts|tsx|java|kt|kts|scala|groovy|c|h|cpp|cxx|cc|"
            r"hpp|cs|swift|m|mm|go|rs|rb|php|pl|pm|lua|r|dart|ex|exs|erl|"
            r"hrl|fs|fsx|vb|vbs|sh|bash|zsh|fish|ps1|bat|cmd|sql|asm|s|"
            r"sol|clj|cljs|hs|lhs|jl|"
            r"html|htm|xhtml|css|scss|sass|less|wasm|json|jsonl|xml|yaml|"
            r"yml|toml|ini|cfg|conf|config|properties|plist|env|db|sqlite|"
            r"sqlite3|db3|mdb|accdb|dump|bak|dat|"
            r"ttf|otf|woff|woff2|eot|fon|"
            r"srt|vtt|ass|ssa|sub|idx|sup|"
            r"eml|msg|emlx|mbox|pst|ost|"
            r"pem|crt|cer|der|p7b|p7c|p12|pfx|pub|ppk|asc|"
            r"obj|fbx|stl|dae|gltf|glb|3ds|blend|ply|step|stp|iges|igs|"
            r"dwg|dxf|dgn)s?(?![a-z0-9])",
            q
        )
        extensions.update(explicit_exts)

        # Common user shorthand for category names.
        category_query = q
        category_query = re.sub(r"\bdocs\b", "documents", category_query)
        category_query = re.sub(r"\bdoc\b", "document", category_query)
        category_query = re.sub(r"\bpics\b", "photos", category_query)
        category_query = re.sub(r"\bpictures?\b", "photos", category_query)
        category_query = re.sub(r"\bmovies?\b", "movies", category_query)
        category_query = re.sub(r"\bsounds?\b", "audio", category_query)
        category_query = re.sub(r"\btracks?\b", "music", category_query)

        for word, exts in category_map.items():
            if re.search(r"\b" + re.escape(word) + r"\b", category_query):
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
            "file", "files", "named", "name", "called", "title", "titled", "with", "that",
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
            "css", "js", "jpe", "jfif", "pjpeg", "pjp", "dib", "avif",
            "tif", "tiff", "raw", "dng", "cr2", "cr3", "nef", "nrw", "arw",
            "orf", "rw2", "raf", "srw", "pef", "ico", "svg", "svgz", "m4v",
            "qt", "wmv", "asf", "flv", "f4v", "3gp", "3g2", "mpeg", "mpg",
            "mpe", "m1v", "m2v", "mts", "m2ts", "ts", "vob", "ogv", "rm",
            "rmvb", "divx", "m4b", "m4p", "wave", "oga", "opus", "wma",
            "aiff", "aif", "aifc", "alac", "ape", "amr", "3ga", "mid", "midi",
            "mka", "ac3", "dts", "ra", "ram", "docm", "dot", "dotx", "dotm",
            "ott", "pages", "md", "markdown", "tex", "latex", "epub", "mobi",
            "azw", "azw3", "fb2", "djvu", "xps", "xlsm", "xlsb", "xlt",
            "xltx", "xltm", "ots", "numbers", "ppt", "pptm", "pps", "ppsx",
            "ppsm", "pot", "potx", "potm", "odp", "otp", "key", "zipx", "tgz",
            "bz", "tbz", "tbz2", "xz", "txz", "z", "lz", "lz4", "lzh", "cab",
            "arj", "ace", "iso", "img", "dmg", "apks", "apkx", "aab", "exe",
            "msi", "deb", "rpm", "pkg", "pyw", "jsx", "tsx", "java", "kt",
            "kts", "scala", "groovy", "cpp", "cxx", "cc", "hpp", "cs", "swift",
            "mm", "go", "rs", "rb", "php", "pl", "pm", "lua", "dart", "ex",
            "exs", "erl", "hrl", "fs", "fsx", "vb", "vbs", "sh", "bash", "zsh",
            "fish", "ps1", "bat", "cmd", "asm", "sol", "clj", "cljs", "hs",
            "lhs", "jl", "html", "htm", "xhtml", "scss", "sass", "less", "wasm",
            "json", "jsonl", "yaml", "yml", "toml", "ini", "cfg", "conf",
            "config", "properties", "plist", "env", "db", "sqlite", "sqlite3",
            "db3", "mdb", "accdb", "dump", "bak", "dat", "ttf", "otf", "woff",
            "woff2", "eot", "fon", "srt", "vtt", "ass", "ssa", "sub", "idx",
            "sup", "eml", "msg", "emlx", "mbox", "pst", "ost", "pem", "crt",
            "cer", "der", "p7b", "p7c", "p12", "pfx", "pub", "ppk", "asc",
            "obj", "fbx", "stl", "dae", "gltf", "glb", "3ds", "blend", "ply",
            "step", "stp", "iges", "igs", "dwg", "dxf", "dgn", "presentations",
            "presentation", "slides", "tables", "text", "compressed", "app",
            "packages", "code", "programming", "scripts", "web", "website",
            "websites", "data", "database", "databases", "config", "fonts",
            "font", "subtitles", "subtitle", "captions", "email", "emails",
            "certificates", "keys", "3d", "cad",
        }

        words = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_.-]*\b", q)

        keywords = []
        for word in words:
            clean = word.strip("._-")
            if not clean or clean in stopwords:
                continue
            # Never treat file-type/category words as filename keywords.
            # This includes pluralized explicit extensions such as "pdfs" and
            # shorthand category terms such as "docs", "pics", and "sounds".
            category_words = set(category_map.keys()) | {
                "docs", "pics", "sounds", "tracks"
            }
            if clean in extensions or clean.rstrip("s") in extensions:
                continue
            if clean in category_words:
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

        # For an explicit empty filename / "all files" request, there is
        # intentionally no filename or extension restriction.
        if empty_name_search:
            keywords = []
            extensions = set()
        elif filename_only_search:
            # Keep the filename term(s), but remove every type/category restriction.
            extensions = set()

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

        if filename_only_search and keywords:
            explanation_parts.insert(0, "all file types")

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
        # Ignore any directory filters from any caller or LLM output.
        params = dict(params)
        params["directories"] = []

        target_type = params.get("target_storage", "all").lower()

        # Directory filtering is intentionally disabled. The search engine
        # ONLY filters by file type (extension), filename keywords, and date.
        # It always scans recursively from each selected storage root.
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
                    # File-type searches are ALWAYS based on the file's actual
                    # extension. Directory names are never considered here.
                    if extensions:
                        file_ext = file_path.suffix.lower().lstrip(".")
                        if file_ext not in extensions:
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

    # Print every result by default. Set MAX_PRINT_RESULTS to an integer
    # above if you want to limit console output.
    if MAX_PRINT_RESULTS is None:
        results_to_print = results
    else:
        results_to_print = results[:max(0, int(MAX_PRINT_RESULTS))]

    def highlight_keywords(filepath):
        """Highlight each filename-search keyword in red in the printed path."""
        text = str(filepath)

        if not HIGHLIGHT_KEYWORDS_RED:
            return text

        keywords = params.get("keywords", []) or []
        # Longest first prevents a shorter keyword from consuming part of a
        # longer keyword before it can be highlighted.
        keywords = sorted(
            {str(k) for k in keywords if str(k)},
            key=len,
            reverse=True,
        )

        if not keywords:
            return text

        pattern = re.compile(
            "|".join(re.escape(keyword) for keyword in keywords),
            re.IGNORECASE,
        )

        return pattern.sub(
            lambda match: RED + match.group(0) + RESET_COLOR,
            text,
        )

    for filepath in results_to_print:
        print(f"  └─ {highlight_keywords(filepath)}")

    if MAX_PRINT_RESULTS is not None and len(results) > len(results_to_print):
        print(f"  ... and {len(results) - len(results_to_print)} more.")


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

