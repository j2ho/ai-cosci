"""Tool implementations for the bioinformatics agent."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional
import requests
import sys
from io import StringIO
import signal
from contextlib import contextmanager
import os

# CRITICAL: Set environment variables at module level BEFORE any PaperQA imports
# LiteLLM reads these at import time, not runtime!
from dotenv import load_dotenv
load_dotenv()  # Load .env file to get OPENROUTER_KEY

# Ensure OPENROUTER_KEY is available for PaperQA/LiteLLM
# Note: LiteLLM expects OPENROUTER_KEY (not OPENROUTER_API_KEY) based on constants.py
if not os.getenv("OPENROUTER_KEY"):
    print("WARNING: OPENROUTER_KEY not set - PaperQA will fail", file=sys.stderr)
else:
    print(f"✓ OPENROUTER_KEY loaded: {os.getenv('OPENROUTER_KEY')[:15]}...", file=sys.stderr)


class ToolResult:
    """Result wrapper for tool execution."""

    def __init__(self, success: bool, output: Any, error: Optional[str] = None):
        self.success = success
        self.output = output
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }


class PersistentPythonExecutor:
    """Maintains a persistent Python environment across execute_python calls.

    This allows variables, imports, and state to persist between calls,
    enabling multi-step data analysis without reloading data.
    """

    def __init__(self):
        """Initialize the persistent Python environment."""
        self.globals_dict = {
            '__builtins__': __builtins__,
        }
        self.locals_dict = {}

    def execute(self, code: str, timeout: int = 30) -> tuple[bool, Optional[str], Optional[str]]:
        """Execute code in the persistent environment.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Tuple of (success, output, error)
        """
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            # Execute code in persistent namespace
            exec(code, self.globals_dict, self.locals_dict)

            # Get output
            output = sys.stdout.getvalue()
            error = sys.stderr.getvalue()

            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            if error:
                return False, None, error

            return True, output.strip() if output else "Code executed successfully (no output)", None

        except Exception as e:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            return False, None, f"Execution error: {type(e).__name__}: {str(e)}"

    def reset(self):
        """Reset the persistent environment (clears all variables)."""
        self.globals_dict = {
            '__builtins__': __builtins__,
        }
        self.locals_dict = {}


# Global persistent executor instance
_persistent_executor = PersistentPythonExecutor()


def execute_python(code: str, timeout: int = 30, reset: bool = False) -> ToolResult:
    """Execute Python code in a persistent environment.

    Variables, imports, and state persist across calls, allowing multi-step
    analyses without reloading data.

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds (not strictly enforced in persistent mode)
        reset: If True, reset the persistent environment before execution

    Returns:
        ToolResult with output or error
    """
    global _persistent_executor

    if reset:
        _persistent_executor.reset()

    try:
        success, output, error = _persistent_executor.execute(code, timeout)

        if not success:
            return ToolResult(False, None, error)

        return ToolResult(True, output)

    except Exception as e:
        return ToolResult(False, None, f"Error executing code: {str(e)}")


def search_pubmed(query: str, max_results: int = 10, retmax: int = 100) -> ToolResult:
    """Search PubMed for articles.

    Args:
        query: Search query string
        max_results: Maximum results to return
        retmax: Maximum results to fetch from NCBI

    Returns:
        ToolResult with list of articles
    """
    import xml.etree.ElementTree as ET
    
    try:
        # NCBI E-utilities endpoints
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        # Search for PMIDs
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
        }

        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_response.raise_for_status()
        search_data = search_response.json()

        pmids = search_data.get("esearchresult", {}).get("idlist", [])[:max_results]

        if not pmids:
            return ToolResult(True, [])

        # Fetch full article details using efetch (includes abstracts)
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }

        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=15)
        fetch_response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(fetch_response.content)

        articles = []
        for article in root.findall(".//PubmedArticle"):
            # Extract PMID
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "N/A"
            
            # Extract title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "N/A"
            
            # Extract abstract (combine all AbstractText elements)
            abstract_parts = []
            for abstract_text in article.findall(".//AbstractText"):
                # Check for labeled sections (e.g., BACKGROUND, METHODS)
                label = abstract_text.get("Label", "")
                text = abstract_text.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts) if abstract_parts else "N/A"
            
            # Extract authors (first 3)
            authors = []
            for author in article.findall(".//Author")[:3]:
                last_name = author.find(".//LastName")
                initials = author.find(".//Initials")
                if last_name is not None:
                    author_name = last_name.text
                    if initials is not None:
                        author_name += f" {initials.text}"
                    authors.append(author_name)
            
            # Extract publication date
            pub_date = article.find(".//PubDate")
            date_str = "N/A"
            if pub_date is not None:
                year = pub_date.find("Year")
                month = pub_date.find("Month")
                if year is not None:
                    date_str = year.text
                    if month is not None:
                        date_str = f"{year.text} {month.text}"

            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "pubdate": date_str,
            })

        return ToolResult(True, articles)

    except Exception as e:
        return ToolResult(False, None, f"PubMed search error: {str(e)}")


def _chunked_search(file_path: str, sep: str, column: str, search_value: str, limit: int = 10, chunk_size: int = 10000, max_chunks: int = 50) -> tuple[list, int]:
    """Search a large file in chunks until enough results are found.

    Args:
        file_path: Path to the file
        sep: Delimiter
        column: Column to search in
        search_value: Value to search for
        limit: Maximum results to return
        chunk_size: Rows per chunk
        max_chunks: Maximum chunks to search (safety limit)

    Returns:
        Tuple of (results_list, total_rows_searched)
    """
    import pandas as pd

    results = []
    total_searched = 0

    for chunk in pd.read_csv(file_path, sep=sep, chunksize=chunk_size, low_memory=False):
        total_searched += len(chunk)

        # Search this chunk
        matches = chunk[chunk[column].astype(str).str.contains(search_value, case=False, na=False)]
        results.extend(matches.to_dict('records'))

        # Stop if we have enough results
        if len(results) >= limit:
            return results[:limit], total_searched

        # Safety limit
        if total_searched >= chunk_size * max_chunks:
            break

    return results, total_searched


def query_database(db_name: str, query: str, limit: int = 10, data_dir: str = "/home.galaxy4/sumin/project/aisci/Competition_Data") -> ToolResult:
    """Query a local database file.

    Args:
        db_name: Database name (e.g., 'drugbank', 'bindingdb', 'pharos', 'string', 'gwas')
        query: Query string - can be:
               - Column name to search (e.g., 'Target Name:EGFR')
               - 'info' to get database info (shape, columns, sample)
               - 'all' to get first N rows
        limit: Maximum rows to return (default: 10)
        data_dir: Path to database files

    Returns:
        ToolResult with query results. For searches, includes 'rows_searched' to show coverage.
    """
    try:
        import pandas as pd

        db_name_lower = db_name.lower()
        data_path = Path(data_dir)

        # Handle different databases based on their actual file structure
        if db_name_lower == "bindingdb":
            file_path = data_path / "Drug" / "BindingDB" / "BindingDB_All.tsv"
            if not file_path.exists():
                return ToolResult(False, None, f"BindingDB file not found at {file_path}")

            # For large files, read with chunksize or nrows
            if query.lower() == "info":
                df_sample = pd.read_csv(file_path, sep="\t", nrows=5, low_memory=False)
                return ToolResult(True, {
                    "database": "BindingDB",
                    "file": str(file_path),
                    "columns": df_sample.columns.tolist(),
                    "sample": df_sample.to_dict('records')
                })
            elif ":" in query:
                # Column-based search: "Target Name:EGFR"
                col, value = query.split(":", 1)
                # Use chunked search to iteratively search through file
                results, rows_searched = _chunked_search(file_path, "\t", col, value, limit=limit)
                return ToolResult(True, {
                    "count": len(results),
                    "rows_searched": rows_searched,
                    "results": results,
                    "message": f"Searched {rows_searched:,} rows, found {len(results)} matches"
                })
            else:
                df = pd.read_csv(file_path, sep="\t", nrows=limit, low_memory=False)
                return ToolResult(True, df.to_dict('records'))

        elif db_name_lower == "drugbank":
            drugbank_path = data_path / "Drug" / "DrugBank"
            if not drugbank_path.exists():
                return ToolResult(False, None, f"DrugBank directory not found at {drugbank_path}")

            # List available files
            if query.lower() == "info":
                files = [f.name for f in drugbank_path.glob("*.parquet")]
                return ToolResult(True, {
                    "database": "DrugBank",
                    "available_files": files,
                    "message": "Use query like 'file:interactions' to query specific file"
                })
            elif query.lower().startswith("file:"):
                # Query specific file: "file:interactions"
                file_name = query.split(":", 1)[1].strip()
                file_path = drugbank_path / f"{file_name}.parquet"
                if not file_path.exists():
                    return ToolResult(False, None, f"File {file_name}.parquet not found")
                df = pd.read_parquet(file_path)
                return ToolResult(True, {
                    "file": file_name,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "sample": df.head(limit).to_dict('records')
                })
            else:
                # Default to interactions file
                file_path = drugbank_path / "interactions.parquet"
                df = pd.read_parquet(file_path)
                return ToolResult(True, {
                    "file": "interactions",
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "sample": df.head(limit).to_dict('records')
                })

        elif db_name_lower == "pharos":
            pharos_path = data_path / "Drug" / "Pharos"
            if not pharos_path.exists():
                return ToolResult(False, None, f"Pharos directory not found at {pharos_path}")

            if query.lower() == "info":
                files = [f.name for f in pharos_path.glob("*.csv")]
                return ToolResult(True, {
                    "database": "Pharos",
                    "available_files": files,
                    "message": "Use query like 'file:pharos_drugs' to query specific file"
                })
            elif query.lower().startswith("file:"):
                file_name = query.split(":", 1)[1].strip()
                file_path = pharos_path / f"{file_name}.csv"
                if not file_path.exists():
                    return ToolResult(False, None, f"File {file_name}.csv not found")
                df = pd.read_csv(file_path, nrows=limit, low_memory=False)
                return ToolResult(True, {
                    "file": file_name,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "sample": df.to_dict('records')
                })
            else:
                # Default to drugs file
                file_path = pharos_path / "pharos_drugs.csv"
                df = pd.read_csv(file_path, nrows=limit, low_memory=False)
                return ToolResult(True, {
                    "file": "pharos_drugs",
                    "columns": df.columns.tolist(),
                    "sample": df.to_dict('records')
                })

        elif db_name_lower == "gwas":
            file_path = data_path / "GWAS" / "gwas_catalog_association.tsv"
            if not file_path.exists():
                return ToolResult(False, None, f"GWAS file not found at {file_path}")

            if query.lower() == "info":
                df_sample = pd.read_csv(file_path, sep="\t", nrows=5, low_memory=False)
                return ToolResult(True, {
                    "database": "GWAS",
                    "file": str(file_path),
                    "columns": df_sample.columns.tolist(),
                    "sample": df_sample.to_dict('records')
                })
            elif ":" in query:
                # Column-based search with chunked iteration
                col, value = query.split(":", 1)
                results, rows_searched = _chunked_search(file_path, "\t", col, value, limit=limit)
                return ToolResult(True, {
                    "count": len(results),
                    "rows_searched": rows_searched,
                    "results": results,
                    "message": f"Searched {rows_searched:,} rows, found {len(results)} matches"
                })
            else:
                df = pd.read_csv(file_path, sep="\t", nrows=limit, low_memory=False)
                return ToolResult(True, df.to_dict('records'))

        elif db_name_lower == "string" or db_name_lower == "stringdb":
            string_path = data_path / "PPI" / "StringDB"
            if not string_path.exists():
                # Try alternative paths
                alt_path = data_path / "StringDB"
                if alt_path.exists():
                    string_path = alt_path
                else:
                    return ToolResult(False, None,
                        f"STRING directory not found. Searched:\n"
                        f"  - {string_path}\n"
                        f"  - {alt_path}\n"
                        f"Expected structure: {{data_dir}}/PPI/StringDB/")

            if query.lower() == "info":
                files = [f.name for f in string_path.glob("*.txt")]
                return ToolResult(True, {
                    "database": "STRING",
                    "available_files": files,
                    "message": "Use query like 'file:sapiens.9606.protein.info.v12.0' to query specific file"
                })
            elif query.lower().startswith("file:"):
                file_name = query.split(":", 1)[1].strip()
                file_path = string_path / f"{file_name}.txt"
                if not file_path.exists():
                    return ToolResult(False, None, f"File {file_name}.txt not found")
                # STRING files use space delimiter for interactions, tab for others
                sep = " " if "links" in file_name else "\t"
                df = pd.read_csv(file_path, sep=sep, nrows=limit, low_memory=False)
                return ToolResult(True, {
                    "file": file_name,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "sample": df.to_dict('records')
                })
            else:
                # Default to protein info
                file_path = string_path / "sapiens.9606.protein.info.v12.0.txt"
                df = pd.read_csv(file_path, sep="\t", nrows=limit, low_memory=False)
                return ToolResult(True, {
                    "file": "protein.info",
                    "columns": df.columns.tolist(),
                    "sample": df.to_dict('records')
                })
        else:
            return ToolResult(False, None, f"Unknown database: {db_name}. Available: bindingdb, drugbank, pharos, gwas, string")

    except Exception as e:
        return ToolResult(False, None, f"Database query error: {str(e)}")


def read_file(file_path: str, input_dir: str = "./data") -> ToolResult:
    """Read a file from the input data directory.

    Args:
        file_path: Path to file (relative to input directory)
        input_dir: Base directory for question-specific input data

    Returns:
        ToolResult with file contents
    """
    try:
        data_dir = Path(input_dir)
        target_path = (data_dir / file_path).resolve()

        # Security check: ensure we're reading within data directory
        if not str(target_path).startswith(str(data_dir.resolve())):
            return ToolResult(False, None, "Access denied: path outside input directory")

        if not target_path.exists():
            return ToolResult(False, None, f"File not found: {file_path}")

        # Handle different file types
        if target_path.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(target_path)
            return ToolResult(True, {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "head": df.head().to_dict('records'),
            })

        elif target_path.suffix in [".csv", ".tsv"]:
            import pandas as pd
            sep = "\t" if target_path.suffix == ".tsv" else ","
            df = pd.read_csv(target_path, sep=sep, low_memory=False)
            return ToolResult(True, {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "head": df.head().to_dict('records'),
            })

        else:
            # Text file
            with open(target_path, "r") as f:
                content = f.read()
            return ToolResult(True, content)

    except Exception as e:
        return ToolResult(False, None, f"File read error: {str(e)}")


def find_files(
    pattern: Optional[str] = None,
    category: Optional[str] = None,
    extension: Optional[str] = None,
    name_contains: Optional[str] = None,
    question_context: Optional[str] = None,
    workspace_root: str = ".",
    data_dir: Optional[str] = None
) -> ToolResult:
    """Intelligently find relevant files in workspace without repeated directory traversal.
    
    This tool maintains an index of workspace files for efficient discovery.
    Much faster than multiple execute_python + os.listdir() calls.
    
    Args:
        pattern: Glob pattern to match (e.g., '**/Q5/*.csv', '**/*exhaustion*.csv')
        category: Filter by type: 'data', 'config', 'script', 'doc'
        extension: File extension (e.g., 'csv', '.csv', 'parquet')
        name_contains: Find files with this in the name (case-insensitive)
        question_context: Research question - will score files by relevance
        workspace_root: Workspace root directory (default: current directory)
        data_dir: Separate data directory to index (e.g., /path/to/Competition_Data)
        
    Returns:
        ToolResult with list of file paths and metadata
        
    Examples:
        find_files(extension='csv', question_context='T-cell exhaustion')
        find_files(pattern='**/Q5/*.csv')
        find_files(category='data', name_contains='DEG')
    """
    try:
        from src.utils.file_index import get_file_index
        
        # Get or create file index
        index = get_file_index(workspace_root, data_dir)
        
        if question_context:
            # Smart search based on question
            file_metadata = index.get_data_files(question_context=question_context)
        else:
            # Manual search by criteria
            file_metadata = index.find_files(
                pattern=pattern,
                category=category,
                extension=extension,
                name_contains=name_contains
            )
        
        # Format results for readability
        results = []
        for meta in file_metadata:
            results.append({
                "path": meta.path,
                "name": meta.name,
                "type": f"{meta.category}/{meta.subcategory}" if meta.subcategory else meta.category,
                "size_mb": round(meta.size_bytes / (1024 * 1024), 2)
            })
            
        summary = {
            "total_files": len(results),
            "files": results[:50]  # Limit to top 50 to avoid overwhelming output
        }
        
        if len(results) > 50:
            summary["note"] = f"Showing top 50 of {len(results)} files. Refine search if needed."
            
        return ToolResult(True, summary)
        
    except Exception as e:
        return ToolResult(False, None, f"File search error: {str(e)}")


def search_literature(
    question: str,
    mode: str = "auto",
    paper_dir: Optional[str] = None,
    max_sources: int = 5
) -> ToolResult:
    """Advanced literature search using PaperQA with local PDFs and/or internet search.

    This tool provides deep literature analysis by:
    - **PRIORITIZING** local PDF library (if available) 
    - Searching online databases (PubMed, arXiv) only if local papers insufficient
    - Reading full-text papers (not just abstracts)
    - Generating cited, evidence-based answers

    Search Strategy (mode='auto', RECOMMENDED):
    1. First queries local PDF library
    2. If local papers provide good answer → returns immediately (faster, more focused)
    3. If local papers insufficient → supplements with online search
    
    This ensures your curated local papers are always checked first!

    Args:
        question: Research question to answer (natural language)
        mode: Search mode:
            - 'auto' (default): Prioritize local, supplement with online if needed
            - 'local': Only local PDFs
            - 'online': Skip local, only internet search
            - 'hybrid': Search both simultaneously (no prioritization)
        paper_dir: Directory containing local PDF papers (uses config default if None)
        max_sources: Maximum number of source contexts to retrieve (default: 5)

    Returns:
        ToolResult with:
        - answer: Evidence-based answer with citations
        - contexts: List of relevant text passages from papers
        - references: Bibliography of cited papers
        - sources_used: Information about which sources were queried

    Note: More rigorous than search_pubmed as it reads full papers and synthesizes answers.
    """
    try:
        # Lazy import to avoid loading PaperQA unless needed
        from paperqa import Docs, Settings
        from paperqa.settings import AnswerSettings
        from pathlib import Path
        import os

        # Get configuration
        from src.config import get_global_config
        config = get_global_config()

        # IMPORTANT: Check embedding config format
        # PaperQA v5.x expects "st-model-name" for SentenceTransformer models
        # NOT "sentence-transformers/model-name"
        embedding_config = config.paperqa_embedding
        
        # Determine if using local embeddings based on "st-" prefix
        using_local_embeddings = embedding_config.startswith("st-")

        # CRITICAL FIX: Explicitly set API keys in os.environ for LiteLLM
        # LiteLLM checks os.environ at runtime, not just at import
        # For OpenRouter models, LiteLLM uses OpenAI client with custom base_url
        # The OpenAI client looks for OPENAI_API_KEY, so we must set it!
        if config.paperqa_llm.startswith("openrouter/"):
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
            if not api_key:
                return ToolResult(
                    False,
                    None,
                    "OPENROUTER_API_KEY not found. Set it in .env file or environment."
                )
            # Set all API key variants for maximum compatibility
            os.environ["OPENROUTER_API_KEY"] = api_key
            os.environ["OPENROUTER_KEY"] = api_key
            # CRITICAL: For openrouter/ models, LiteLLM uses OpenAI client internally
            # which expects OPENAI_API_KEY to be set!
            os.environ["OPENAI_API_KEY"] = api_key
            print(f"[DEBUG] Set API keys in os.environ (OPENROUTER_API_KEY, OPENAI_API_KEY): {api_key[:15]}...", file=sys.stderr)

        # For non-local embeddings, also need the API key
        if not using_local_embeddings and embedding_config.startswith("openrouter/"):
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
            if not api_key:
                return ToolResult(
                    False,
                    None,
                    "OPENROUTER_API_KEY needed for OpenRouter embeddings. Use local embeddings (st-model-name) to avoid API calls."
                )
            os.environ["OPENROUTER_API_KEY"] = api_key
            os.environ["OPENROUTER_KEY"] = api_key

        # Use provided paper_dir or fall back to config
        if paper_dir is None:
            paper_dir = config.paper_library_dir

        paper_dir_path = Path(paper_dir)

        # Check if local papers exist
        has_local_papers = paper_dir_path.exists() and any(paper_dir_path.glob("**/*.pdf"))

        # Adjust mode based on available resources
        if mode == "local" and not has_local_papers:
            return ToolResult(
                False,
                None,
                f"No local papers found in {paper_dir}. Use mode='online' or add PDFs to the directory."
            )

        if mode == "auto":
            # Auto mode: prioritize local, supplement with online if needed
            if has_local_papers:
                actual_mode = "local_first"  # Try local first, then add online if needed
            else:
                actual_mode = "online"
        else:
            actual_mode = mode

        # embedding_config was already set earlier (before API key checks)
        # No need to process it again here

        # Debug: Print what we're actually using
        print(f"[DEBUG] PaperQA LLM: {config.paperqa_llm}", file=sys.stderr)
        print(f"[DEBUG] PaperQA Embedding: {embedding_config}", file=sys.stderr)
        print(f"[DEBUG] Using local embeddings: {using_local_embeddings}", file=sys.stderr)

        # For local embeddings, verify sentence-transformers is installed
        if using_local_embeddings:
            try:
                import sentence_transformers
                print(f"[DEBUG] sentence-transformers version: {sentence_transformers.__version__}", file=sys.stderr)
            except ImportError:
                return ToolResult(
                    False,
                    None,
                    "Local embeddings require sentence-transformers. Install with: pip install sentence-transformers"
                )

        # Create settings - LiteLLM will automatically use OPENROUTER_KEY from environment
        # CRITICAL: Disable LLM usage during PDF parsing to avoid API calls
        # The LLM will ONLY be used during the query phase (docs.query())
        from paperqa.settings import ParsingSettings

        settings_kwargs = {
            "llm": config.paperqa_llm,
            "summary_llm": config.paperqa_llm,  # Use same LLM for summaries
            "embedding": embedding_config,
            # DISABLE ALL LLM USAGE DURING PDF PARSING
            # This ensures NO API calls when loading PDFs
            "parsing": ParsingSettings(
                use_doc_details=False,  # Don't use LLM to extract document details
                chunk_size=3000,  # Standard chunk size
                overlap=100,  # Standard overlap
                multimodal=False,  # CRITICAL: Disable image/figure processing (requires LLM)
                enrichment_llm=config.paperqa_llm,  # Use same LLM for enrichment (vision tasks)
            )
        }

        # Reduce token usage if using free model
        if ":free" in config.paperqa_llm.lower():
            settings_kwargs["answer"] = AnswerSettings(
                answer_max_sources=min(max_sources, 3),  # Reduce sources for free models
                evidence_k=5,  # Reduce evidence contexts
            )
        else:
            settings_kwargs["answer"] = AnswerSettings(
                answer_max_sources=max_sources,
                evidence_k=10
            )

        settings = Settings(**settings_kwargs)
        print(f"[DEBUG] Parsing config: use_doc_details={settings.parsing.use_doc_details} (LLM disabled during PDF loading)", file=sys.stderr)

        # Create document collection
        docs = Docs()

        sources_used = []
        local_answer = None

        # STAGE 1: Try local papers first (if mode allows)
        if actual_mode in ["local", "hybrid", "local_first"]:
            import time
            pdf_count = 0
            pdf_errors = []
            for pdf_file in paper_dir_path.glob("**/*.pdf"):
                # Retry with exponential backoff for rate limit errors
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        print(f"[DEBUG] Loading PDF: {pdf_file.name}...", file=sys.stderr)
                        # Provide a basic citation to SKIP LLM call during PDF loading
                        # The citation can be simple - the LLM will only be used during query
                        simple_citation = f"{pdf_file.stem}, Local PDF"
                        docs.add(str(pdf_file), citation=simple_citation, settings=settings)
                        pdf_count += 1
                        print(f"[DEBUG] Successfully loaded {pdf_file.name} (no LLM call)", file=sys.stderr)
                        break  # Success, move to next PDF
                    except Exception as e:
                        error_str = str(e).lower()
                        # Detailed error logging
                        print(f"[DEBUG] Error loading {pdf_file.name}: {str(e)}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()

                        # Check if it's a rate limit error
                        if "rate" in error_str or "429" in error_str:
                            if attempt < max_retries - 1:
                                # Exponential backoff: 2, 4, 8 seconds
                                wait_time = 2 ** (attempt + 1)
                                print(f"[DEBUG] Rate limit hit, retrying in {wait_time}s...", file=sys.stderr)
                                time.sleep(wait_time)
                                continue  # Retry
                        # Non-rate-limit error or final retry failed
                        # Include more error details for diagnosis
                        error_msg = str(e)
                        if "api_key" in error_str:
                            error_msg = f"API key error: {error_msg}"
                        pdf_errors.append(f"{pdf_file.name}: {error_msg}")
                        break  # Don't retry non-rate-limit errors

            sources_used.append(f"local_library ({pdf_count} PDFs)")

            # Report errors if any PDFs failed to load
            if pdf_errors and pdf_count == 0:
                error_details = '\n'.join(pdf_errors[:3])
                diagnosis = "\n\nDiagnosis:\n"
                if any("api_key" in e.lower() for e in pdf_errors):
                    diagnosis += "- API key not found by LiteLLM during PDF processing\n"
                    diagnosis += f"- OPENROUTER_API_KEY in env: {bool(os.getenv('OPENROUTER_API_KEY'))}\n"
                    diagnosis += f"- OPENROUTER_KEY in env: {bool(os.getenv('OPENROUTER_KEY'))}\n"
                    diagnosis += f"- Using local embeddings: {using_local_embeddings}\n"
                    if using_local_embeddings:
                        diagnosis += "- Local embeddings should NOT require API key\n"
                        diagnosis += "- This suggests PaperQA is making unexpected API calls\n"
                        diagnosis += "- Try: pip install --upgrade paper-qa sentence-transformers\n"

                return ToolResult(
                    False,
                    None,
                    f"Failed to load any PDFs from {paper_dir}.\n\nErrors:\n{error_details}{diagnosis}"
                )
            
            # Query local papers first
            if pdf_count > 0:
                try:
                    print(f"[DEBUG] Querying with LLM: {settings.llm}", file=sys.stderr)
                    print(f"[DEBUG] Querying with embedding: {settings.embedding}", file=sys.stderr)
                    local_answer = docs.query(question, settings=settings)
                    
                    # Check if local answer is sufficient (has contexts and not "I cannot answer")
                    has_good_local_answer = (
                        local_answer 
                        and local_answer.contexts 
                        and len(local_answer.contexts) > 0
                        and "cannot answer" not in local_answer.answer.lower()
                    )
                    
                    # If local_first mode and we have a good answer, return it without online search
                    if actual_mode == "local_first" and has_good_local_answer:
                        contexts = [
                            {
                                "text": ctx.context,
                                "citation": ctx.text.name if hasattr(ctx.text, 'name') else "Unknown",
                                "score": ctx.score if hasattr(ctx, 'score') else None
                            }
                            for ctx in local_answer.contexts
                        ]
                        
                        return ToolResult(True, {
                            "answer": local_answer.answer,
                            "contexts": contexts,
                            "references": local_answer.references if hasattr(local_answer, 'references') else [],
                            "sources_used": sources_used + ["(local papers sufficient - skipped online search)"],
                            "mode": "local"
                        })
                        
                except Exception as e:
                    # If local query fails, continue to online
                    sources_used.append(f"(local query error: {str(e)[:50]})")

        # STAGE 2: Add online search if needed
        if actual_mode in ["online", "hybrid", "local_first"]:
            import requests
            import urllib.parse
            import tempfile
            import xml.etree.ElementTree as ET

            total_papers_added = 0

            # Try Semantic Scholar first (good for general scientific papers)
            try:
                s2_url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {
                    "query": question,
                    "limit": max_sources,
                    "fields": "title,authors,year,abstract,openAccessPdf,externalIds,citationCount"
                }

                headers = {}
                if s2_api_key := os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
                    headers["x-api-key"] = s2_api_key

                response = requests.get(s2_url, params=params, headers=headers, timeout=30.0)
                response.raise_for_status()
                search_data = response.json()

                papers_data = search_data.get("data", [])
                s2_papers_added = 0

                for paper in papers_data[:max_sources]:
                    try:
                        pdf_info = paper.get("openAccessPdf")
                        if not pdf_info or not pdf_info.get("url"):
                            # Try arXiv if available
                            external_ids = paper.get("externalIds", {})
                            arxiv_id = external_ids.get("ArXiv")
                            if arxiv_id:
                                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                            else:
                                continue
                        else:
                            pdf_url = pdf_info["url"]

                        title = paper.get("title", "Unknown")
                        year = paper.get("year", "")
                        authors = paper.get("authors", [])
                        author_names = ", ".join([a.get("name", "") for a in authors[:3]])
                        if len(authors) > 3:
                            author_names += " et al."

                        citation = f"{author_names}. {title}. {year}" if year else f"{author_names}. {title}"

                        # Download PDF with proper headers to avoid 403 errors
                        pdf_headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        }
                        pdf_response = requests.get(pdf_url, headers=pdf_headers, timeout=60)
                        pdf_response.raise_for_status()

                        # Verify it's actually a PDF
                        content_type = pdf_response.headers.get("Content-Type", "")
                        if "pdf" not in content_type.lower() and not pdf_response.content.startswith(b"%PDF"):
                            continue

                        # Save to temporary file and add to docs
                        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as tmp_file:
                            tmp_file.write(pdf_response.content)
                            tmp_path = tmp_file.name

                        try:
                            docs.add(tmp_path, citation=citation, settings=settings)
                            s2_papers_added += 1
                        finally:
                            # Clean up temporary file
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass

                    except Exception:
                        continue

                if s2_papers_added > 0:
                    sources_used.append(f"semantic_scholar ({s2_papers_added} papers)")
                    total_papers_added += s2_papers_added

            except Exception:
                pass

            # Try PubMed/PMC (good for biomedical papers)
            try:
                search_term = urllib.parse.quote(question)
                search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_term}&retmode=json&retmax={max_sources}"

                search_response = requests.get(search_url, timeout=30)
                search_response.raise_for_status()
                search_data = search_response.json()

                pmids = search_data.get("esearchresult", {}).get("idlist", [])
                pmc_papers_added = 0

                for pmid in pmids[:max_sources]:
                    try:
                        # Check if paper is available in PMC (open access)
                        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
                        pmc_response = requests.get(pmc_url, timeout=30)
                        pmc_data = pmc_response.json()

                        records = pmc_data.get("records", [])
                        if records and records[0].get("pmcid"):
                            pmcid = records[0]["pmcid"]

                            # Get paper details for citation
                            details_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
                            details_response = requests.get(details_url, timeout=30)
                            details_data = details_response.json()

                            paper_info = details_data.get("result", {}).get(pmid, {})
                            title = paper_info.get("title", "Unknown")
                            authors = paper_info.get("authors", [])
                            author_names = ", ".join([a.get("name", "") for a in authors[:3]])
                            if len(authors) > 3:
                                author_names += " et al."

                            # PMC PDF URL
                            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"

                            citation = f"{author_names}. {title}. PMID: {pmid}"

                            # Download PDF with proper headers
                            pdf_headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                            }
                            pdf_response = requests.get(pdf_url, headers=pdf_headers, timeout=60)
                            pdf_response.raise_for_status()

                            # Save to temporary file and add to docs
                            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as tmp_file:
                                tmp_file.write(pdf_response.content)
                                tmp_path = tmp_file.name

                            try:
                                docs.add(tmp_path, citation=citation, settings=settings)
                                pmc_papers_added += 1
                            finally:
                                # Clean up temporary file
                                try:
                                    os.unlink(tmp_path)
                                except Exception:
                                    pass

                    except Exception:
                        continue

                if pmc_papers_added > 0:
                    sources_used.append(f"pubmed_pmc ({pmc_papers_added} papers)")
                    total_papers_added += pmc_papers_added

            except Exception:
                pass

        # Query the combined document collection
        answer_obj = docs.query(question, settings=settings)

        # Extract contexts and references
        contexts = [
            {
                "text": ctx.context,
                "citation": ctx.text.name if hasattr(ctx.text, 'name') else "Unknown",
                "score": ctx.score if hasattr(ctx, 'score') else None
            }
            for ctx in answer_obj.contexts
        ]

        return ToolResult(True, {
            "answer": answer_obj.answer,
            "contexts": contexts,
            "references": answer_obj.references if hasattr(answer_obj, 'references') else [],
            "sources_used": sources_used,
            "mode": actual_mode
        })

    except ImportError as e:
        return ToolResult(
            False,
            None,
            "PaperQA not installed. Run: pip install paper-qa"
        )
    except Exception as e:
        return ToolResult(False, None, f"Literature search error: {str(e)}")


def get_tool_definitions() -> list[dict[str, Any]]:
    """Get tool definitions for OpenRouter API.

    Returns:
        List of tool definition dicts
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": "Execute Python code to analyze data, perform calculations, or create visualizations. Use for bioinformatics analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute. Assume pandas, numpy, biopython are available.",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_pubmed",
                "description": "Search PubMed for scientific articles related to a query. Returns titles, abstracts, and author information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "PubMed search query (e.g., 'SARS-CoV-2 vaccine', 'alzheimer's disease protein')",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Query one of the competition databases (DrugBank, BindingDB, Pharos, STRING, GWAS). For large databases (BindingDB, GWAS), searches automatically iterate through chunks until finding enough results. Use 'info' query to see database structure.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "db_name": {
                            "type": "string",
                            "description": "Database name: 'drugbank', 'bindingdb', 'pharos', 'string', or 'gwas'",
                        },
                        "query": {
                            "type": "string",
                            "description": "Query specification: 'info' for database info, 'file:filename' for specific file, 'Column:value' for search (will search iteratively through file), or 'all' for sample rows",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of rows to return (default: 10). For searches, this is the target number of matches to find.",
                            "default": 10,
                        },
                    },
                    "required": ["db_name", "query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the input data directory (question-specific data like gene signatures, expression data). Supports parquet, CSV, TSV, and text files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to file relative to input directory (e.g., 'Q5/exhaustion_signature.csv')",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_files",
                "description": "Intelligently find relevant files in the workspace without repeated directory traversal. Much more efficient than using execute_python with os.listdir(). Can search by pattern, file type, or question relevance. Use this FIRST before trying to read files - it will help you discover what data is available.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Optional glob pattern to match (e.g., '**/Q5/*.csv', '**/*exhaustion*.csv')",
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional filter by file category: 'data', 'config', 'script', 'doc'",
                            "enum": ["data", "config", "script", "doc"],
                        },
                        "extension": {
                            "type": "string",
                            "description": "Optional file extension filter (e.g., 'csv', 'parquet', 'bam')",
                        },
                        "name_contains": {
                            "type": "string",
                            "description": "Optional: find files with this substring in the name (case-insensitive)",
                        },
                        "question_context": {
                            "type": "string",
                            "description": "Optional: research question text - will intelligently score and rank files by relevance to the question",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_literature",
                "description": "Advanced AI-powered literature search using PaperQA. PRIORITIZES local PDF library first, then supplements with online databases (PubMed, arXiv) if needed. Reads full-text papers and generates evidence-based answers with citations. More rigorous than search_pubmed - use this when you need detailed, cited information from research papers. Default 'auto' mode checks local PDFs first and only searches online if local papers don't provide a good answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Research question to answer in natural language (e.g., 'What are the mechanisms of EGFR inhibitor resistance?')",
                        },
                        "mode": {
                            "type": "string",
                            "description": "Search mode: 'local' (only local PDFs), 'online' (skip local, only internet), 'auto' (prioritize local, supplement with online if needed - RECOMMENDED), 'hybrid' (search both simultaneously)",
                            "enum": ["local", "online", "auto", "hybrid"],
                            "default": "auto",
                        },
                        "paper_dir": {
                            "type": "string",
                            "description": "Optional: Override default paper library directory (uses configured default if not provided)",
                        },
                        "max_sources": {
                            "type": "integer",
                            "description": "Maximum number of source contexts to retrieve (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["question"],
                },
            },
        },
    ]
