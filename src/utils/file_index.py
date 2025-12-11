"""Smart file discovery and indexing system for efficient data access."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class FileMetadata:
    """Metadata for indexed files."""
    path: str
    name: str
    extension: str
    size_bytes: int
    category: str  # 'data', 'config', 'script', 'doc', 'unknown'
    subcategory: Optional[str] = None  # e.g., 'csv', 'parquet', 'bam', 'pod5'


class FileIndex:
    """Index files in workspace for efficient discovery."""
    
    # File categorization rules
    DATA_EXTENSIONS = {'.csv', '.tsv', '.parquet', '.txt', '.json', '.xlsx', '.bam', '.pod5', '.vcf', '.fastq', '.fasta'}
    CONFIG_EXTENSIONS = {'.env', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
    SCRIPT_EXTENSIONS = {'.py', '.r', '.sh', '.bash', '.ipynb'}
    DOC_EXTENSIONS = {'.md', '.rst', '.pdf', '.html', '.tex'}
    
    def __init__(self, workspace_root: str, data_dir: Optional[str] = None):
        """Initialize file indexer.
        
        Args:
            workspace_root: Root directory of workspace
            data_dir: Optional separate data directory to index
        """
        self.workspace_root = Path(workspace_root)
        self.data_dir = Path(data_dir) if data_dir else None
        self.index: Dict[str, FileMetadata] = {}
        self._indexed = False
        
    def build_index(self, force_refresh: bool = False) -> None:
        """Build file index by scanning workspace.
        
        Args:
            force_refresh: Rebuild even if already indexed
        """
        if self._indexed and not force_refresh:
            return
            
        self.index.clear()
        
        # Index workspace
        self._scan_directory(self.workspace_root, max_depth=5)
        
        # Index separate data directory if specified
        if self.data_dir and self.data_dir.exists():
            self._scan_directory(self.data_dir, max_depth=4)
            
        self._indexed = True
        
    def _scan_directory(self, root_path: Path, max_depth: int, current_depth: int = 0) -> None:
        """Recursively scan directory and index files.
        
        Args:
            root_path: Directory to scan
            max_depth: Maximum recursion depth
            current_depth: Current recursion level
        """
        if current_depth > max_depth:
            return
            
        # Skip common irrelevant directories
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env', '.tox', 
                    'dist', 'build', '.pytest_cache', '.mypy_cache', 'egg-info'}
        
        try:
            for item in root_path.iterdir():
                # Skip hidden files and directories at root level
                if item.name.startswith('.') and current_depth == 0:
                    continue
                    
                if item.is_dir():
                    # Skip irrelevant directories
                    if item.name in skip_dirs:
                        continue
                    # Recurse into subdirectory
                    self._scan_directory(item, max_depth, current_depth + 1)
                    
                elif item.is_file():
                    # Index this file
                    metadata = self._create_metadata(item)
                    # Use absolute path as key to handle files from different root directories
                    abs_path = str(item.absolute())
                    self.index[abs_path] = metadata
                    
        except PermissionError:
            # Skip directories we can't read
            pass
            
    def _create_metadata(self, file_path: Path) -> FileMetadata:
        """Create metadata entry for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            FileMetadata object
        """
        ext = file_path.suffix.lower()
        
        # Categorize file
        if ext in self.DATA_EXTENSIONS:
            category = 'data'
            subcategory = ext[1:]  # Remove leading dot
        elif ext in self.CONFIG_EXTENSIONS:
            category = 'config'
            subcategory = ext[1:]
        elif ext in self.SCRIPT_EXTENSIONS:
            category = 'script'
            subcategory = ext[1:]
        elif ext in self.DOC_EXTENSIONS:
            category = 'doc'
            subcategory = ext[1:]
        else:
            category = 'unknown'
            subcategory = None
            
        return FileMetadata(
            path=str(file_path),
            name=file_path.name,
            extension=ext,
            size_bytes=file_path.stat().st_size,
            category=category,
            subcategory=subcategory
        )
        
    def find_files(self, 
                   pattern: Optional[str] = None,
                   category: Optional[str] = None,
                   extension: Optional[str] = None,
                   name_contains: Optional[str] = None) -> List[FileMetadata]:
        """Search indexed files by criteria.
        
        Args:
            pattern: Glob pattern to match against filename (e.g., '*exhaustion*.csv', '**/Q5/*.csv')
            category: Filter by category ('data', 'config', etc.)
            extension: Filter by extension (with or without dot)
            name_contains: Filter files whose name contains this string (case-insensitive)
            
        Returns:
            List of matching FileMetadata objects
        """
        if not self._indexed:
            self.build_index()
            
        results = []
        
        for abs_path, metadata in self.index.items():
            # Apply filters
            if category and metadata.category != category:
                continue
                
            if extension:
                ext = extension if extension.startswith('.') else f'.{extension}'
                if metadata.extension != ext.lower():
                    continue
                    
            if name_contains:
                if name_contains.lower() not in metadata.name.lower():
                    continue
                    
            if pattern:
                from fnmatch import fnmatch
                # Match pattern against filename or path
                if not (fnmatch(metadata.name, pattern) or fnmatch(abs_path, pattern)):
                    continue
                    
            results.append(metadata)
            
        return results
        
    def get_data_files(self, question_context: Optional[str] = None) -> List[FileMetadata]:
        """Get relevant data files, optionally filtered by question context.
        
        Args:
            question_context: Question text to help identify relevant files
            
        Returns:
            List of data file metadata
        """
        data_files = self.find_files(category='data')
        
        if not question_context:
            return data_files
            
        # Score files by relevance to question
        scored_files = []
        question_lower = question_context.lower()
        
        for file_meta in data_files:
            score = 0
            file_name_lower = file_meta.name.lower()
            file_path_lower = file_meta.path.lower()
            
            # Check for keyword matches
            keywords = ['exhaustion', 'tcell', 't-cell', 'deg', 'differential', 
                       'drug', 'binding', 'target', 'metadata', 'signature']
            
            for keyword in keywords:
                if keyword in question_lower:
                    if keyword in file_name_lower or keyword in file_path_lower:
                        score += 10
                        
            # Prefer certain file types
            if file_meta.extension in {'.csv', '.tsv', '.parquet'}:
                score += 5
                
            # Boost files in question-relevant directories
            if 'q5' in question_lower and 'Q5' in file_meta.path:
                score += 20
            if 'q2' in question_lower and 'Q2' in file_meta.path:
                score += 20
                
            scored_files.append((score, file_meta))
            
        # Sort by score (descending) and return
        scored_files.sort(reverse=True, key=lambda x: x[0])
        return [f for _, f in scored_files if _ > 0]  # Return only relevant files
        
    def to_json(self, output_path: Optional[str] = None) -> str:
        """Export index to JSON.
        
        Args:
            output_path: Optional file path to save JSON
            
        Returns:
            JSON string representation
        """
        index_dict = {path: asdict(meta) for path, meta in self.index.items()}
        json_str = json.dumps(index_dict, indent=2)
        
        if output_path:
            Path(output_path).write_text(json_str)
            
        return json_str
        
    def get_summary(self) -> Dict[str, int]:
        """Get summary statistics of indexed files.
        
        Returns:
            Dictionary with category counts
        """
        summary = {}
        for metadata in self.index.values():
            cat = metadata.category
            summary[cat] = summary.get(cat, 0) + 1
        return summary


# Singleton instance for workspace
_file_index: Optional[FileIndex] = None


def get_file_index(workspace_root: str = ".", data_dir: Optional[str] = None) -> FileIndex:
    """Get or create the global file index.
    
    Args:
        workspace_root: Workspace root directory
        data_dir: Optional separate data directory
        
    Returns:
        FileIndex instance
    """
    global _file_index
    
    if _file_index is None:
        _file_index = FileIndex(workspace_root, data_dir)
        _file_index.build_index()
        
    return _file_index


def smart_find_files(question: str, workspace_root: str = ".", data_dir: Optional[str] = None) -> List[str]:
    """Smart file discovery based on question context.
    
    This replaces multiple manual os.listdir() calls with intelligent search.
    
    Args:
        question: Research question to analyze
        workspace_root: Workspace root directory  
        data_dir: Optional separate data directory
        
    Returns:
        List of relevant file paths
    """
    index = get_file_index(workspace_root, data_dir)
    relevant_files = index.get_data_files(question_context=question)
    return [f.path for f in relevant_files]
