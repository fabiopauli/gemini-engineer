#!/usr/bin/env python3

"""
Main application for Gemini Engineer
Handles commands, conversation flow, and AI interactions
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from textwrap import dedent

# Third-party imports
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv

# Rich console imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Prompt toolkit imports
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle

# Import our modules
from config import (
    os_info, base_dir, git_context, model_context, security_context, logging_context,
    ADD_COMMAND_PREFIX, COMMIT_COMMAND_PREFIX, GIT_BRANCH_COMMAND_PREFIX,
    FUZZY_AVAILABLE, DEFAULT_MODEL, REASONER_MODEL, tools, SYSTEM_PROMPT,
    MAX_FILES_IN_ADD_DIR, MAX_FILE_CONTENT_SIZE_CREATE, EXCLUDED_FILES, EXCLUDED_EXTENSIONS,
    MAX_MULTIPLE_READ_SIZE, config
)
from utils import (
    console, detect_available_shells, get_context_usage_info, smart_truncate_history,
    validate_tool_calls, get_prompt_indicator, normalize_path, is_binary_file,
    read_local_file, add_file_context_smartly, find_best_matching_file,
    apply_fuzzy_diff_edit, run_bash_command, run_powershell_command,
    get_directory_tree_summary, toggle_logging, get_logging_status, log_user_message,
    log_api_response, log_tool_execution
)

# Initialize Gemini client
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize prompt session
prompt_session = PromptSession(
    style=PromptStyle.from_dict({
        'prompt': '#0066ff bold',
        'completion-menu.completion': 'bg:#1e3a8a fg:#ffffff',
        'completion-menu.completion.current': 'bg:#3b82f6 fg:#ffffff bold',
    })
)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class FileToCreate(BaseModel):
    path: str
    content: str

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str

# =============================================================================
# FILE OPERATIONS
# =============================================================================

def create_file(path: str, content: str, require_confirmation: bool = True) -> None:
    """
    Create or overwrite a file with given content.
    
    Args:
        path: File path
        content: File content
        require_confirmation: If True, prompt for confirmation when overwriting existing files
        
    Raises:
        ValueError: If file content exceeds size limit, path contains invalid characters, 
                   or user cancels overwrite
    """
    file_path = Path(path)
    if any(part.startswith('~') for part in file_path.parts):
        raise ValueError("Home directory references not allowed")
    
    # Check content size limit
    if len(content.encode('utf-8')) > MAX_FILE_CONTENT_SIZE_CREATE:
        raise ValueError(f"File content exceeds maximum size limit of {MAX_FILE_CONTENT_SIZE_CREATE} bytes")
    
    normalized_path_str = normalize_path(str(file_path))
    normalized_path = Path(normalized_path_str)
    
    # Check if file exists and prompt for confirmation if required
    if require_confirmation and normalized_path.exists():
        try:
            # Get file info for the confirmation prompt
            file_size = normalized_path.stat().st_size
            file_size_str = f"{file_size:,} bytes" if file_size < 1024 else f"{file_size/1024:.1f} KB"
            
            confirm = prompt_session.prompt(
                f"🔵 File '{normalized_path_str}' exists ({file_size_str}). Overwrite? (y/N): ",
                default="n"
            ).strip().lower()
            
            if confirm not in ["y", "yes"]:
                raise ValueError("File overwrite cancelled by user")
                
        except (KeyboardInterrupt, EOFError):
            raise ValueError("File overwrite cancelled by user")
    
    # Create the file
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    with open(normalized_path_str, "w", encoding="utf-8") as f:
        f.write(content)
    
    action = "Updated" if normalized_path.exists() else "Created"
    console.print(f"[bold blue]✓[/bold blue] {action} file at '[bright_cyan]{normalized_path_str}[/bright_cyan]'")
    
    if git_context['enabled'] and not git_context['skip_staging']:
        stage_file(normalized_path_str)

def add_directory_to_conversation(directory_path: str, conversation_history: List[Dict[str, Any]]) -> None:
    """
    Add all files from a directory to the conversation context.
    
    Args:
        directory_path: Path to directory to scan
        conversation_history: Conversation history to add files to
    """
    with console.status("[bold bright_blue]🔍 Scanning directory...[/bold bright_blue]") as status:
        skipped: List[str] = []
        added: List[str] = []
        total_processed = 0
        
        for root, dirs, files in os.walk(directory_path):
            if total_processed >= MAX_FILES_IN_ADD_DIR: 
                console.print(f"[yellow]⚠ Max files ({MAX_FILES_IN_ADD_DIR}) reached for dir scan.")
                break
            status.update(f"[bold bright_blue]🔍 Scanning {root}...[/bold bright_blue]")
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in EXCLUDED_FILES]
            
            for file in files:
                if total_processed >= MAX_FILES_IN_ADD_DIR: 
                    break
                if (file.startswith('.') or 
                    file in EXCLUDED_FILES or 
                    os.path.splitext(file)[1] in EXCLUDED_EXTENSIONS):
                    continue
                    
                full_path = os.path.join(root, file)
                try:
                    if is_binary_file(full_path): 
                        skipped.append(f"{full_path} (binary)")
                        continue
                        
                    norm_path = normalize_path(full_path)
                    content = read_local_file(norm_path)
                    if add_file_context_smartly(conversation_history, norm_path, content):
                        added.append(norm_path)
                    else:
                        skipped.append(f"{full_path} (too large for context)")
                    total_processed += 1
                except (OSError, ValueError) as e: 
                    skipped.append(f"{full_path} (error: {e})")
                    
        console.print(f"[bold blue]✓[/bold blue] Added folder '[bright_cyan]{directory_path}[/bright_cyan]'.")
        if added: 
            console.print(f"\n[bold bright_blue]📁 Added:[/bold bright_blue] ({len(added)} of {total_processed} valid) {[Path(f).name for f in added[:5]]}{'...' if len(added) > 5 else ''}")
        if skipped: 
            console.print(f"\n[yellow]⏭ Skipped:[/yellow] ({len(skipped)}) {[Path(f).name for f in skipped[:3]]}{'...' if len(skipped) > 3 else ''}")
        console.print()

# =============================================================================
# GIT OPERATIONS
# =============================================================================

def stage_file(file_path_str: str) -> bool:
    """
    Stage a file for git commit.
    
    Args:
        file_path_str: Path to file to stage
        
    Returns:
        True if staging was successful
    """
    if not git_context['enabled'] or git_context['skip_staging']: 
        return False
    try:
        repo_root = base_dir
        abs_file_path = Path(file_path_str).resolve() 
        rel_path = abs_file_path.relative_to(repo_root)
        result = subprocess.run(["git", "add", str(rel_path)], cwd=str(repo_root), capture_output=True, text=True, check=False)
        if result.returncode == 0: 
            console.print(f"[green dim]✓ Staged {rel_path}[/green dim]")
            return True
        else: 
            console.print(f"[yellow]⚠ Failed to stage {rel_path}: {result.stderr.strip()}[/yellow]")
            return False
    except ValueError: 
        console.print(f"[yellow]⚠ File {file_path_str} outside repo ({base_dir}), skipping staging[/yellow]")
        return False
    except Exception as e: 
        console.print(f"[red]✗ Error staging {file_path_str}: {e}[/red]")
        return False

def get_git_status_porcelain() -> Tuple[bool, List[Tuple[str, str]]]:
    """
    Get git status in porcelain format.
    
    Returns:
        Tuple of (has_changes, list_of_file_changes)
    """
    if not git_context['enabled']: 
        return False, []
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=str(base_dir))
        if not result.stdout.strip(): 
            return False, []
        changed_files = []
        for line in result.stdout.strip().split('\n'):
            if line:
                if len(line) >= 2 and line[1] == ' ':
                    status_code = line[:2]
                    filename = line[2:]
                else:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        status_code = parts[0].ljust(2)
                        filename = parts[1]
                    else:
                        status_code = line[:2] if len(line) >= 2 else line
                        filename = line[2:] if len(line) > 2 else ""
                
                changed_files.append((status_code, filename))
        return True, changed_files
    except subprocess.CalledProcessError as e: 
        console.print(f"[red]Error getting Git status: {e.stderr}[/red]")
        return False, []
    except FileNotFoundError: 
        console.print("[red]Git not found.[/red]")
        git_context['enabled'] = False
        return False, []

def create_gitignore() -> None:
    """Create a comprehensive .gitignore file if it doesn't exist."""
    gitignore_path = base_dir / ".gitignore"
    if gitignore_path.exists(): 
        console.print("[yellow]⚠ .gitignore exists, skipping.[/yellow]")
        return
        
    patterns = [
        "# Python", "__pycache__/", "*.pyc", "*.pyo", "*.pyd", ".Python", 
        "env/", "venv/", ".venv", "ENV/", "*.egg-info/", "dist/", "build/", 
        ".pytest_cache/", ".mypy_cache/", ".coverage", "htmlcov/", "", 
        "# Env", ".env", ".env*.local", "!.env.example", "", 
        "# IDE", ".vscode/", ".idea/", "*.swp", "*.swo", ".DS_Store", "", 
        "# Logs", "*.log", "logs/", "", 
        "# Temp", "*.tmp", "*.temp", "*.bak", "*.cache", "Thumbs.db", 
        "desktop.ini", "", 
        "# Node", "node_modules/", "npm-debug.log*", "yarn-debug.log*", 
        "pnpm-lock.yaml", "package-lock.json", "", 
        "# Local", "*.session", "*.checkpoint"
    ]
    
    console.print("\n[bold bright_blue]📝 Creating .gitignore[/bold bright_blue]")
    if prompt_session.prompt("🔵 Add custom patterns? (y/n, default n): ", default="n").strip().lower() in ["y", "yes"]:
        console.print("[dim]Enter patterns (empty line to finish):[/dim]")
        patterns.append("\n# Custom")
        while True: 
            pattern = prompt_session.prompt("  Pattern: ").strip()
            if pattern: 
                patterns.append(pattern)
            else: 
                break 
    try:
        with gitignore_path.open("w", encoding="utf-8") as f: 
            f.write("\n".join(patterns) + "\n")
        console.print(f"[green]✓ Created .gitignore ({len(patterns)} patterns)[/green]")
        if git_context['enabled']: 
            stage_file(str(gitignore_path))
    except OSError as e: 
        console.print(f"[red]✗ Error creating .gitignore: {e}[/red]")

def user_commit_changes(message: str) -> bool:
    """
    Commit STAGED changes with a given message. Prompts the user if nothing is staged.
    
    Args:
        message: Commit message
        
    Returns:
        True if commit was successful or action was taken.
    """
    if not git_context['enabled']:
        console.print("[yellow]Git not enabled.[/yellow]")
        return False
        
    try:
        # Check if there are any staged changes.
        staged_check = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=str(base_dir))
        
        # If exit code is 0, it means there are NO staged changes.
        if staged_check.returncode == 0:
            console.print("[yellow]No changes are staged for commit.[/yellow]")
            # Check if there are unstaged changes we can offer to add
            unstaged_check = subprocess.run(["git", "diff", "--quiet"], cwd=str(base_dir))
            if unstaged_check.returncode != 0: # Unstaged changes exist
                try:
                    confirm = prompt_session.prompt(
                        "🔵 However, there are unstaged changes. Stage all changes and commit? (y/N): ",
                        default="n"
                    ).strip().lower()
                    
                    if confirm in ["y", "yes"]:
                        console.print("[dim]Staging all changes...[/dim]")
                        subprocess.run(["git", "add", "-A"], cwd=str(base_dir), check=True)
                    else:
                        console.print("[yellow]Commit aborted. Use `/git add <files>` to stage changes.[/yellow]")
                        return True
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[yellow]Commit aborted.[/yellow]")
                    return True
            else: # No staged and no unstaged changes
                console.print("[dim]Working tree is clean. Nothing to commit.[/dim]")
                return True

        # At this point, we know there are staged changes, so we can commit.
        commit_res = subprocess.run(["git", "commit", "-m", message], cwd=str(base_dir), capture_output=True, text=True)
        
        if commit_res.returncode == 0:
            console.print(f"[green]✓ Committed successfully![/green]")
            log_info = subprocess.run(["git", "log", "--oneline", "-1"], cwd=str(base_dir), capture_output=True, text=True).stdout.strip()
            if log_info:
                console.print(f"[dim]Commit: {log_info}[/dim]")
            return True
        else:
            console.print(f"[red]✗ Commit failed:[/red]\n{commit_res.stderr.strip()}")
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(f"[red]✗ Git error: {e}[/red]")
        if isinstance(e, FileNotFoundError):
            git_context['enabled'] = False
        return False

# =============================================================================
# COMMAND HANDLERS
# =============================================================================

def show_git_status_cmd() -> bool:
    """Show git status."""
    if not git_context['enabled']: 
        console.print("[yellow]Git not enabled.[/yellow]")
        return True
    has_changes, files = get_git_status_porcelain()
    branch_raw = subprocess.run(["git", "branch", "--show-current"], cwd=str(base_dir), capture_output=True, text=True)
    branch_msg = f"On branch {branch_raw.stdout.strip()}" if branch_raw.returncode == 0 and branch_raw.stdout.strip() else "Not on any branch?"
    console.print(Panel(branch_msg, title="Git Status", border_style="blue", expand=False))
    if not has_changes: 
        console.print("[green]Working tree clean.[/green]")
        return True
    table = Table(show_header=True, header_style="bold bright_blue", border_style="blue")
    table.add_column("Sts", width=3)
    table.add_column("File Path")
    table.add_column("Description", style="dim")
    s_map = {
        " M": (" M", "Mod (unstaged)"), "MM": ("MM", "Mod (staged&un)"), 
        " A": (" A", "Add (unstaged)"), "AM": ("AM", "Add (staged&mod)"), 
        "AD": ("AD", "Add (staged&del)"), " D": (" D", "Del (unstaged)"), 
        "??": ("??", "Untracked"), "M ": ("M ", "Mod (staged)"), 
        "A ": ("A ", "Add (staged)"), "D ": ("D ", "Del (staged)"), 
        "R ": ("R ", "Ren (staged)"), "C ": ("C ", "Cop (staged)"), 
        "U ": ("U ", "Unmerged")
    }
    staged, unstaged, untracked = False, False, False
    for code, filename in files:
        disp_code, desc = s_map.get(code, (code, "Unknown"))
        table.add_row(disp_code, filename, desc)
        if code == "??": 
            untracked = True
        elif code.startswith(" "): 
            unstaged = True
        else: 
            staged = True
    console.print(table)
    if not staged and (unstaged or untracked): 
        console.print("\n[yellow]No changes added to commit.[/yellow]")
    if staged: 
        console.print("\n[green]Changes to be committed.[/green]")
    if unstaged: 
        console.print("[yellow]Changes not staged for commit.[/yellow]")
    if untracked: 
        console.print("[cyan]Untracked files present.[/cyan]")
    return True

def initialize_git_repo_cmd() -> bool:
    """Initialize a git repository."""
    if (base_dir / ".git").exists(): 
        console.print("[yellow]Git repo already exists.[/yellow]")
        git_context['enabled'] = True
        return True
    try:
        subprocess.run(["git", "init"], cwd=str(base_dir), check=True, capture_output=True)
        git_context['enabled'] = True
        branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(base_dir), capture_output=True, text=True)
        git_context['branch'] = branch_res.stdout.strip() if branch_res.returncode == 0 else "main"
        console.print(f"[green]✓ Initialized Git repo in {base_dir}/.git/ (branch: {git_context['branch']})[/green]")
        if not (base_dir / ".gitignore").exists() and prompt_session.prompt("🔵 No .gitignore. Create one? (y/n, default y): ", default="y").strip().lower() in ["y", "yes"]: 
            create_gitignore()
        elif git_context['enabled'] and (base_dir / ".gitignore").exists(): 
            stage_file(".gitignore")
        if prompt_session.prompt(f"🔵 Initial commit? (y/n, default n): ", default="n").strip().lower() in ["y", "yes"]: 
            user_commit_changes("Initial commit")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        console.print(f"[red]✗ Failed to init Git: {e}[/red]")
        if isinstance(e, FileNotFoundError): 
            git_context['enabled'] = False
        return False

def create_git_branch_cmd(branch_name: str) -> bool:
    """Create and switch to a git branch."""
    if not git_context['enabled']: 
        console.print("[yellow]Git not enabled.[/yellow]")
        return True
    if not branch_name: 
        console.print("[yellow]Branch name empty.[/yellow]")
        return True
    try:
        existing_raw = subprocess.run(["git", "branch", "--list", branch_name], cwd=str(base_dir), capture_output=True, text=True)
        if existing_raw.stdout.strip():
            console.print(f"[yellow]Branch '{branch_name}' exists.[/yellow]")
            current_raw = subprocess.run(["git", "branch", "--show-current"], cwd=str(base_dir), capture_output=True, text=True)
            if current_raw.stdout.strip() != branch_name and prompt_session.prompt(f"🔵 Switch to '{branch_name}'? (y/n, default y): ", default="y").strip().lower() in ["y", "yes"]:
                subprocess.run(["git", "checkout", branch_name], cwd=str(base_dir), check=True, capture_output=True)
                git_context['branch'] = branch_name
                console.print(f"[green]✓ Switched to branch '{branch_name}'[/green]")
            return True
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=str(base_dir), check=True, capture_output=True)
        git_context['branch'] = branch_name
        console.print(f"[green]✓ Created & switched to new branch '{branch_name}'[/green]")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        console.print(f"[red]✗ Branch op failed: {e}[/red]")
        if isinstance(e, FileNotFoundError): 
            git_context['enabled'] = False
        return False

def try_handle_git_info_command(user_input: str) -> bool:
    """Handle /git-info command to show git capabilities."""
    if user_input.strip().lower() == "/git-info":
        console.print("I can use Git commands to interact with a Git repository. Here's what I can do for you:\n\n"
                      "1. **Initialize a Git repository**: Use `git_init` to create a new Git repository in the current directory.\n"
                      "2. **Stage files for commit**: Use `git_add` to stage specific files for the next commit.\n"
                      "3. **Commit changes**: Use `git_commit` to commit staged changes with a message.\n"
                      "4. **Create and switch to a new branch**: Use `git_create_branch` to create a new branch and switch to it.\n"
                      "5. **Check Git status**: Use `git_status` to see the current state of the repository (staged, unstaged, or untracked files).\n\n"
                      "Let me know what you'd like to do, and I can perform the necessary Git operations for you. For example:\n"
                      "- Do you want to initialize a new repository?\n"
                      "- Stage and commit changes?\n"
                      "- Create a new branch? \n\n"
                      "Just provide the details, and I'll handle the rest!")
        return True
    return False

def try_handle_r1_command(user_input: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """Handle /r command for one-off reasoner calls."""
    if user_input.strip().lower() == "/r":
        try:
            user_prompt = prompt_session.prompt("🔵 Enter your reasoning prompt: ").strip()
            if not user_prompt:
                console.print("[yellow]No input provided. Aborting.[/yellow]")
                return True
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Cancelled.[/yellow]")
            return True
        
        temp_conversation = conversation_history + [{"role": "user", "content": user_prompt}]
        
        try:
            with console.status("[bold yellow]Gemini (Reasoner) is thinking...[/bold yellow]", spinner="dots"):
                full_response_content, accumulated_tool_calls = call_gemini_api(temp_conversation, REASONER_MODEL, is_reasoner=True)
            
            console.print("[bold bright_magenta]🧠 Gemini:[/bold bright_magenta] ", end="")
            if full_response_content:
                clean_content = full_response_content.replace("<think>", "").replace("</think>", "")
                console.print(clean_content, style="bright_magenta")
            else:
                console.print("[dim]Processing tool calls...[/dim]", style="bright_magenta")
            
            conversation_history.append({"role": "user", "content": user_prompt})
            assistant_message = {"role": "assistant", "content": full_response_content}
            
            valid_tool_calls = validate_tool_calls(accumulated_tool_calls)
            if valid_tool_calls:
                assistant_message["tool_calls"] = valid_tool_calls
                console.print("[dim]Note: Reasoner made tool calls. Executing...[/dim]")
                for tool_call in valid_tool_calls:
                    try:
                        result = execute_function_call_dict(tool_call)
                        tool_response = {
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": tool_call["id"]
                        }
                        conversation_history.append(tool_response)
                    except Exception as e:
                        console.print(f"[red]✗ Reasoner tool call error: {e}[/red]")
            
            conversation_history.append(assistant_message)
            return True
            
        except Exception as e:
            console.print(f"\n[red]✗ R1 reasoner error: {e}[/red]")
            return True
    
    return False

def try_handle_reasoner_command(user_input: str) -> bool:
    """Handle /reasoner command to toggle between chat and reasoner modes."""
    if user_input.strip().lower() == "/reasoner":
        if not model_context.get('is_reasoner', False):
            model_context['current_model'] = REASONER_MODEL
            model_context['is_reasoner'] = True
            console.print(f"[green]✓ Switched to {REASONER_MODEL} (reasoner mode) 🧠[/green]")
            console.print("[dim]All subsequent conversations will use thinking/reasoning capabilities.[/dim]")
        else:
            model_context['current_model'] = DEFAULT_MODEL
            model_context['is_reasoner'] = False
            console.print(f"[green]✓ Switched to {DEFAULT_MODEL} (chat mode) 💬[/green]")
            console.print("[dim]All subsequent conversations will use standard chat mode.[/dim]")
        return True
    return False

def try_handle_clear_command(user_input: str) -> bool:
    """Handle /clear command to clear screen."""
    if user_input.strip().lower() == "/clear":
        console.clear()
        return True
    return False

def try_handle_clear_context_command(user_input: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """Handle /clear-context command to clear conversation history."""
    if user_input.strip().lower() == "/clear-context":
        if len(conversation_history) <= 1:
            console.print("[yellow]Context already empty (only system prompt).[/yellow]")
            return True
            
        file_contexts = sum(1 for msg in conversation_history if msg["role"] == "system" and "User added file" in msg["content"])
        total_messages = len(conversation_history) - 1
        
        console.print(f"[yellow]Current context: {total_messages} messages, {file_contexts} file contexts[/yellow]")
        
        confirm = prompt_session.prompt("🔵 Clear conversation context? This cannot be undone (y/n): ").strip().lower()
        if confirm in ["y", "yes"]:
            original_system_prompt = conversation_history[0]
            conversation_history[:] = [original_system_prompt]
            console.print("[green]✓ Conversation context cleared. Starting fresh![/green]")
            console.print("[green]  All file contexts and conversation history removed.[/green]")
        else:
            console.print("[yellow]Context clear cancelled.[/yellow]")
        return True
    return False

def try_handle_folder_command(user_input: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """Handle /folder command to manage base directory."""
    import config
    global base_dir
    if user_input.strip().lower().startswith("/folder"):
        folder_path = user_input[len("/folder"):].strip()
        if not folder_path:
            console.print(f"[yellow]Current base directory: '{base_dir}'[/yellow]")
            console.print("[yellow]Usage: /folder <path> or /folder reset[/yellow]")
            return True
        if folder_path.lower() == "reset":
            old_base = base_dir
            current_cwd = Path.cwd()
            base_dir = current_cwd
            config.base_dir = current_cwd  # Update the config module's base_dir
            console.print(f"[green]✓ Base directory reset from '{old_base}' to: '{base_dir}'[/green]")
            console.print(f"[green]  Synchronized with current working directory: '{current_cwd}'[/green]")
            
            # Add directory change to conversation context so the assistant knows
            dir_summary = get_directory_tree_summary(base_dir)
            conversation_history.append({
                "role": "system",
                "content": f"Working directory reset to: {base_dir}\n\nCurrent directory structure:\n\n{dir_summary}"
            })
            
            return True
        try:
            new_base = Path(folder_path).resolve()
            if not new_base.exists() or not new_base.is_dir():
                console.print(f"[red]✗ Path does not exist or is not a directory: '{folder_path}'[/red]")
                return True
            test_file = new_base / ".eng-git-test"
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                console.print(f"[red]✗ No write permissions in directory: '{new_base}'[/red]")
                return True
            old_base = base_dir
            base_dir = new_base
            config.base_dir = new_base  # Update the config module's base_dir
            console.print(f"[green]✓ Base directory changed from '{old_base}' to: '{base_dir}'[/green]")
            console.print(f"[green]  All relative paths will now be resolved against this directory.[/green]")
            
            # Add directory change to conversation context so the assistant knows
            dir_summary = get_directory_tree_summary(base_dir)
            conversation_history.append({
                "role": "system",
                "content": f"Working directory changed to: {base_dir}\n\nNew directory structure:\n\n{dir_summary}"
            })
            
            return True
        except Exception as e:
            console.print(f"[red]✗ Error setting base directory: {e}[/red]")
            return True
    return False

def try_handle_exit_command(user_input: str) -> bool:
    """Handle /exit and /quit commands."""
    if user_input.strip().lower() in ("/exit", "/quit"):
        console.print("[bold blue]👋 Goodbye![/bold blue]")
        sys.exit(0)
    return False

def try_handle_context_command(user_input: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """Handle /context command to show context usage statistics."""
    if user_input.strip().lower() == "/context":
        context_info = get_context_usage_info(conversation_history, model_context.get('current_model'))
        
        context_table = Table(title="📊 Context Usage Statistics", show_header=True, header_style="bold bright_blue")
        context_table.add_column("Metric", style="bright_cyan")
        context_table.add_column("Value", style="white")
        context_table.add_column("Status", style="white")
        
        context_table.add_row("Total Messages", str(context_info["total_messages"]), "📝")
        context_table.add_row("Estimated Tokens", f"{context_info['estimated_tokens']:,}", f"{context_info['token_usage_percent']:.1f}% of {context_info['max_tokens']:,}")
        context_table.add_row("File Contexts", str(context_info["file_contexts"]), f"Max: 5")
        
        if context_info["critical_limit"]:
            status_color = "red"
            status_text = "🔴 Critical - aggressive truncation active"
        elif context_info["approaching_limit"]:
            status_color = "yellow"
            status_text = "🟡 Warning - approaching limits"
        else:
            status_color = "green"
            status_text = "🟢 Healthy - plenty of space"
        
        context_table.add_row("Context Health", status_text, "")
        console.print(context_table)
        
        if context_info["token_breakdown"]:
            breakdown_table = Table(title="📋 Token Breakdown by Role", show_header=True, header_style="bold bright_blue", border_style="blue")
            breakdown_table.add_column("Role", style="bright_cyan")
            breakdown_table.add_column("Tokens", style="white")
            breakdown_table.add_column("Percentage", style="white")
            
            total_tokens = context_info["estimated_tokens"]
            for role, tokens in context_info["token_breakdown"].items():
                if tokens > 0:
                    percentage = (tokens / total_tokens * 100) if total_tokens > 0 else 0
                    breakdown_table.add_row(
                        role.capitalize(),
                        f"{tokens:,}",
                        f"{percentage:.1f}%"
                    )
            
            console.print(breakdown_table)
        
        if context_info["approaching_limit"]:
            console.print("\n[yellow]💡 Recommendations to manage context:[/yellow]")
            console.print("[yellow]  • Use /clear-context to start fresh[/yellow]")
            console.print("[yellow]  • Remove large files from context[/yellow]")
            console.print("[yellow]  • Work with smaller file sections[/yellow]")
        
        return True
    return False

def try_handle_help_command(user_input: str) -> bool:
    """Handle /help command to show available commands."""
    if user_input.strip().lower() == "/help":
        help_table = Table(title="📝 Available Commands", show_header=True, header_style="bold bright_blue")
        help_table.add_column("Command", style="bright_cyan")
        help_table.add_column("Description", style="white")
        
        # General commands
        help_table.add_row("/help", "Show this help")
        help_table.add_row("/r", "Call Reasoner model for one-off reasoning tasks")
        help_table.add_row("/reasoner", "Toggle between chat and reasoner models")
        help_table.add_row("/clear", "Clear screen")
        help_table.add_row("/clear-context", "Clear conversation context")
        help_table.add_row("/context", "Show context usage statistics")
        help_table.add_row("/log", "Toggle message logging (saves to logs directory)")
        help_table.add_row("/os", "Show operating system information")
        help_table.add_row("/exit, /quit", "Exit application")
        
        # Directory & file management
        help_table.add_row("/folder", "Show current base directory")
        help_table.add_row("/folder <path>", "Set base directory for file operations")
        help_table.add_row("/folder reset", "Reset base directory to current working directory")
        help_table.add_row(f"{ADD_COMMAND_PREFIX.strip()} <path>", "Add file/dir to conversation context (supports fuzzy matching)")
        
        # Git workflow commands
        help_table.add_row("/git init", "Initialize Git repository")
        help_table.add_row("/git status", "Show Git status")
        help_table.add_row(f"{GIT_BRANCH_COMMAND_PREFIX.strip()} <name>", "Create & switch to new branch")
        help_table.add_row("/git add <. or <file1> <file2>", "Stage all files or specific ones for commit")
        help_table.add_row("/git commit", "Commit changes (prompts if no message)")
        help_table.add_row("/git-info", "Show detailed Git capabilities")
        
        console.print(help_table)
        
        # Show current model status
        current_model_name = "Reasoner 🧠" if model_context['is_reasoner'] else "Chat 💬"
        console.print(f"\n[dim]Current model: {current_model_name}[/dim]")
        
        # Show logging status
        logging_status = get_logging_status()
        console.print(f"[dim]{logging_status}[/dim]")
        
        # Show fuzzy matching status
        fuzzy_status = "✓ Available" if FUZZY_AVAILABLE else "✗ Not installed (pip install thefuzz python-levenshtein)"
        console.print(f"[dim]Fuzzy matching: {fuzzy_status}[/dim]")
        
        # Show OS and shell status
        available_shells = [shell for shell, available in os_info['shell_available'].items() if available]
        shell_status = ", ".join(available_shells) if available_shells else "None detected"
        console.print(f"[dim]OS: {os_info['system']} | Available shells: {shell_status}[/dim]")
        
        return True
    return False

def try_handle_os_command(user_input: str) -> bool:
    """Handle /os command to show operating system information."""
    if user_input.strip().lower() == "/os":
        os_table = Table(title="🖥️ Operating System Information", show_header=True, header_style="bold bright_blue")
        os_table.add_column("Property", style="bright_cyan")
        os_table.add_column("Value", style="white")
        
        # Basic OS info
        os_table.add_row("System", os_info['system'])
        os_table.add_row("Release", os_info['release'])
        os_table.add_row("Version", os_info['version'])
        os_table.add_row("Machine", os_info['machine'])
        if os_info['processor']:
            os_table.add_row("Processor", os_info['processor'])
        os_table.add_row("Python Version", os_info['python_version'])
        
        console.print(os_table)
        
        # Shell availability
        shell_table = Table(title="🐚 Shell Availability", show_header=True, header_style="bold bright_blue")
        shell_table.add_column("Shell", style="bright_cyan")
        shell_table.add_column("Status", style="white")
        
        for shell, available in os_info['shell_available'].items():
            status = "✓ Available" if available else "✗ Not available"
            shell_table.add_row(shell.capitalize(), status)
        
        console.print(shell_table)
        
        # Platform-specific recommendations
        if os_info['is_windows']:
            console.print("\n[yellow]💡 Windows detected:[/yellow]")
            console.print("[yellow]  • PowerShell commands are preferred[/yellow]")
            if os_info['shell_available']['bash']:
                console.print("[yellow]  • Bash is available (WSL or Git Bash)[/yellow]")
        elif os_info['is_mac']:
            console.print("\n[yellow]💡 macOS detected:[/yellow]")
            console.print("[yellow]  • Bash and zsh commands are preferred[/yellow]")
            console.print("[yellow]  • PowerShell Core may be available[/yellow]")
        elif os_info['is_linux']:
            console.print("\n[yellow]💡 Linux detected:[/yellow]")
            console.print("[yellow]  • Bash commands are preferred[/yellow]")
            console.print("[yellow]  • PowerShell Core may be available[/yellow]")
        
        return True
    return False

def try_handle_log_command(user_input: str) -> bool:
    """Handle /log command to toggle message logging."""
    if user_input.strip().lower() == "/log":
        status_message = toggle_logging()
        console.print(f"[bold blue]{status_message}[/bold blue]")
        
        # Show additional info when logging is enabled
        if logging_context['enabled']:
            console.print(f"[dim]Log file: {logging_context['current_log_file'].name}[/dim]")
            console.print("[dim]Logs include: user messages, API responses, token usage, and tool calls[/dim]")
        
        return True
    return False

def try_handle_git_add_command(user_input: str) -> bool:
    """Handle the /git add command for staging files."""
    GIT_ADD_COMMAND_PREFIX = "/git add "
    
    if user_input.strip().lower().startswith(GIT_ADD_COMMAND_PREFIX.strip()):
        if not git_context['enabled']:
            console.print("[yellow]Git not enabled. Use `/git init` first.[/yellow]")
            return True
            
        files_to_add_str = user_input[len(GIT_ADD_COMMAND_PREFIX):].strip()
        if not files_to_add_str:
            console.print("[yellow]Usage: /git add <file1> <file2> ... or /git add .[/yellow]")
            return True
            
        file_paths = files_to_add_str.split()
        
        staged_ok: List[str] = []
        failed_stage: List[str] = []
        
        for fp_str in file_paths:
            if fp_str == ".":
                try:
                    subprocess.run(["git", "add", "."], cwd=str(base_dir), check=True, capture_output=True)
                    console.print("[green]✓ Staged all changes in the current directory.[/green]")
                    return True
                except subprocess.CalledProcessError as e:
                    console.print(f"[red]✗ Failed to stage all changes: {e.stderr}[/red]")
                    return True

            try:
                if stage_file(fp_str):
                    staged_ok.append(fp_str)
                else:
                    failed_stage.append(fp_str)
            except Exception as e:
                failed_stage.append(f"{fp_str} (error: {e})")
        
        if staged_ok:
            console.print(f"[green]✓ Staged:[/green] {', '.join(staged_ok)}")
        if failed_stage:
            console.print(f"[yellow]⚠ Failed to stage:[/yellow] {', '.join(failed_stage)}")
        
        show_git_status_cmd()
        return True
        
    return False

def try_handle_commit_command(user_input: str) -> bool:
    """Handle /git commit command for git commits."""
    if user_input.strip().lower().startswith(COMMIT_COMMAND_PREFIX.strip()):
        if not git_context['enabled']:
            console.print("[yellow]Git not enabled. Use `/git init` first.[/yellow]")
            return True

        message = user_input[len(COMMIT_COMMAND_PREFIX):].strip()

        if not message:
            try:
                message = prompt_session.prompt("🔵 Enter commit message: ").strip()
                if not message:
                    console.print("[yellow]Commit aborted. Message cannot be empty.[/yellow]")
                    return True
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Commit aborted by user.[/yellow]")
                return True

        user_commit_changes(message)
        return True
    return False

def try_handle_git_command(user_input: str) -> bool:
    """Handle various git commands."""
    cmd = user_input.strip().lower()
    if cmd == "/git init": 
        return initialize_git_repo_cmd()
    elif cmd.startswith(GIT_BRANCH_COMMAND_PREFIX.strip()):
        branch_name = user_input[len(GIT_BRANCH_COMMAND_PREFIX.strip()):].strip()
        if not branch_name and cmd == GIT_BRANCH_COMMAND_PREFIX.strip():
             console.print("[yellow]Specify branch name: /git branch <name>[/yellow]")
             return True
        return create_git_branch_cmd(branch_name)
    elif cmd == "/git status": 
        return show_git_status_cmd()
    return False

def try_handle_add_command(user_input: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """Handle /add command with fuzzy file finding support."""
    if user_input.strip().lower().startswith(ADD_COMMAND_PREFIX):
        path_to_add = user_input[len(ADD_COMMAND_PREFIX):].strip()
        
        # 1. Try direct path first
        try:
            p = (base_dir / path_to_add).resolve()
            if p.exists():
                normalized_path = str(p)
            else:
                # This will raise an error if it doesn't exist, triggering the fuzzy search
                _ = p.resolve(strict=True) 
        except (FileNotFoundError, OSError):
            # 2. If direct path fails, try fuzzy finding
            console.print(f"[dim]Path '{path_to_add}' not found directly, attempting fuzzy search...[/dim]")
            fuzzy_match = find_best_matching_file(base_dir, path_to_add)

            if fuzzy_match:
                # Optional: Confirm with user for better UX
                relative_fuzzy = Path(fuzzy_match).relative_to(base_dir)
                confirm = prompt_session.prompt(f"🔵 Did you mean '[bright_cyan]{relative_fuzzy}[/bright_cyan]'? (Y/n): ", default="y").strip().lower()
                if confirm in ["y", "yes"]:
                    normalized_path = fuzzy_match
                else:
                    console.print("[yellow]Add command cancelled.[/yellow]")
                    return True
            else:
                console.print(f"[bold red]✗[/bold red] Path does not exist: '[bright_cyan]{path_to_add}[/bright_cyan]'")
                if FUZZY_AVAILABLE:
                    console.print("[dim]Tip: Try a partial filename (e.g., 'main.py' instead of exact path)[/dim]")
                return True
        
        # --- Process the found file/directory ---
        try:
            if Path(normalized_path).is_dir():
                add_directory_to_conversation(normalized_path, conversation_history)
            else:
                content = read_local_file(normalized_path)
                if add_file_context_smartly(conversation_history, normalized_path, content):
                    console.print(f"[bold blue]✓[/bold blue] Added file '[bright_cyan]{normalized_path}[/bright_cyan]' to conversation.\n")
                else:
                    console.print(f"[bold yellow]⚠[/bold yellow] File '[bright_cyan]{normalized_path}[/bright_cyan]' too large for context.\n")
        except (OSError, ValueError) as e:
            console.print(f"[bold red]✗[/bold red] Could not add path '[bright_cyan]{path_to_add}[/bright_cyan]': {e}\n")
        return True
    return False

# =============================================================================
# GEMINI API HELPER FUNCTIONS
# =============================================================================

def convert_conversation_to_gemini(conversation_history: List[Dict[str, Any]], system_prompt: str) -> List[types.Content]:
    """Convert conversation history to Gemini format."""
    contents = []
    
    for message in conversation_history:
        if message["role"] == "system":
            continue  # System messages will be handled separately
        elif message["role"] == "user":
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=message["content"])]
            ))
        elif message["role"] == "assistant":
            # For assistant messages, we need to handle both text and tool calls
            parts = []
            if message.get("content"):
                parts.append(types.Part.from_text(text=message["content"]))
            
            # Only add assistant message if it has content
            if parts:
                contents.append(types.Content(
                    role="model",
                    parts=parts
                ))
        elif message["role"] == "tool":
            # Tool responses should be included as model responses to maintain context
            contents.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=message['content'])]
            ))
    
    return contents

def call_gemini_api(conversation_history: List[Dict[str, Any]], current_model: str, is_reasoner: bool = False) -> Tuple[str, List[Dict[str, Any]], Any]:
    """Call Gemini API with proper format and return content + tool calls + response object."""
    from config import convert_tools_to_gemini
    
    # Extract system prompt from conversation
    system_prompt = ""
    for msg in conversation_history:
        if msg["role"] == "system":
            system_prompt += msg["content"] + "\n"
    
    # Convert conversation to Gemini format
    contents = convert_conversation_to_gemini(conversation_history, system_prompt.strip())
    
    # Get tools in Gemini format
    gemini_tools = convert_tools_to_gemini()
    
    # Configure generation
    thinking_budget = -1 if is_reasoner else 0
    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        thinking_config=types.ThinkingConfig(
            thinking_budget=thinking_budget,
        ),
        tools=gemini_tools,
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text=system_prompt.strip()),
        ],
    )
    
    # Call API (non-streaming for tool calls)
    response = client.models.generate_content(
        model=current_model,
        contents=contents,
        config=generate_content_config,
    )
    
    # Extract content and tool calls
    response_content = ""
    tool_calls = []
    
    if response.candidates and response.candidates[0].content:
        candidate = response.candidates[0]
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                response_content += part.text
            elif hasattr(part, 'function_call') and part.function_call:
                # Convert Gemini function call to our format
                func_call = part.function_call
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",  # Generate ID
                    "type": "function",
                    "function": {
                        "name": func_call.name,
                        "arguments": json.dumps(dict(func_call.args))
                    }
                })
    
    return response_content, tool_calls, response

# =============================================================================
# LLM TOOL HANDLER FUNCTIONS
# =============================================================================

def ensure_file_in_context(file_path: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """
    Ensure a file is loaded in the conversation context.
    
    Args:
        file_path: Path to the file
        conversation_history: Conversation history to add to
        
    Returns:
        True if file was successfully added to context
    """
    try:
        normalized_path = normalize_path(file_path)
        content = read_local_file(normalized_path)
        marker = f"User added file '{normalized_path}'"
        if not any(msg["role"] == "system" and marker in msg["content"] for msg in conversation_history):
            return add_file_context_smartly(conversation_history, normalized_path, content)
        return True
    except (OSError, ValueError) as e:
        console.print(f"[red]✗ Error reading file for context '{file_path}': {e}[/red]")
        return False

def llm_git_init() -> str:
    """LLM tool handler for git init."""
    if (base_dir / ".git").exists(): 
        git_context['enabled'] = True
        return "Git repository already exists."
    try:
        subprocess.run(["git", "init"], cwd=str(base_dir), check=True, capture_output=True)
        git_context['enabled'] = True
        branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(base_dir), capture_output=True, text=True)
        git_context['branch'] = branch_res.stdout.strip() if branch_res.returncode == 0 else "main"
        if not (base_dir / ".gitignore").exists(): 
            create_gitignore()
        elif git_context['enabled']: 
            stage_file(".gitignore")
        return f"Git repository initialized successfully in {base_dir}/.git/ (branch: {git_context['branch']})."

    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        if isinstance(e, FileNotFoundError):
            git_context['enabled'] = False
        return f"Failed to initialize Git repository: {e}"

def llm_git_add(file_paths: List[str]) -> str:
    """LLM tool handler for git add."""
    if not git_context['enabled']: 
        return "Git not initialized."
    if not file_paths: 
        return "No file paths to stage."
    staged_ok: List[str] = []
    failed_stage: List[str] = []
    for fp_str in file_paths:
        try: 
            norm_fp = normalize_path(fp_str)
            if stage_file(norm_fp):
                staged_ok.append(norm_fp)
            else:
                failed_stage.append(norm_fp)
        except ValueError as e: 
            failed_stage.append(f"{fp_str} (path error: {e})")
        except Exception as e: 
            failed_stage.append(f"{fp_str} (error: {e})")
    res = []
    if staged_ok: 
        res.append(f"Staged: {', '.join(Path(p).name for p in staged_ok)}")
    if failed_stage: 
        res.append(f"Failed to stage: {', '.join(str(Path(p).name if isinstance(p,str) else p) for p in failed_stage)}")
    return ". ".join(res) + "." if res else "No files staged. Check paths."

def llm_git_commit(message: str, require_confirmation: bool = True) -> str:
    """
    LLM tool handler for git commit with optional confirmation.
    
    Args:
        message: Commit message
        require_confirmation: If True, prompt for confirmation when there are uncommitted changes
    
    Returns:
        Commit result message
    """
    if not git_context['enabled']: 
        return "Git not initialized."
    if not message: 
        return "Commit message empty."
    
    try:
        # Check if there are staged changes
        staged_check = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=str(base_dir))
        if staged_check.returncode == 0: 
            return "No changes staged. Use git_add first."
        
        # Check for uncommitted changes in working directory
        if require_confirmation:
            uncommitted_check = subprocess.run(["git", "diff", "--quiet"], cwd=str(base_dir))
            if uncommitted_check.returncode != 0:
                # There are uncommitted changes
                try:
                    confirm = prompt_session.prompt(
                        "🔵 There are uncommitted changes in your working directory. "
                        "Commit staged changes anyway? (y/N): ",
                        default="n"
                    ).strip().lower()
                    
                    if confirm not in ["y", "yes"]:
                        return "Commit cancelled by user. Consider staging all changes first."
                        
                except (KeyboardInterrupt, EOFError):
                    return "Commit cancelled by user."
        
        # Show what will be committed
        staged_files = subprocess.run(
            ["git", "diff", "--staged", "--name-only"], 
            cwd=str(base_dir), 
            capture_output=True, 
            text=True
        ).stdout.strip()
        
        if staged_files:
            console.print(f"[dim]Committing files: {staged_files.replace(chr(10), ', ')}[/dim]")
        
        # Perform the commit
        result = subprocess.run(["git", "commit", "-m", message], cwd=str(base_dir), capture_output=True, text=True)
        if result.returncode == 0:
            info_raw = subprocess.run(["git", "log", "-1", "--pretty=%h %s"], cwd=str(base_dir), capture_output=True, text=True).stdout.strip()
            return f"Committed successfully. Commit: {info_raw}"
        return f"Failed to commit: {result.stderr.strip()}"
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        if isinstance(e, FileNotFoundError):
            git_context['enabled'] = False
        return f"Git commit error: {e}"
    except Exception as e: 
        console.print_exception()
        return f"Unexpected commit error: {e}"

def llm_git_create_branch(branch_name: str) -> str:
    """LLM tool handler for git branch creation."""
    if not git_context['enabled']: 
        return "Git not initialized."
    bn = branch_name.strip()
    if not bn: 
        return "Branch name empty."
    try:
        exist_res = subprocess.run(["git", "rev-parse", "--verify", f"refs/heads/{bn}"], cwd=str(base_dir), capture_output=True, text=True)
        if exist_res.returncode == 0:
            current_raw = subprocess.run(["git", "branch", "--show-current"], cwd=str(base_dir), capture_output=True, text=True)
            if current_raw.stdout.strip() == bn: 
                return f"Already on branch '{bn}'."
            subprocess.run(["git", "checkout", bn], cwd=str(base_dir), check=True, capture_output=True, text=True)
            git_context['branch'] = bn
            return f"Branch '{bn}' exists. Switched to it."
        subprocess.run(["git", "checkout", "-b", bn], cwd=str(base_dir), check=True, capture_output=True, text=True)
        git_context['branch'] = bn
        return f"Created & switched to new branch '{bn}'."
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        if isinstance(e, FileNotFoundError):
            git_context['enabled'] = False
        return f"Branch op failed for '{bn}': {e}"

def llm_git_status() -> str:
    """LLM tool handler for git status."""
    if not git_context['enabled']: 
        return "Git not initialized."
    try:
        branch_res = subprocess.run(["git", "branch", "--show-current"], cwd=str(base_dir), capture_output=True, text=True)
        branch_name = branch_res.stdout.strip() if branch_res.returncode == 0 and branch_res.stdout.strip() else "detached HEAD"
        has_changes, files = get_git_status_porcelain()
        if not has_changes: 
            return f"On branch '{branch_name}'. Working tree clean."
        lines = [f"On branch '{branch_name}'."]
        staged: List[str] = []
        unstaged: List[str] = []
        untracked: List[str] = []
        for code, filename in files:
            if code == "??": 
                untracked.append(filename)
            elif code.startswith(" "): 
                unstaged.append(f"{code.strip()} {filename}")
            else: 
                staged.append(f"{code.strip()} {filename}")
        if staged: 
            lines.extend(["\nChanges to be committed:"] + [f"  {s}" for s in staged])
        if unstaged: 
            lines.extend(["\nChanges not staged for commit:"] + [f"  {s}" for s in unstaged])
        if untracked: 
            lines.extend(["\nUntracked files:"] + [f"  {f}" for f in untracked])
        return "\n".join(lines)
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        if isinstance(e, FileNotFoundError):
            git_context['enabled'] = False
        return f"Git status error: {e}"

def llm_change_directory(directory_path: str) -> str:
    """
    LLM tool handler for changing directory.
    
    Args:
        directory_path: Path to change to, or 'reset' to return to original directory
        
    Returns:
        Result message
    """
    import config
    from utils import get_directory_tree_summary
    
    if not directory_path:
        return f"Current working directory: {config.base_dir}"
    
    if directory_path.lower() == "reset":
        old_base = config.base_dir
        current_cwd = Path.cwd()
        config.base_dir = current_cwd
        return f"Directory reset from '{old_base}' to: '{config.base_dir}'"
    
    try:
        new_base = Path(directory_path).resolve()
        if not new_base.exists():
            return f"Error: Directory does not exist: '{directory_path}'"
        if not new_base.is_dir():
            return f"Error: Path is not a directory: '{directory_path}'"
        
        # Test write permissions
        test_file = new_base / ".eng-git-test"
        try:
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            return f"Error: No write permissions in directory: '{new_base}'"
        
        old_base = config.base_dir
        config.base_dir = new_base
        
        console.print(f"[green]✓ Working directory changed from '{old_base}' to: '{config.base_dir}'[/green]")
        console.print(f"[green]  All file and command operations will now use this directory.[/green]")
        
        # Get directory summary for the return message
        dir_summary = get_directory_tree_summary(config.base_dir)
        return f"Successfully changed working directory to: {config.base_dir}\n\nDirectory structure:\n{dir_summary}"
        
    except Exception as e:
        return f"Error changing directory: {e}"

def execute_function_call_dict(tool_call_dict: Dict[str, Any]) -> str:
    """
    Execute a function call from the LLM with enhanced fuzzy matching and security.
    
    Args:
        tool_call_dict: Dictionary containing function call information
        
    Returns:
        String result of the function execution
    """
    func_name = "unknown_function"
    try:
        func_name = tool_call_dict["function"]["name"]
        args = json.loads(tool_call_dict["function"]["arguments"])
        
        if func_name == "read_file":
            norm_path = normalize_path(args["file_path"])
            
            # Check file size before reading to prevent context overflow
            try:
                file_size = Path(norm_path).stat().st_size
                # Estimate tokens (roughly 4 chars per token)
                estimated_tokens = file_size // 4
                
                # Get model-specific context limit
                from config import get_max_tokens_for_model
                current_model = model_context.get('current_model', DEFAULT_MODEL)
                max_tokens = get_max_tokens_for_model(current_model)
                
                # Don't read files that would use more than 60% of context window
                max_file_tokens = int(max_tokens * 0.6)
                
                if estimated_tokens > max_file_tokens:
                    file_size_kb = file_size / 1024
                    return f"Error: File '{norm_path}' is too large ({file_size_kb:.1f}KB, ~{estimated_tokens} tokens) to read safely. Current model ({current_model}) has a context limit of {max_tokens} tokens. Maximum safe file size is ~{max_file_tokens} tokens ({(max_file_tokens * 4) / 1024:.1f}KB). Consider reading the file in smaller sections or using a different approach."
                    
            except OSError as e:
                return f"Error: Could not check file size for '{norm_path}': {e}"
            
            content = read_local_file(norm_path)
            console.print(f"[green]Successfully read file '{norm_path}' ({len(content)} characters)[/green]")
            return content
            
        elif func_name == "read_multiple_files":
            response_data = {
                "files_read": {},
                "errors": {}
            }
            total_content_size = 0
            
            # Get model-specific context limit for multiple files
            from config import get_max_tokens_for_model
            current_model = model_context.get('current_model', DEFAULT_MODEL)
            max_tokens = get_max_tokens_for_model(current_model)
            # Use smaller percentage for multiple files to be safer
            max_total_tokens = int(max_tokens * 0.4)
            max_total_size = max_total_tokens * 4  # Convert tokens back to character estimate

            for fp in args["file_paths"]:
                try:
                    norm_path = normalize_path(fp)
                    
                    # Check individual file size first
                    try:
                        file_size = Path(norm_path).stat().st_size
                        if file_size > max_total_size // 2:  # Individual file shouldn't be more than half the total budget
                            response_data["errors"][norm_path] = f"File too large ({file_size/1024:.1f}KB) for multiple file read operation."
                            continue
                    except OSError:
                        pass  # Continue with normal reading if size check fails
                    
                    content = read_local_file(norm_path)

                    if total_content_size + len(content) > max_total_size:
                        response_data["errors"][norm_path] = f"Could not read file, as total content size would exceed the safety limit ({max_total_size/1024:.1f}KB for model {current_model})."
                        continue

                    response_data["files_read"][norm_path] = content
                    total_content_size += len(content)
                    console.print(f"[green]✓ Read file '{norm_path}' ({len(content)} characters)[/green]")

                except (OSError, ValueError) as e:
                    # Use the original path in the error if normalization fails
                    error_key = str(base_dir / fp)
                    response_data["errors"][error_key] = str(e)
                    console.print(f"[red]✗ Failed to read '{fp}': {e}[/red]")

            # Show summary to user
            files_read_count = len(response_data["files_read"])
            errors_count = len(response_data["errors"])
            console.print(f"[blue]Read {files_read_count} files successfully, {errors_count} errors ({total_content_size} total characters)[/blue]")
            
            # Return a JSON string, which is much easier for the LLM to parse reliably
            return json.dumps(response_data, indent=2)
            
        elif func_name == "create_file": 
            create_file(args["file_path"], args["content"])
            return f"File '{args['file_path']}' created/updated."
            
        elif func_name == "create_multiple_files":
            created: List[str] = []
            errors: List[str] = []
            for f_info in args["files"]:
                try: 
                    create_file(f_info["path"], f_info["content"])
                    created.append(f_info["path"])
                except Exception as e: 
                    errors.append(f"Error creating {f_info.get('path','?path')}: {e}")
            res_parts = []
            if created: 
                res_parts.append(f"Created/updated {len(created)} files: {', '.join(created)}")
            if errors: 
                res_parts.append(f"Errors: {'; '.join(errors)}")
            return ". ".join(res_parts) if res_parts else "No files processed."
            
        elif func_name == "edit_file":
            fp = args["file_path"]
            # Check if file exists before editing
            if not Path(fp).exists():
                return f"Error: File '{fp}' does not exist."
            try: 
                apply_fuzzy_diff_edit(fp, args["original_snippet"], args["new_snippet"])
                return f"Edit applied successfully to '{fp}'. Check console for details."
            except Exception as e:
                return f"Error during edit_file call for '{fp}': {e}."
                
        elif func_name == "git_init": 
            return llm_git_init()
        elif func_name == "git_add": 
            return llm_git_add(args.get("file_paths", []))
        elif func_name == "git_commit": 
            return llm_git_commit(args.get("message", "Auto commit"))
        elif func_name == "git_create_branch": 
            return llm_git_create_branch(args.get("branch_name", ""))
        elif func_name == "git_status": 
            return llm_git_status()
        elif func_name == "run_powershell":
            command = args["command"]
            
            # SECURITY GATE
            if security_context["require_powershell_confirmation"]:
                console.print(Panel(
                    f"The assistant wants to run this PowerShell command:\n\n[bold yellow]{command}[/bold yellow]", 
                    title="🚨 Security Confirmation Required", 
                    border_style="red"
                ))
                confirm = prompt_session.prompt("🔵 Do you want to allow this command to run? (y/N): ", default="n").strip().lower()
                
                if confirm not in ["y", "yes"]:
                    console.print("[red]Execution denied by user.[/red]")
                    return "PowerShell command execution was denied by the user."
            
            output, error, returncode = run_powershell_command(command)
            if returncode != 0:
                return f"PowerShell command failed (exit code {returncode}):\nSTDOUT: {output}\nSTDERR: {error}"
            else:
                result_parts = []
                if output.strip():
                    result_parts.append(f"PowerShell Output:\n{output}")
                if error.strip():
                    result_parts.append(f"PowerShell Messages:\n{error}")
                return "\n".join(result_parts) if result_parts else "PowerShell Output:\nCommand completed successfully (no output)"
        elif func_name == "run_bash":
            command = args["command"]
            
            # SECURITY GATE
            if security_context["require_bash_confirmation"]:
                console.print(Panel(
                    f"The assistant wants to run this bash command:\n\n[bold yellow]{command}[/bold yellow]", 
                    title="🚨 Security Confirmation Required", 
                    border_style="red"
                ))
                confirm = prompt_session.prompt("🔵 Do you want to allow this command to run? (y/N): ", default="n").strip().lower()
                
                if confirm not in ["y", "yes"]:
                    console.print("[red]Execution denied by user.[/red]")
                    return "Bash command execution was denied by the user."
            
            output, error, returncode = run_bash_command(command)
            if returncode != 0:
                return f"Bash command failed (exit code {returncode}):\nSTDOUT: {output}\nSTDERR: {error}"
            else:
                result_parts = []
                if output.strip():
                    result_parts.append(f"Bash Output:\n{output}")
                if error.strip():
                    result_parts.append(f"Bash Messages:\n{error}")
                return "\n".join(result_parts) if result_parts else "Bash Output:\nCommand completed successfully (no output)"
        elif func_name == "change_directory":
            return llm_change_directory(args.get("directory_path", ""))
        else: 
            return f"Unknown LLM function: {func_name}"
            
    except json.JSONDecodeError as e: 
        console.print(f"[red]JSON Decode Error for {func_name}: {e}\nArgs: {tool_call_dict.get('function',{}).get('arguments','')}[/red]")
        return f"Error: Invalid JSON args for {func_name}."
    except KeyError as e: 
        console.print(f"[red]KeyError in {func_name}: Missing key {e}[/red]")
        return f"Error: Missing param for {func_name} (KeyError: {e})."
    except Exception as e: 
        console.print(f"[red]Unexpected Error in LLM func '{func_name}':[/red]")
        console.print_exception()
        return f"Unexpected error in {func_name}: {e}"

# =============================================================================
# MAIN LOOP & ENTRY POINT
# =============================================================================

def initialize_application() -> None:
    """Initialize the application and check for existing git repository."""
    # Detect available shells
    detect_available_shells()
    
    if (base_dir / ".git").exists() and (base_dir / ".git").is_dir():
        git_context['enabled'] = True
        try:
            res = subprocess.run(["git", "branch", "--show-current"], cwd=str(base_dir), capture_output=True, text=True, check=False)
            if res.returncode == 0 and res.stdout.strip(): 
                git_context['branch'] = res.stdout.strip()
            else:
                init_branch_res = subprocess.run(["git", "config", "init.defaultBranch"], cwd=str(base_dir), capture_output=True, text=True)
                git_context['branch'] = init_branch_res.stdout.strip() if init_branch_res.returncode == 0 and init_branch_res.stdout.strip() else "main"
        except FileNotFoundError: 
            console.print("[yellow]Git not found. Git features disabled.[/yellow]")
            git_context['enabled'] = False
        except Exception as e: 
            console.print(f"[yellow]Could not get Git branch: {e}.[/yellow]")

def main_loop() -> None:
    """Main application loop."""
    # Initialize conversation history
    conversation_history: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # Add initial context
    dir_summary = get_directory_tree_summary(base_dir)
    conversation_history.append({
        "role": "system",
        "content": f"Project directory structure at startup:\n\n{dir_summary}"
    })
    
    # Add OS and shell info
    shell_status = ", ".join([f"{shell}({'✓' if available else '✗'})" 
                             for shell, available in os_info['shell_available'].items()])
    conversation_history.append({
        "role": "system",
        "content": f"Runtime environment: {os_info['system']} {os_info['release']}, "
                  f"Python {os_info['python_version']}, Shells: {shell_status}"
    })

    while True:
        try:
            prompt_indicator = get_prompt_indicator(conversation_history, model_context['current_model'])
            user_input = prompt_session.prompt(f"{prompt_indicator} You: ")
            
            if not user_input.strip(): 
                continue

            # Handle commands
            if try_handle_add_command(user_input, conversation_history): continue
            if try_handle_git_add_command(user_input): continue
            if try_handle_commit_command(user_input): continue
            if try_handle_git_command(user_input): continue
            if try_handle_git_info_command(user_input): continue
            if try_handle_r1_command(user_input, conversation_history): continue
            if try_handle_reasoner_command(user_input): continue
            if try_handle_clear_command(user_input): continue
            if try_handle_clear_context_command(user_input, conversation_history): continue
            if try_handle_context_command(user_input, conversation_history): continue
            if try_handle_folder_command(user_input, conversation_history): continue
            if try_handle_os_command(user_input): continue
            if try_handle_log_command(user_input): continue
            if try_handle_exit_command(user_input): continue
            if try_handle_help_command(user_input): continue
            
            # Add user message to conversation
            conversation_history.append({"role": "user", "content": user_input})
            
            # Log user message
            current_model = model_context['current_model']
            log_user_message(user_input, current_model)
            
            # Determine which model to use
            model_name = "Gemini"
            
            # Check context usage and force truncation if needed
            context_info = get_context_usage_info(conversation_history, current_model)
            
            # Always truncate if we're over the limit (not just 95%)
            if context_info["estimated_tokens"] > context_info["max_tokens"] or context_info["token_usage_percent"] > 90:
                console.print(f"[red]🚨 Context exceeded ({context_info['estimated_tokens']} > {context_info['max_tokens']} tokens). Force truncating...[/red]")
                conversation_history = smart_truncate_history(conversation_history, model_name=current_model)
                context_info = get_context_usage_info(conversation_history, current_model)  # Recalculate after truncation
                console.print(f"[green]✓ Context truncated to {context_info['estimated_tokens']} tokens ({context_info['token_usage_percent']:.1f}% of limit)[/green]")
            elif context_info["critical_limit"] and len(conversation_history) % 10 == 0:
                console.print(f"[red]⚠ Context critical: {context_info['token_usage_percent']:.1f}% used. Consider /clear-context or /context for details.[/red]")
            elif context_info["approaching_limit"] and len(conversation_history) % 20 == 0:
                console.print(f"[yellow]⚠ Context high: {context_info['token_usage_percent']:.1f}% used. Use /context for details.[/yellow]")

            # Final safety check before API call
            final_context_info = get_context_usage_info(conversation_history, current_model)
            if final_context_info["estimated_tokens"] > final_context_info["max_tokens"]:
                console.print(f"[red]🚨 Final safety check failed: {final_context_info['estimated_tokens']} > {final_context_info['max_tokens']} tokens. Emergency truncation...[/red]")
                conversation_history = smart_truncate_history(conversation_history, model_name=current_model)
                final_context_info = get_context_usage_info(conversation_history, current_model)
                console.print(f"[green]✓ Emergency truncation complete: {final_context_info['estimated_tokens']} tokens[/green]")

            # Make API call with Gemini
            with console.status(f"[bold yellow]{model_name} is thinking...[/bold yellow]", spinner="dots"):
                is_reasoner_mode = model_context['is_reasoner']
                full_response_content, accumulated_tool_calls, api_response = call_gemini_api(conversation_history, current_model, is_reasoner_mode)
            
            # Log API response
            log_api_response(api_response, user_input, current_model, accumulated_tool_calls)

            # Display the response content
            console.print(f"[bold bright_magenta]🤖 {model_name}:[/bold bright_magenta] ", end="")
            if full_response_content:
                # Strip <think> and </think> tags from the content
                clean_content = full_response_content.replace("<think>", "").replace("</think>", "")
                console.print(clean_content, style="bright_magenta")
            else:
                console.print("[dim]No text response, checking for tool calls...[/dim]", style="bright_magenta")

            # Always add assistant message to maintain conversation flow
            assistant_message: Dict[str, Any] = {"role": "assistant"}
            assistant_message["content"] = full_response_content

            # Validate and add tool calls if any
            valid_tool_calls = validate_tool_calls(accumulated_tool_calls)
            if valid_tool_calls:
                assistant_message["tool_calls"] = valid_tool_calls
            
            # Always add the assistant message
            conversation_history.append(assistant_message)

            # Execute tool calls and allow assistant to continue naturally
            if valid_tool_calls:
                # Execute all tool calls first
                for tool_call_to_exec in valid_tool_calls: 
                    console.print(Panel(
                        f"[bold blue]Calling:[/bold blue] {tool_call_to_exec['function']['name']}\n"
                        f"[bold blue]Args:[/bold blue] {tool_call_to_exec['function']['arguments']}",
                        title="🛠️ Function Call", border_style="yellow", expand=False
                    ))
                    tool_output = execute_function_call_dict(tool_call_to_exec)
                    log_tool_execution(tool_call_to_exec, tool_output)
                    console.print(Panel(tool_output, title=f"↪️ Output of {tool_call_to_exec['function']['name']}", border_style="green", expand=False))
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_to_exec["id"],
                        "name": tool_call_to_exec["function"]["name"],
                        "content": tool_output 
                    })
                
                # Always allow the assistant to continue processing after tool calls
                # The model might provide text along with tool calls, and we should still let it continue
                should_continue = True
                
                if should_continue:
                    # Now let the assistant continue with the tool results
                    max_continuation_rounds = 5
                    current_round = 0
                    
                    while current_round < max_continuation_rounds:
                        current_round += 1
                        
                        with console.status(f"[bold yellow]{model_name} is processing results...[/bold yellow]", spinner="dots"):
                            is_reasoner_mode = model_context['is_reasoner']
                            continuation_content, continuation_tool_calls, continuation_response = call_gemini_api(conversation_history, current_model, is_reasoner_mode)

                        # Display the continuation content
                        console.print(f"[bold bright_magenta]🤖 {model_name}:[/bold bright_magenta] ", end="")
                        if continuation_content:
                            clean_content = continuation_content.replace("<think>", "").replace("</think>", "")
                            console.print(clean_content, style="bright_magenta")
                        else:
                            console.print("[dim]Continuing with tool calls...[/dim]", style="bright_magenta")
                        
                        # Add the continuation response to conversation history
                        continuation_message: Dict[str, Any] = {"role": "assistant", "content": continuation_content}
                        
                        # Check if there are more tool calls to execute
                        valid_continuation_tools = validate_tool_calls(continuation_tool_calls)
                        
                        # Filter out duplicate tool calls (same command executed in the last 3 messages only)
                        if valid_continuation_tools:
                            # Get the last few tool calls from conversation history to check for duplicates
                            recent_tool_calls = []
                            for msg in reversed(conversation_history[-3:]):  # Check last 3 messages only
                                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                                    recent_tool_calls.extend(msg["tool_calls"])
                            
                            # Remove exact duplicates only (same function and arguments)
                            filtered_tools = []
                            for tool in valid_continuation_tools:
                                is_duplicate = False
                                for recent_tool in recent_tool_calls:
                                    if (tool["function"]["name"] == recent_tool["function"]["name"] and 
                                        tool["function"]["arguments"] == recent_tool["function"]["arguments"]):
                                        is_duplicate = True
                                        break
                                if not is_duplicate:
                                    filtered_tools.append(tool)
                            
                            valid_continuation_tools = filtered_tools
                        
                        # Execute tool calls if present, regardless of text content
                        # The model can provide explanatory text along with tool calls
                        
                        if valid_continuation_tools:
                            continuation_message["tool_calls"] = valid_continuation_tools
                            conversation_history.append(continuation_message)
                            
                            # Execute the additional tool calls
                            for tool_call_to_exec in valid_continuation_tools:
                                console.print(Panel(
                                    f"[bold blue]Calling:[/bold blue] {tool_call_to_exec['function']['name']}\n"
                                    f"[bold blue]Args:[/bold blue] {tool_call_to_exec['function']['arguments']}",
                                    title="🛠️ Function Call", border_style="yellow", expand=False
                                ))
                                tool_output = execute_function_call_dict(tool_call_to_exec)
                                log_tool_execution(tool_call_to_exec, tool_output)
                                console.print(Panel(tool_output, title=f"↪️ Output of {tool_call_to_exec['function']['name']}", border_style="green", expand=False))
                                conversation_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call_to_exec["id"],
                                    "name": tool_call_to_exec["function"]["name"],
                                    "content": tool_output
                                })
                            
                            # Continue the loop to let assistant process these new results
                            continue
                        else:
                            # No more tool calls, add the final response and break
                            conversation_history.append(continuation_message)
                            break
                    
                    # If we hit the max rounds, warn about it
                    if current_round >= max_continuation_rounds:
                        console.print(f"[yellow]⚠ Reached maximum continuation rounds ({max_continuation_rounds}). Conversation continues.[/yellow]")
            
            # Smart truncation that preserves tool call sequences
            conversation_history = smart_truncate_history(conversation_history, model_name=current_model)

        except KeyboardInterrupt: 
            console.print("\n[yellow]⚠ Interrupted. Ctrl+D or /exit to quit.[/yellow]")
        except EOFError: 
            console.print("[blue]👋 Goodbye! (EOF)[/blue]")
            sys.exit(0)
        except Exception as e:
            console.print(f"\n[red]✗ Unexpected error in main loop:[/red]")
            console.print_exception(width=None, extra_lines=1, show_locals=True)

def main() -> None:
    """Application entry point."""
    console.print(Panel.fit(
        "[bold bright_blue] Gemini Engineer - Enhanced Edition[/bold bright_blue]\n"
        "[dim]✨ Now with fuzzy matching for files and cross-platform shell support![/dim]\n"
        "[dim]Type /help for commands. Ctrl+C to interrupt, Ctrl+D or /exit to quit.[/dim]",
        border_style="bright_blue"
    ))

    # Show fuzzy matching status on startup
    if FUZZY_AVAILABLE:
        console.print("[green]✓ Fuzzy matching enabled for intelligent file finding and code editing[/green]")
    else:
        console.print("[yellow]⚠ Fuzzy matching disabled. Install with: pip install thefuzz python-levenshtein[/yellow]")

    # Initialize application first (detects git repo and shells)
    initialize_application()
    
    # Show detected shells
    available_shells = [shell for shell, available in os_info['shell_available'].items() if available]
    if available_shells:
        console.print(f"[green]✓ Detected shells: {', '.join(available_shells)}[/green]")
    else:
        console.print("[yellow]⚠ No supported shells detected[/yellow]")
    
    # Start the main loop
    main_loop()

if __name__ == "__main__":
    main()


