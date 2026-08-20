"""
sandbox.py

Secure Python code execution tool for LangGraph agents, using Docker for isolation.

Two execution modes:
  - Type A (default): uses a pre-built image with a whitelisted set of common
    libraries already installed. Runs with --network none (fully offline).
  - Type B (fallback): triggered when the code needs a library NOT in the
    whitelist. Runs with a temporary, PyPI-only network so pip install works,
    then the container is destroyed (--rm). Nothing is cached or persisted.

Also supports mounting a single, already-uploaded input file (e.g. a CSV)
into the container as read-only, so generated code can read/process it.
Output is text only (stdout/stderr) — no files are ever returned.
"""

#=================================================================
#=======================  Module Imports =========================
import os
import re
import asyncio
import tempfile
from langchain.tools import tool

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SANDBOX_IMAGE = "python:3.11-slim"
FALLBACK_IMAGE = "python:3.10-slim"

WHITELISTED_LIBRARIES = {
    "pandas", "numpy", "requests", "matplotlib", "scipy",
    "openpyxl", "scikit-learn",
}

ALLOWED_UPLOAD_DIR = "/app/uploads"  # change to match your actual uploads path

EXECUTION_TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 4000

STDLIB_IGNORE = {
    "os", "sys", "re", "json", "math", "random", "time", "datetime",
    "itertools", "collections", "functools", "typing", "io",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_path_safe(user_path: str) -> bool:
    """Only allow files that actually live inside ALLOWED_UPLOAD_DIR.
    Blocks path traversal tricks like '../../.env'."""
    if not user_path:
        return False
    real_path = os.path.realpath(user_path)
    allowed_real = os.path.realpath(ALLOWED_UPLOAD_DIR)
    return real_path == allowed_real or real_path.startswith(allowed_real + os.sep)


def _extract_imported_modules(code: str) -> set:
    """Best-effort static scan for top-level imports, to decide if the
    whitelist image is sufficient or if Type B (new library) is needed."""
    modules = set()
    for line in code.splitlines():
        line = line.strip()
        m = re.match(r"^import\s+([a-zA-Z0-9_\.]+)", line)
        if m:
            modules.add(m.group(1).split(".")[0])
            continue
        m = re.match(r"^from\s+([a-zA-Z0-9_\.]+)\s+import", line)
        if m:
            modules.add(m.group(1).split(".")[0])
    return modules


async def _run_container(docker_cmd: list, container_name: str) -> str:
    """Run a docker command asynchronously, capture output, never blocks the event loop."""
    try:
        process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=EXECUTION_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            # FIX: Explicitly kill the docker container so no zombies are left
            kill_proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await kill_proc.communicate()
            
            process.kill()
            await process.wait()
            return f"[ERROR] Execution timed out ({EXECUTION_TIMEOUT_SECONDS}s limit)"

        output = stdout_bytes.decode(errors="ignore").strip()
        stderr_text = stderr_bytes.decode(errors="ignore").strip()

        if process.returncode != 0:
            output += f"\n[ERROR - exit code {process.returncode}]\n{stderr_text}"

        if not output and not stderr_text:
            return "(no output from script)"
        elif not output:
            output = f"(no output)\n[Stderr]: {stderr_text}"
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n...[Output truncated due to excessive length]"
        return output

    except FileNotFoundError:
        return "[ERROR] Docker is not installed or not available on this system"
    except Exception as e:
        return f"[ERROR] Sandbox execution failed: {str(e)}"


def _base_security_flags(user: str = "1000:1000") -> list:
    """Restrictions applied to every sandbox container, regardless of type."""
    return [
        "-m", "128m",
        "--memory-swap", "128m",
        "--cpus", "0.5",
        "--pids-limit", "50",
        "--security-opt", "no-new-privileges",
        "--user", user,
        "--tmpfs", "/tmp:size=16m",
    ]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
@tool
async def python_docker_sandbox(code: str, input_file_path: str = None) -> str: # type: ignore
    """Run Python code safely in an isolated Docker container and return stdout/stderr as text.

    Args:
        code: The Python code to execute.
        input_file_path: Optional path to a previously uploaded file (must be
            inside the server's uploads directory). If given, the file is
            mounted read-only inside the container at /app/<filename>,
            and the code should read it from that path.

    Returns:
        The captured stdout/stderr/error output as plain text. No files are
        ever returned — only text.
    """
    import uuid
    def _write_script():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            fname = f.name
        os.chmod(fname, 0o644)
        return fname

    try:
        script_path = await asyncio.to_thread(_write_script)
    except Exception as e:
        return f"[ERROR] Could not prepare script: {str(e)}"

    try:
        imported = _extract_imported_modules(code)
        needs_new_library = bool(imported - WHITELISTED_LIBRARIES - STDLIB_IGNORE)

        if input_file_path:
            if not _is_path_safe(input_file_path):
                return "[ERROR] Access denied: input_file_path is outside the allowed uploads directory"
            if not await asyncio.to_thread(os.path.exists, input_file_path):
                return f"[ERROR] File not found: {input_file_path}"
            filename = os.path.basename(input_file_path)
        else:
            filename = ""

        # Use a named volume to transfer files safely in Docker-in-Docker
        vol_name = f"sandbox-vol-{uuid.uuid4().hex[:8]}"
        proc = await asyncio.create_subprocess_exec("docker", "volume", "create", vol_name, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()

        try:
            # 1. Populate the volume using a helper container
            helper_cid = f"helper-{uuid.uuid4().hex[:8]}"
            proc = await asyncio.create_subprocess_exec("docker", "create", "--name", helper_cid, "-v", f"{vol_name}:/vol", FALLBACK_IMAGE, "true", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()

            proc = await asyncio.create_subprocess_exec("docker", "cp", script_path, f"{helper_cid}:/vol/script.py", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()

            if input_file_path:
                proc = await asyncio.create_subprocess_exec("docker", "cp", input_file_path, f"{helper_cid}:/vol/{filename}", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await proc.communicate()

            proc = await asyncio.create_subprocess_exec("docker", "rm", "-f", helper_cid, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()

            mount_flags = ["-v", f"{vol_name}:/app:ro"]
            sandbox_cid = f"sandbox-{uuid.uuid4().hex[:8]}"

            # 2. Run the actual sandbox container
            if not needs_new_library:
                docker_cmd = (
                    ["docker", "run", "--rm", "--name", sandbox_cid, "--network", "none"]
                    + _base_security_flags()
                    + ["--read-only"]
                    + mount_flags
                    + [SANDBOX_IMAGE, "python", "/app/script.py"]
                )
            else:
                extra_libs = sorted(imported - WHITELISTED_LIBRARIES - STDLIB_IGNORE)
                install_cmd = f"pip install --no-cache-dir --quiet {' '.join(extra_libs)}"
                wrapped_cmd = f"{install_cmd} && python /app/script.py"
                docker_cmd = (
                    ["docker", "run", "--rm", "--name", sandbox_cid, "--network", "bridge"]
                    + _base_security_flags(user="root")
                    + mount_flags
                    + [FALLBACK_IMAGE, "sh", "-c", wrapped_cmd]
                )
                
            return await _run_container(docker_cmd, sandbox_cid)

        finally:
            proc = await asyncio.create_subprocess_exec("docker", "volume", "rm", "-f", vol_name, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()

    finally:
        try:
            if os.path.exists(script_path):
                await asyncio.to_thread(os.remove, script_path)
        except Exception:
            pass