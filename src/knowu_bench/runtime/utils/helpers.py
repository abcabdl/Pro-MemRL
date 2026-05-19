import copy
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timedelta

from loguru import logger
from pydantic import BaseModel


class AdbResponse(BaseModel):
    """Response model for ADB command execution."""

    success: bool
    output: str = ""
    error: str = ""
    return_code: int = 0
    command: str = ""

    def __str__(self) -> str:
        """Return output string for backward compatibility."""
        return self.output if self.success else "ERROR"

    def __bool__(self) -> bool:
        """Allow boolean checks for success."""
        return self.success

    def __eq__(self, other: object) -> bool:
        """Support comparison with 'ERROR' string for backward compatibility."""
        if isinstance(other, str):
            if other == "ERROR":
                return not self.success
            return self.output == other
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        """Support != comparison."""
        return not self.__eq__(other)


def time_within_ten_secs(time1: str | AdbResponse, time2: str | AdbResponse):
    """Compare two time strings or AdbResponse objects to check if within 10 seconds."""

    def parse_time(t: str | AdbResponse):
        if isinstance(t, AdbResponse):
            if not t.success:
                raise ValueError(f"Cannot parse time from failed command: {t.error}")
            t_str = t.output
        else:
            t_str = t

        if "+" in t_str:
            t_str = t_str.split()[1]
            t_str = t_str.split(".")[0] + "." + t_str.split(".")[1][:6]  # 仅保留到微秒
            format = "%H:%M:%S.%f"
        else:
            format = "%H:%M:%S"
        return datetime.strptime(t_str, format)

    # 解析两个时间
    time1_parsed = parse_time(time1)
    time2_parsed = parse_time(time2)

    # 计算时间差并判断
    time_difference = abs(time1_parsed - time2_parsed)

    return time_difference <= timedelta(seconds=10)


def pretty_print_messages(messages: list[dict], max_messages: int = 2) -> None:
    """
    Pretty print messages with base64 images replaced and limiting to recent messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content' fields
        max_messages: Maximum number of recent messages to display (default: 2)
    """

    messages_print = copy.deepcopy(messages)

    final_str = ""

    if len(messages_print) > max_messages:
        omitted_count = len(messages_print) - max_messages
        messages_print = messages_print[-max_messages:]
        final_str += f"\n[... {omitted_count} earlier message(s) omitted ...]\n"

    for message in messages_print:
        if "content" in message and isinstance(message["content"], list):
            for content_item in message["content"]:
                if isinstance(content_item, dict):
                    if "image_url" in content_item and "url" in content_item["image_url"]:
                        url = content_item["image_url"]["url"]
                        if url.startswith("data:image/") and "base64," in url:
                            content_item["image_url"]["url"] = "[IMAGE_BASE64]"

    final_str += f"messages:\n{json.dumps(messages_print, indent=2, ensure_ascii=False)}"
    logger.info(final_str)


_ADB_CONNECTED_TARGETS: set[str] = set()


def _run_command(command: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _adb_available(env: dict[str, str]) -> bool:
    result = _run_command("adb version", env)
    return result.returncode == 0


def _docker_env_container(env: dict[str, str]) -> str | None:
    explicit = os.getenv("KNOWU_ENV_CONTAINER") or os.getenv("MOBILE_WORLD_CONTAINER")
    if explicit:
        return explicit.strip()
    result = _run_command("docker ps --format \"{{.Names}}\"", env)
    if result.returncode != 0:
        return None
    for name in result.stdout.splitlines():
        if name.startswith("knowu_bench_env_"):
            return name.strip()
    return None


def _dockerize_adb_command(adb_command: str, env: dict[str, str]) -> str | None:
    container = _docker_env_container(env)
    if not container:
        return None
    command = adb_command
    if not command.startswith("adb "):
        command = "adb " + command
    command = re.sub(r"\s-s\s+127\.0\.0\.1:\d+", " -s emulator-5554", command, count=1)
    command = re.sub(r"\s-s\s+localhost:\d+", " -s emulator-5554", command, count=1)
    if " -s " not in command:
        command = command.replace("adb ", "adb -s emulator-5554 ", 1)
    escaped = command.replace("\\", "\\\\").replace('"', '\\"')
    return f'docker exec {container} bash -lc "{escaped}"'


def _execute_adb_via_docker(adb_command: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    container = _docker_env_container(env)
    if not container:
        return _run_command("__missing_knowu_container__", env)

    command = adb_command if adb_command.startswith("adb ") else "adb " + adb_command
    command = re.sub(r"\s-s\s+127\.0\.0\.1:\d+", " -s emulator-5554", command, count=1)
    command = re.sub(r"\s-s\s+localhost:\d+", " -s emulator-5554", command, count=1)
    if " -s " not in command:
        command = command.replace("adb ", "adb -s emulator-5554 ", 1)

    push_match = re.match(r"adb\s+(?:-s\s+\S+\s+)?push\s+(.+?)\s+(\S+)\s*$", command)
    if push_match:
        local_path = push_match.group(1).strip().strip('"').strip("'")
        remote_path = push_match.group(2).strip()
        container_tmp = f"/tmp/knowu_adb_push_{os.path.basename(local_path)}"
        copy_result = subprocess.run(
            ["docker", "cp", local_path, f"{container}:{container_tmp}"],
            capture_output=True,
            text=True,
            env=env,
        )
        if copy_result.returncode != 0:
            return copy_result
        device_match = re.search(r"\s-s\s+(\S+)", command)
        device = device_match.group(1) if device_match else "emulator-5554"
        return subprocess.run(
            [
                "docker",
                "exec",
                container,
                "adb",
                "-s",
                device,
                "push",
                container_tmp,
                remote_path,
            ],
            capture_output=True,
            text=True,
            env=env,
        )

    pull_match = re.match(r"adb\s+(?:-s\s+\S+\s+)?pull\s+(\S+)\s+(.+?)\s*$", command)
    if pull_match:
        remote_path = pull_match.group(1).strip()
        local_path = pull_match.group(2).strip().strip('"').strip("'")
        container_tmp = f"/tmp/knowu_adb_pull_{os.path.basename(remote_path)}"
        device_match = re.search(r"\s-s\s+(\S+)", command)
        device = device_match.group(1) if device_match else "emulator-5554"
        pull_result = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "adb",
                "-s",
                device,
                "pull",
                remote_path,
                container_tmp,
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if pull_result.returncode != 0:
            return pull_result
        copy_result = subprocess.run(
            ["docker", "cp", f"{container}:{container_tmp}", local_path],
            capture_output=True,
            text=True,
            env=env,
        )
        return copy_result if copy_result.returncode != 0 else pull_result

    try:
        argv = shlex.split(command)
    except ValueError:
        argv = ["bash", "-lc", command]
    return subprocess.run(
        ["docker", "exec", container, *argv],
        capture_output=True,
        text=True,
        env=env,
    )


def _list_adb_devices(env: dict[str, str]) -> list[str]:
    result = _run_command("adb devices", env)
    if result.returncode != 0:
        return []
    devices = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def _docker_adb_target(env: dict[str, str]) -> str | None:
    explicit = (
        os.getenv("KNOWU_MEMRL_ADB_DEVICE")
        or os.getenv("KNOWU_ADB_DEVICE")
        or os.getenv("ANDROID_SERIAL")
    )
    if explicit:
        return explicit.strip()

    aw_host = os.getenv("KNOWU_AW_HOST") or os.getenv("AW_HOST") or os.getenv("MOBILE_WORLD_HOST")
    if aw_host:
        match = re.search(r":(\d+)(?:/|$)", aw_host)
        if match:
            port = int(match.group(1))
            if 6800 <= port < 6900:
                return f"127.0.0.1:{port - 1244}"

    result = _run_command("docker ps --format \"{{.Ports}}\"", env)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        match = re.search(r"0\.0\.0\.0:(55\d\d)->5556/tcp", line)
        if match:
            return f"127.0.0.1:{match.group(1)}"
    return None


def _ensure_adb_target_connected(device: str, env: dict[str, str]) -> bool:
    if not device or ":" not in device:
        return True
    if device in _ADB_CONNECTED_TARGETS and device in _list_adb_devices(env):
        return True
    result = _run_command(f"adb connect {device}", env)
    if result.returncode == 0 and (
        "connected" in result.stdout.lower()
        or "already connected" in result.stdout.lower()
        or device in _list_adb_devices(env)
    ):
        _ADB_CONNECTED_TARGETS.add(device)
        return True
    logger.warning("Failed to connect adb target {}: {}", device, result.stderr or result.stdout)
    return False


def _rewrite_adb_command(adb_command: str, env: dict[str, str]) -> str:
    if not adb_command.startswith("adb "):
        adb_command = "adb " + adb_command

    if re.search(r"\s-s\s+\S+", adb_command):
        match = re.search(r"\s-s\s+(\S+)", adb_command)
        device = match.group(1) if match else ""
        if device == "emulator-5554" and device not in _list_adb_devices(env):
            target = _docker_adb_target(env)
            if target:
                _ensure_adb_target_connected(target, env)
                return re.sub(r"\s-s\s+\S+", f" -s {target}", adb_command, count=1)
        _ensure_adb_target_connected(device, env)
        return adb_command

    devices = _list_adb_devices(env)
    if len(devices) == 1:
        return adb_command

    target = _docker_adb_target(env)
    if target and _ensure_adb_target_connected(target, env):
        return adb_command.replace("adb ", f"adb -s {target} ", 1)

    return adb_command


def _adb_root_command(device: str | None, subcommand: str) -> str:
    if device:
        return f"adb -s {device} {subcommand}"
    return f"adb {subcommand}"


def _extract_adb_device(adb_command: str) -> str | None:
    match = re.search(r"\s-s\s+(\S+)", adb_command)
    return match.group(1) if match else None


def execute_adb(adb_command: str, output: bool = True, root_required=False) -> AdbResponse:
    env = os.environ.copy()
    host_adb_available = _adb_available(env)
    if not host_adb_available:
        docker_command = _dockerize_adb_command(adb_command, env)
        if not docker_command:
            message = "adb executable not found in PATH and no KnowU Docker container is available"
            if output:
                logger.error(message)
            return AdbResponse(success=False, error=message, return_code=127, command=adb_command)
        result = _execute_adb_via_docker(adb_command, env)
        if result.returncode == 0:
            return AdbResponse(
                success=True,
                output=result.stdout.strip(),
                return_code=result.returncode,
                command=docker_command,
            )
        if output:
            logger.error(f"Command execution failed: {docker_command}")
            logger.error(result.stderr)
        return AdbResponse(
            success=False,
            error=result.stderr or "Command execution failed",
            return_code=result.returncode,
            command=docker_command,
        )

    adb_command = _rewrite_adb_command(adb_command, env)
    device = _extract_adb_device(adb_command)

    if root_required:
        whoami_check = _run_command(_adb_root_command(device, "shell whoami"), env)
        if whoami_check.returncode == 0 and whoami_check.stdout.strip() != "root":
            root_attempt = _run_command(_adb_root_command(device, "root"), env)
            if root_attempt.returncode != 0:
                if output:
                    logger.error("Failed to gain root access to the emulator")
                    logger.error(root_attempt.stderr)
                return AdbResponse(
                    success=False,
                    error=root_attempt.stderr or "Failed to gain root access",
                    return_code=root_attempt.returncode,
                    command=adb_command,
                )

            verify_check = _run_command(_adb_root_command(device, "shell whoami"), env)
            if verify_check.returncode != 0 or verify_check.stdout.strip() != "root":
                if output:
                    logger.error("Root permission required but not available on the emulator")
                return AdbResponse(
                    success=False,
                    error="Root permission required but not available on the emulator",
                    return_code=verify_check.returncode,
                    command=adb_command,
                )

    result = _run_command(adb_command, env)
    if result.returncode == 0:
        return AdbResponse(
            success=True,
            output=result.stdout.strip(),
            return_code=result.returncode,
            command=adb_command,
        )
    if output:
        logger.error(f"Command execution failed: {adb_command}")
        logger.error(result.stderr)
    return AdbResponse(
        success=False,
        error=result.stderr or "Command execution failed",
        return_code=result.returncode,
        command=adb_command,
    )


def execute_root_sql(db_path: str, sql_query: str) -> str:
    """
    Execute a SQL query that requires root access.
    """

    adb_commands = [
        f"adb shell \"su 0 sqlite3 {db_path} '{sql_query}'\"",
        f"adb shell \"su root sqlite3 {db_path} '{sql_query}'\"",
        f'adb shell su 0 sqlite3 {db_path} "{sql_query}"',
    ]

    for adb_command in adb_commands:
        result = execute_adb(adb_command, output=False)
        if result.success and result.output and "error" not in result.output.lower():
            return result.output

    return None
