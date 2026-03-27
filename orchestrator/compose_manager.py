"""Docker Compose lifecycle manager for agent software stacks."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class StackStatus(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()


@dataclass
class StackInfo:
    """Tracks the state of a launched agent stack."""

    agent_name: str
    compose_file: str
    env: dict[str, str] = field(default_factory=dict)
    status: StackStatus = StackStatus.STOPPED
    error_message: str = ""
    project_name: str = ""

    def __post_init__(self) -> None:
        if not self.project_name:
            # Docker Compose project name — unique per agent
            self.project_name = f"sim_{self.agent_name}"


class ComposeManager:
    """Manages Docker Compose stacks for simulated agents.

    Each agent can have an associated Compose file that defines its full
    software stack (estimator, autopilot, Zenoh bridge, etc.). This manager
    handles launching, stopping, and monitoring those stacks.

    Requires the Docker socket to be mounted into the orchestrator container.
    """

    def __init__(self) -> None:
        self._stacks: dict[str, StackInfo] = {}
        self._lock = threading.Lock()

    def launch(
        self,
        agent_name: str,
        compose_file: str,
        env: dict[str, str] | None = None,
    ) -> StackInfo:
        """Launch a Docker Compose stack for an agent.

        Args:
            agent_name: The agent this stack belongs to.
            compose_file: Path to the docker-compose YAML file.
            env: Environment variables to pass to the stack.

        Returns:
            StackInfo with current status.
        """
        with self._lock:
            if agent_name in self._stacks:
                existing = self._stacks[agent_name]
                if existing.status in (StackStatus.RUNNING, StackStatus.STARTING):
                    logger.warning(
                        "Stack for %s already %s", agent_name, existing.status.name
                    )
                    return existing

            info = StackInfo(
                agent_name=agent_name,
                compose_file=compose_file,
                env=env or {},
                status=StackStatus.STARTING,
            )
            self._stacks[agent_name] = info

        # Launch in background thread to avoid blocking
        thread = threading.Thread(
            target=self._do_launch, args=(info,), daemon=True
        )
        thread.start()
        return info

    def stop(self, agent_name: str) -> StackInfo | None:
        """Stop and remove a running agent stack."""
        with self._lock:
            info = self._stacks.get(agent_name)
            if info is None:
                logger.warning("No stack info found for %s", agent_name)
                return None
            if info.status == StackStatus.STOPPED:
                logger.info("Stack for %s already stopped", agent_name)
                return info
            logger.info(
                "Requesting stop for %s (current status: %s)",
                agent_name, info.status.name,
            )
            info.status = StackStatus.STOPPING

        thread = threading.Thread(
            target=self._do_stop, args=(info,), daemon=True
        )
        thread.start()
        return info

    def get_status(self, agent_name: str) -> StackInfo | None:
        with self._lock:
            return self._stacks.get(agent_name)

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Return status of all known stacks as a serializable dict."""
        with self._lock:
            return {
                aid: {
                    "status": info.status.name,
                    "compose_file": info.compose_file,
                    "error": info.error_message,
                }
                for aid, info in self._stacks.items()
            }

    def stop_all(self) -> None:
        """Stop all running stacks. Used during shutdown."""
        with self._lock:
            running = [
                info
                for info in self._stacks.values()
                if info.status == StackStatus.RUNNING
            ]
        for info in running:
            self._do_stop(info)

    # -- Internal ------------------------------------------------------------

    def _do_launch(self, info: StackInfo) -> None:
        """Execute docker compose up in a subprocess."""
        try:
            cmd = [
                "docker", "compose",
                "-f", info.compose_file,
                "-p", info.project_name,
                "up", "-d",
                "--remove-orphans",
            ]

            logger.info("Launching stack for %s: %s", info.agent_name, " ".join(cmd))

            # Build environment with agent-specific vars
            run_env = os.environ.copy()
            run_env.update(info.env)

            result = subprocess.run(
                cmd,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                with self._lock:
                    info.status = StackStatus.RUNNING
                logger.info("Stack for %s is running", info.agent_name)
            else:
                with self._lock:
                    info.status = StackStatus.ERROR
                    info.error_message = result.stderr[:500]
                logger.error(
                    "Failed to launch stack for %s: %s",
                    info.agent_name,
                    result.stderr[:200],
                )
        except subprocess.TimeoutExpired:
            with self._lock:
                info.status = StackStatus.ERROR
                info.error_message = "Launch timed out after 120s"
            logger.error("Stack launch timed out for %s", info.agent_name)
        except Exception as e:
            with self._lock:
                info.status = StackStatus.ERROR
                info.error_message = str(e)
            logger.exception("Error launching stack for %s", info.agent_name)

    def _do_stop(self, info: StackInfo) -> None:
        """Execute docker compose down in a subprocess."""
        try:
            cmd = [
                "docker", "compose",
                "-f", info.compose_file,
                "-p", info.project_name,
                "down",
                "--remove-orphans",
                "--timeout", "10",
            ]

            logger.info("Stopping stack for %s", info.agent_name)

            run_env = os.environ.copy()
            run_env.update(info.env)

            result = subprocess.run(
                cmd,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=30,
            )

            with self._lock:
                if result.returncode == 0:
                    info.status = StackStatus.STOPPED
                    info.error_message = ""
                    logger.info("Stack for %s stopped", info.agent_name)
                else:
                    info.status = StackStatus.ERROR
                    info.error_message = result.stderr[:500]
                    logger.error(
                        "Error stopping stack for %s: %s",
                        info.agent_name,
                        result.stderr[:200],
                    )
        except subprocess.TimeoutExpired:
            # Compose didn't finish in time — force-kill via docker compose kill
            logger.warning("Stop timed out for %s, force-killing", info.agent_name)
            try:
                subprocess.run(
                    ["docker", "compose", "-p", info.project_name, "kill"],
                    capture_output=True, text=True, timeout=15,
                )
                subprocess.run(
                    ["docker", "compose", "-p", info.project_name, "down", "--remove-orphans", "--timeout", "0"],
                    capture_output=True, text=True, timeout=15,
                )
                with self._lock:
                    info.status = StackStatus.STOPPED
                    info.error_message = ""
                    logger.info("Force-stopped stack for %s", info.agent_name)
            except Exception as kill_err:
                with self._lock:
                    info.status = StackStatus.ERROR
                    info.error_message = f"Force-kill failed: {kill_err}"
                logger.exception("Force-kill failed for %s", info.agent_name)
        except Exception as e:
            with self._lock:
                info.status = StackStatus.ERROR
                info.error_message = str(e)
            logger.exception("Error stopping stack for %s", info.agent_name)
