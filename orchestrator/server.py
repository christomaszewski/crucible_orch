"""Stack orchestrator — WebSocket server for managing agent stacks.

Runs as a standalone service (not a ROS2 node) since it only needs
Docker socket access and WebSocket connectivity to the frontend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal

import websockets

from orchestrator.compose_manager import ComposeManager

logger = logging.getLogger(__name__)

ORCH_PORT = 9091


class OrchestratorServer:
    """WebSocket server for stack lifecycle management."""

    def __init__(self) -> None:
        self._manager = ComposeManager()
        self._clients: set = set()

    async def handler(self, ws) -> None:
        self._clients.add(ws)
        logger.info("Orchestrator client connected")
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    await self._handle(ws, data)
                except json.JSONDecodeError:
                    await ws.send(
                        json.dumps({"type": "error", "message": "Invalid JSON"})
                    )
        finally:
            self._clients.discard(ws)
            logger.info("Orchestrator client disconnected")

    async def _handle(self, ws, data: dict) -> None:
        cmd = data.get("cmd")

        if cmd == "launch_stack":
            if not data.get("agent_name") or not data.get("compose_file"):
                await ws.send(json.dumps({
                    "type": "error",
                    "message": "launch_stack requires 'agent_name' and 'compose_file'",
                }))
                return
            info = self._manager.launch(
                agent_name=data["agent_name"],
                compose_file=data["compose_file"],
                env=data.get("env", {}),
            )
            await ws.send(
                json.dumps({
                    "type": "stack_update",
                    "agent_name": info.agent_name,
                    "status": info.status.name,
                })
            )
            # Poll until status settles, then send final update
            asyncio.create_task(self._poll_status(data["agent_name"]))

        elif cmd == "stop_stack":
            if not data.get("agent_name"):
                await ws.send(json.dumps({
                    "type": "error",
                    "message": "stop_stack requires 'agent_name'",
                }))
                return
            info = self._manager.stop(data["agent_name"])
            status = info.status.name if info else "UNKNOWN"
            await ws.send(
                json.dumps({
                    "type": "stack_update",
                    "agent_name": data["agent_name"],
                    "status": status,
                })
            )
            if info:
                asyncio.create_task(self._poll_status(data["agent_name"]))

        elif cmd == "get_stack_status":
            statuses = self._manager.get_all_status()
            await ws.send(
                json.dumps({"type": "stack_status", "stacks": statuses})
            )

        else:
            await ws.send(
                json.dumps({"type": "error", "message": f"Unknown command: {cmd}"})
            )

    async def _poll_status(self, agent_name: str, max_wait: float = 130.0) -> None:
        """Poll stack status until it settles, then broadcast the result."""
        import time

        start = time.monotonic()
        while time.monotonic() - start < max_wait:
            await asyncio.sleep(2.0)
            info = self._manager.get_status(agent_name)
            if info is None:
                break
            if info.status.name in ("RUNNING", "STOPPED", "ERROR"):
                msg = json.dumps({
                    "type": "stack_update",
                    "agent_name": agent_name,
                    "status": info.status.name,
                    "error": info.error_message,
                })
                for client in list(self._clients):
                    try:
                        await client.send(msg)
                    except Exception:
                        self._clients.discard(client)
                break

    async def run(self) -> None:
        logger.info("Stack orchestrator listening on port %d", ORCH_PORT)
        async with websockets.serve(self.handler, "0.0.0.0", ORCH_PORT):
            await asyncio.Future()

    def shutdown(self) -> None:
        logger.info("Shutting down — stopping all stacks")
        self._manager.stop_all()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    server = OrchestratorServer()

    loop = asyncio.new_event_loop()

    def handle_signal():
        server.shutdown()
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    try:
        loop.run_until_complete(server.run())
    except Exception:
        server.shutdown()


if __name__ == "__main__":
    main()
