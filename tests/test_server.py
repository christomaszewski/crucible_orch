"""Tests for orchestrator.server — WebSocket command handling."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.compose_manager import StackInfo, StackStatus
from orchestrator.server import OrchestratorServer


class _FakeWS:
    """Minimal async-iterable WebSocket mock."""

    def __init__(self, messages=None):
        self.messages = list(messages or [])
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        return self.messages.pop(0)


@pytest.fixture
def server():
    with patch.dict("os.environ", {"STACKS_HOST_PATH": "/host/stacks"}):
        srv = OrchestratorServer()
    return srv


@pytest.fixture
def mock_ws():
    ws = AsyncMock()
    ws.send = AsyncMock()
    return ws


# ── Connection ──────────────────────────────────────────────

class TestConnection:
    @pytest.mark.asyncio
    async def test_sends_config_on_connect(self, server):
        ws = _FakeWS()
        await server.handler(ws)

        first_msg = json.loads(ws.sent[0])
        assert first_msg["type"] == "orch_config"
        assert first_msg["stacks_host_path"] == "/host/stacks"
        assert first_msg["stacks_container_path"] == "/opt/stacks"

    @pytest.mark.asyncio
    async def test_client_added_and_removed(self, server):
        ws = _FakeWS()
        await server.handler(ws)
        # After disconnect, client should be removed
        assert ws not in server._clients

    @pytest.mark.asyncio
    async def test_invalid_json(self, server):
        ws = _FakeWS(messages=["not json"])
        await server.handler(ws)

        # Should have sent config + error
        error_msg = json.loads(ws.sent[-1])
        assert error_msg["type"] == "error"
        assert "Invalid JSON" in error_msg["message"]


# ── Command routing ─────────────────────────────────────────

class TestHandleLaunchStack:
    @pytest.mark.asyncio
    async def test_missing_fields(self, server, mock_ws):
        await server._handle(mock_ws, {"cmd": "launch_stack"})
        msg = json.loads(mock_ws.send.call_args[0][0])
        assert msg["type"] == "error"
        assert "agent_name" in msg["message"]

    @pytest.mark.asyncio
    async def test_missing_compose_file(self, server, mock_ws):
        await server._handle(mock_ws, {"cmd": "launch_stack", "agent_name": "uav_01"})
        msg = json.loads(mock_ws.send.call_args[0][0])
        assert msg["type"] == "error"

    @pytest.mark.asyncio
    async def test_success(self, server, mock_ws):
        info = StackInfo(
            agent_name="uav_01", compose_file="x.yml", status=StackStatus.STARTING
        )
        with patch.object(server._manager, "launch", return_value=info):
            with patch("orchestrator.server.asyncio.create_task"):
                await server._handle(
                    mock_ws,
                    {
                        "cmd": "launch_stack",
                        "agent_name": "uav_01",
                        "compose_file": "x.yml",
                        "env": {"FOO": "bar"},
                    },
                )
        msg = json.loads(mock_ws.send.call_args[0][0])
        assert msg["type"] == "stack_update"
        assert msg["agent_name"] == "uav_01"
        assert msg["status"] == "STARTING"


class TestHandleStopStack:
    @pytest.mark.asyncio
    async def test_missing_agent_name(self, server, mock_ws):
        await server._handle(mock_ws, {"cmd": "stop_stack"})
        msg = json.loads(mock_ws.send.call_args[0][0])
        assert msg["type"] == "error"

    @pytest.mark.asyncio
    async def test_stop_known_agent(self, server, mock_ws):
        info = StackInfo(
            agent_name="uav_01", compose_file="x.yml", status=StackStatus.STOPPING
        )
        with patch.object(server._manager, "stop", return_value=info):
            with patch("orchestrator.server.asyncio.create_task"):
                await server._handle(
                    mock_ws, {"cmd": "stop_stack", "agent_name": "uav_01"}
                )
        msg = json.loads(mock_ws.send.call_args[0][0])
        assert msg["status"] == "STOPPING"

    @pytest.mark.asyncio
    async def test_stop_unknown_agent(self, server, mock_ws):
        with patch.object(server._manager, "stop", return_value=None):
            await server._handle(
                mock_ws, {"cmd": "stop_stack", "agent_name": "nobody"}
            )
        msg = json.loads(mock_ws.send.call_args[0][0])
        assert msg["status"] == "UNKNOWN"


class TestHandleStopAllStacks:
    @pytest.mark.asyncio
    async def test_stops_running_stacks(self, server, mock_ws):
        statuses = {
            "uav_01": {"status": "RUNNING", "services": {}},
            "uav_02": {"status": "STOPPED", "services": {}},
            "uav_03": {"status": "DEGRADED", "services": {}},
        }
        with patch.object(server._manager, "get_all_status", return_value=statuses):
            with patch.object(server._manager, "stop") as mock_stop:
                with patch("orchestrator.server.asyncio.create_task"):
                    await server._handle(mock_ws, {"cmd": "stop_all_stacks"})

        # Should stop uav_01 and uav_03 but not uav_02 (already stopped)
        stopped_agents = [c[0][0] for c in mock_stop.call_args_list]
        assert "uav_01" in stopped_agents
        assert "uav_03" in stopped_agents
        assert "uav_02" not in stopped_agents

        msg = json.loads(mock_ws.send.call_args[0][0])
        assert msg["type"] == "info"
        assert "2" in msg["message"]


class TestHandleGetStackStatus:
    @pytest.mark.asyncio
    async def test_returns_all(self, server, mock_ws):
        statuses = {"uav_01": {"status": "RUNNING", "services": {"gw": "running"}}}
        with patch.object(server._manager, "get_all_status", return_value=statuses):
            await server._handle(mock_ws, {"cmd": "get_stack_status"})
        msg = json.loads(mock_ws.send.call_args[0][0])
        assert msg["type"] == "stack_status"
        assert msg["stacks"]["uav_01"]["status"] == "RUNNING"


class TestUnknownCommand:
    @pytest.mark.asyncio
    async def test_unknown_cmd(self, server, mock_ws):
        await server._handle(mock_ws, {"cmd": "fly_to_moon"})
        msg = json.loads(mock_ws.send.call_args[0][0])
        assert msg["type"] == "error"
        assert "fly_to_moon" in msg["message"]


# ── Broadcast ───────────────────────────────────────────────

class TestBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_clients(self, server):
        c1, c2 = AsyncMock(), AsyncMock()
        server._clients = {c1, c2}
        info = MagicMock()
        info.agent_name = "uav_01"
        info.status.name = "RUNNING"
        info.error_message = ""
        info.services = {"gw": "running"}

        await server._broadcast_update(info)
        assert c1.send.call_count == 1
        assert c2.send.call_count == 1

        msg = json.loads(c1.send.call_args[0][0])
        assert msg["agent_name"] == "uav_01"
        assert msg["services"]["gw"] == "running"

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_clients(self, server):
        good = AsyncMock()
        bad = AsyncMock()
        bad.send.side_effect = ConnectionError("gone")
        server._clients = {good, bad}
        info = MagicMock()
        info.agent_name = "uav_01"
        info.status.name = "RUNNING"
        info.error_message = ""
        info.services = {}

        await server._broadcast_update(info)
        assert bad not in server._clients
        assert good in server._clients
