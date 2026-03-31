"""Tests for orchestrator.compose_manager — Docker Compose lifecycle."""

import json
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.compose_manager import ComposeManager, StackInfo, StackStatus


@pytest.fixture
def manager():
    return ComposeManager()


class TestStackInfo:
    def test_auto_project_name(self):
        info = StackInfo(agent_name="uav_01", compose_file="/opt/stacks/test.yml")
        assert info.project_name == "sim_uav_01"

    def test_custom_project_name(self):
        info = StackInfo(agent_name="uav_01", compose_file="x.yml", project_name="custom")
        assert info.project_name == "custom"

    def test_default_status(self):
        info = StackInfo(agent_name="test", compose_file="x.yml")
        assert info.status == StackStatus.STOPPED


class TestComposeManagerLaunch:
    @patch("orchestrator.compose_manager.subprocess.run")
    @patch("orchestrator.compose_manager.threading.Thread")
    def test_launch_creates_stack_info(self, mock_thread, mock_run, manager):
        mock_thread.return_value = MagicMock()
        info = manager.launch("uav_01", "/opt/stacks/agent.yml", env={"AGENT_ID": "1"})
        assert info.agent_name == "uav_01"
        assert info.status == StackStatus.STARTING
        assert info.env == {"AGENT_ID": "1"}
        mock_thread.return_value.start.assert_called_once()

    @patch("orchestrator.compose_manager.subprocess.run")
    @patch("orchestrator.compose_manager.threading.Thread")
    def test_launch_already_running(self, mock_thread, mock_run, manager):
        mock_thread.return_value = MagicMock()
        manager.launch("uav_01", "/opt/stacks/agent.yml")
        # Second launch should return existing without starting new thread
        info = manager.launch("uav_01", "/opt/stacks/agent.yml")
        assert info.status == StackStatus.STARTING
        assert mock_thread.return_value.start.call_count == 1

    @patch("orchestrator.compose_manager.subprocess.run")
    def test_do_launch_success(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        info = StackInfo(
            agent_name="uav_01",
            compose_file="/opt/stacks/agent.yml",
            env={"AGENT_ID": "1"},
            status=StackStatus.STARTING,
        )
        manager._stacks["uav_01"] = info
        manager._do_launch(info)
        assert info.status == StackStatus.RUNNING

    @patch("orchestrator.compose_manager.subprocess.run")
    def test_do_launch_failure(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="compose error")
        info = StackInfo(
            agent_name="uav_01",
            compose_file="/opt/stacks/agent.yml",
            status=StackStatus.STARTING,
        )
        manager._stacks["uav_01"] = info
        manager._do_launch(info)
        assert info.status == StackStatus.ERROR
        assert "compose error" in info.error_message

    @patch("orchestrator.compose_manager.subprocess.run")
    def test_do_launch_env_passed(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        info = StackInfo(
            agent_name="uav_01",
            compose_file="/opt/stacks/agent.yml",
            env={"MY_VAR": "hello"},
            status=StackStatus.STARTING,
        )
        manager._stacks["uav_01"] = info
        manager._do_launch(info)
        # First call is docker compose up (with env); second is check_services
        up_call = mock_run.call_args_list[0]
        env = up_call.kwargs.get("env")
        assert env is not None
        assert env["MY_VAR"] == "hello"


class TestComposeManagerStop:
    @patch("orchestrator.compose_manager.threading.Thread")
    def test_stop_running_stack(self, mock_thread, manager):
        mock_thread.return_value = MagicMock()
        info = StackInfo(agent_name="uav_01", compose_file="x.yml", status=StackStatus.RUNNING)
        manager._stacks["uav_01"] = info
        result = manager.stop("uav_01")
        assert result.status == StackStatus.STOPPING
        mock_thread.return_value.start.assert_called_once()

    def test_stop_nonexistent(self, manager):
        result = manager.stop("nobody")
        assert result is None

    def test_stop_already_stopped(self, manager):
        info = StackInfo(agent_name="uav_01", compose_file="x.yml", status=StackStatus.STOPPED)
        manager._stacks["uav_01"] = info
        result = manager.stop("uav_01")
        assert result.status == StackStatus.STOPPED

    @patch("orchestrator.compose_manager.subprocess.run")
    def test_do_stop_success(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        info = StackInfo(agent_name="uav_01", compose_file="x.yml", status=StackStatus.STOPPING)
        manager._stacks["uav_01"] = info
        manager._do_stop(info)
        assert info.status == StackStatus.STOPPED
        assert info.services == {}


class TestComposeManagerStatus:
    def test_get_status(self, manager):
        info = StackInfo(agent_name="uav_01", compose_file="x.yml", status=StackStatus.RUNNING)
        manager._stacks["uav_01"] = info
        result = manager.get_status("uav_01")
        assert result is info

    def test_get_status_nonexistent(self, manager):
        assert manager.get_status("nobody") is None

    def test_get_all_status(self, manager):
        manager._stacks["uav_01"] = StackInfo(
            agent_name="uav_01", compose_file="x.yml", status=StackStatus.RUNNING
        )
        manager._stacks["uav_02"] = StackInfo(
            agent_name="uav_02", compose_file="y.yml", status=StackStatus.STOPPED
        )
        result = manager.get_all_status()
        assert result["uav_01"]["status"] == "RUNNING"
        assert result["uav_02"]["status"] == "STOPPED"


class TestComposeManagerHealthCheck:
    @patch("orchestrator.compose_manager.subprocess.run")
    def test_check_services_all_running(self, mock_run, manager):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"Service":"gateway","State":"running"}\n{"Service":"bridge","State":"running"}\n',
        )
        info = StackInfo(agent_name="uav_01", compose_file="x.yml", status=StackStatus.RUNNING)
        manager._stacks["uav_01"] = info
        services = manager.check_services("uav_01")
        assert services["gateway"] == "running"
        assert services["bridge"] == "running"
        assert info.status == StackStatus.RUNNING

    @patch("orchestrator.compose_manager.subprocess.run")
    def test_check_services_degraded(self, mock_run, manager):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"Service":"gateway","State":"running"}\n{"Service":"bridge","State":"exited"}\n',
        )
        info = StackInfo(agent_name="uav_01", compose_file="x.yml", status=StackStatus.RUNNING)
        manager._stacks["uav_01"] = info
        manager.check_services("uav_01")
        assert info.status == StackStatus.DEGRADED

    @patch("orchestrator.compose_manager.subprocess.run")
    def test_check_services_all_down(self, mock_run, manager):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"Service":"gateway","State":"exited"}\n{"Service":"bridge","State":"exited"}\n',
        )
        info = StackInfo(agent_name="uav_01", compose_file="x.yml", status=StackStatus.RUNNING)
        manager._stacks["uav_01"] = info
        manager.check_services("uav_01")
        assert info.status == StackStatus.ERROR

    @patch("orchestrator.compose_manager.subprocess.run")
    def test_check_services_recovers(self, mock_run, manager):
        """If all services come back, status recovers from DEGRADED."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"Service":"gateway","State":"running"}\n{"Service":"bridge","State":"running"}\n',
        )
        info = StackInfo(agent_name="uav_01", compose_file="x.yml", status=StackStatus.DEGRADED)
        manager._stacks["uav_01"] = info
        manager.check_services("uav_01")
        assert info.status == StackStatus.RUNNING
