# crucible_orch

Stack orchestrator for the CRUCIBLE SITL framework. Manages Docker Compose lifecycle for per-agent software stacks via a WebSocket API.

## Features
- Launch/stop agent stacks from the browser
- Injects agent-specific environment variables (AGENT_NAME, AGENT_ID, ROS_DOMAIN_ID, etc.)
- Monitors stack health and reports status in real time
- Graceful shutdown of all stacks on exit

## Requirements
- Docker socket access (`/var/run/docker.sock`)
- Docker Compose CLI (`docker compose`)

## Testing

```bash
pip install pytest pytest-asyncio websockets
python3 -m pytest tests/ -v
```

No ROS2 installation required. Tests use mocks for Docker/subprocess interactions.

Part of the [CRUCIBLE](https://github.com/TODO/crucible) framework.
