# Phish Agent

Phish Agent 是一个用于钓鱼邮件检测的多模态智能体项目骨架，旨在整合文本、URL、图像、附件和威胁情报等多源信号，提供统一的扫描 API 与工具集。该仓库目前提供项目结构和占位实现，方便后续扩展。

## 项目结构

```
phish_agent/
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
├── src/
│   └── phish_agent/
│       ├── __init__.py
│       ├── agents/
│       │   └── phish_detection_agent.py
│       ├── tools/
│       │   ├── text_model_tool.py
│       │   ├── url_model_tool.py
│       │   ├── image_model_tool.py
│       │   ├── attachment_model_tool.py
│       │   ├── threat_intel_tool.py
│       │   ├── fusion_tool.py
│       │   └── policy_tool.py
│       ├── runtime/
│       │   ├── loader.py
│       │   └── api_entry.py
│       └── schemas/
│           └── email_schema.py
├── configs/
│   ├── deepagents.local.yaml
│   ├── deepagents.dev.yaml
│   ├── deepagents.prod.yaml
│   └── logging.yaml
├── tests/
│   ├── test_agent_local.py
│   ├── test_tools.py
│   ├── fixtures/
│   └── mock_services/
└── docker/
    ├── Dockerfile
    ├── docker-compose.local.yaml
    └── docker-compose.mock.yaml
```

## 开发指南

1. 使用 [uv](https://github.com/astral-sh/uv) 管理依赖：
   ```bash
   uv venv
   uv sync
   ```
2. 启动本地 API：
   ```bash
   uv run fastapi dev src/phish_agent/runtime/api_entry.py
   ```
3. 运行测试：
   ```bash
   uv run pytest
   ```

后续可以在各个模块内填充具体实现，例如接入深度模型、定义扫描策略等。
