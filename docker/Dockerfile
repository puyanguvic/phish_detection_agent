FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv sync --frozen

COPY src ./src
COPY configs ./configs

ENV PATH="/opt/venv/bin:$PATH"

CMD ["uv", "run", "uvicorn", "phish_agent.runtime.api_entry:app", "--host", "0.0.0.0", "--port", "8000"]
