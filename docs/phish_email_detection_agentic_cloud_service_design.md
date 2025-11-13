# Phish Email Detection Agentic Cloud Service — Design Document

## 1. Overview

The **Phish Email Detection Agentic Cloud Service** is an enterprise-grade, multi-modal phishing detection platform that integrates AI models, a **DeepAgents-based agentic layer**, and cloud-native orchestration to protect corporate communication systems. The system ingests emails from MS365/Gmail/IMAP gateways, processes text, URLs, HTML, headers, images, QR codes, and attachments, and outputs a high-confidence phishing risk decision with auditability and compliance built-in.

This document is written for **industry software development teams**, ensuring smooth implementation with detailed architecture, module definitions, API specs, and operational requirements.

Primary objectives:

* High-accuracy email phishing detection using multi-modal signals.
* Real-time processing with <150 ms end-to-end latency.
* Explainable and compliant decision traces for enterprise audit.
* Scalable cloud-native deployment supporting millions of emails/day.
* **Agentic workflow orchestration implemented with DeepAgents.**

---

## 2. High-Level Architecture

The system consists of seven cooperating layers:

1. **Email Ingestion Layer** — Connectors to MS Graph, Gmail API, IMAP/SMTP, with email normalization.
2. **Preprocessing Pipeline** — Extracts text, URLs, images, headers, metadata.
3. **Multi-Modal AI Model Layer** — Independent microservices for text, URL, image, attachment models.
4. **Agentic Orchestration Layer (DeepAgents)** — Executes the phishing-detection agent, tools, memory, and policy graphs.
5. **Decision Layer** — Fusion scoring, policy enforcement, action recommendation.
6. **Storage & Metadata Layer** — Object storage, feature store, inference logs, audit records.
7. **API & Integration Layer** — Scan API, admin API, internal microservice interfaces.

Data flows from ingestion → preprocessing → modality models → agent (DeepAgents) → fusion + policy → final decision → action.

---

## 3. Core Components

### 3.1 Email Ingestion Connectors

* Connectors for MS365 (Graph API), Gmail API, IMAP/SMTP.
* Supports OAuth2, token refresh, secure webhook events.
* Deduplication using message-ID + thread-ID.
* Converts raw email into unified JSON object (RFC822, MIME parsing).

### 3.2 Feature Extraction Services

* HTML → plaintext cleaner.
* Header parser: SPF/DKIM/DMARC.
* URL extractor + normalizer + redirect expansion.
* Image extractor + lightweight OCR.
* Attachment metadata (MIME, entropy, file signature).

### 3.3 AI Model Microservices

Each model is a standalone microservice, deployed with gRPC:

* `text-classifier-svc` (DistilBERT/FlexBERT)
* `url-risk-svc` (lexical features + TI + ML model)
* `image-detector-svc` (fake login, QR phishing)
* `attachment-risk-svc` (optional)
* `fusion-svc` (late fusion / MLP)

Models are packaged as TorchScript/ONNX, with clear versioning.

### 3.4 Agentic Reasoning Engine (DeepAgents Runtime)

Implemented using **DeepAgents** as the agent framework:

* One primary agent: **PhishDetectionAgent**.
* Tool abstractions wrap internal microservices (text/URL/image models, TI, policy engine).
* Memory components:

  * Short-term: per-email context (intermediate results).
  * Long-term: threat patterns, previous decisions (optional, configurable).
* Planning & control:

  * Agent decides which tools to call (e.g., skip image model if no images).
  * Agent can re-query tools when results are inconsistent.

### 3.5 Action Engine

* Quarantine email (via MS365/Gmail APIs).
* Flag user inbox / add banner.
* Send to human review queue.
* Generate explainability report for SIEM/SOAR integration.

---

## 4. Multi-Modal Detection Models

### 4.1 Text Model

* 20–40M parameter encoder-only model.
* Trained on phishing datasets + enterprise corpus.
* Outputs:

  * Phishing probability.
  * Key-sentence attention map.
  * Content risk labels (credential theft, invoice fraud, etc.).

### 4.2 Image Model

* Lightweight ViT/CNN model.
* Detects fake login pages, brand impersonation, QR-phishing.
* OCR-based secondary feature extraction.

### 4.3 URL Model

* Lexical analysis.
* WHOIS/SSL age.
* ThreatIntel hash/matching.
* Embedding-based similarity scoring.

### 4.4 Attachment Model (Optional)

* Entropy score.
* File signature mismatch detection.
* Macro/static heuristic detection.

### 4.5 Fusion Model

* Late fusion (weighted ensemble) or MLP over model logits.
* Includes uncertainty calibration.

---

## 5. Agentic Workflow Orchestration (DeepAgents)

### 5.1 Agent Topology

* **PhishDetectionAgent** (root agent)

  * Tools:

    * `call_text_model(email_features)`
    * `call_url_model(url_features)`
    * `call_image_model(image_features)`
    * `call_attachment_model(attachment_features)`
    * `call_threat_intel(ioc_list)`
    * `call_fusion(scores, ti_result)`
    * `call_policy_engine(fused_result, tenant_policy)`
  * Memory:

    * Per-email memory for intermediate steps.
    * Optional cross-email memory for pattern analysis (future extension).

### 5.2 Typical DeepAgents Execution Flow

1. Agent receives normalized email JSON from the preprocessing service.
2. Agent inspects structure and **decides which tools to call**:

   * If `has_images == false` → skip `call_image_model`.
   * If `url_count == 0` → skip `call_url_model`.
3. Agent invokes tools via DeepAgents tool interfaces (HTTP/gRPC wrappers).
4. Agent aggregates results and evaluates consistency.
5. Agent calls ThreatIntel tools for suspicious URLs/domains.
6. Agent calls `call_fusion` to combine scores into a unified risk score.
7. Agent calls `call_policy_engine` to transform score into actions.
8. Final result returned to Scan API + written to audit log.

### 5.3 Error Handling & Retries

* Per-tool timeout and retry configuration.
* Agent can fall back to rules-only mode if model services are unavailable.
* All errors are logged with correlation IDs for traceability.

### 5.4 Benefits of Using DeepAgents

* Clear separation between **LLM reasoning** and **specialized tools**.
* Easier to introduce new tools (e.g., new ThreatIntel provider).
* Declarative configuration of workflows and policies.

---

## 6. Data Pipeline & Storage

### 6.1 Storage Layers

* **Object Store**: Emails, images, attachments (encrypted buckets).
* **Feature Store**: Extracted features (Parquet + vector index).
* **ThreatIntel Cache**: Redis for fast risk lookup.
* **Inference Log Store**: OpenSearch/Elastic for analytics.
* **Audit Log Store**: Immutable append-only (e.g., WORM storage).

### 6.2 ETL Pipelines

* Daily dataset refresh (training/finetuning).
* Incremental ThreatIntel update pipeline.
* Data drift and accuracy monitoring pipelines.

### 6.3 Data Schema

* Unified Email JSON schema (headers, body, URLs, attachments).
* Feature schema for text, URL, image, metadata.

---

## 7. Model Serving & Scaling Strategy

### 7.1 Serving Technology

* Triton/TorchServe or custom gRPC inference runtime.
* Containerized GPU/CPU services managed by Kubernetes.

### 7.2 Autoscaling Policies

* HPA based on CPU/GPU usage.
* Queue-length based scaling.
* Model-aware batch size adjustment.

### 7.3 Resource Allocation

* Text model → CPU optimized.
* URL model → CPU only.
* Image model → GPU.
* Fusion → CPU.

### 7.4 Model Lifecycle

* Model register/publish/rollback via Admin API.
* Canary + A/B testing for new models.
* Automatic shadow evaluation before promotion.

---

## 8. API Design (Inbound / Internal / Admin)

### 8.1 Scan API (External)

```http
POST /v1/scan/email
Content-Type: application/json

{
  "email_base64": "...",
  "source": "ms365",        // ms365 | gmail | smtp_gateway
  "tenant_id": "...",
  "return_explanation": true
}
```

Response:

```json
{
  "is_phish": true,
  "score": 0.97,
  "actions": ["quarantine"],
  "explanation": [
    {
      "modality": "text",
      "reason": "Detected credential-stealing language and fake login brand."
    },
    {
      "modality": "url",
      "reason": "Domain registered 2 days ago, flagged in TI feed."
    }
  ],
  "correlation_id": "..."
}
```

### 8.2 Internal Microservice APIs

* **Preprocessing API**: accepts raw RFC822 email, returns normalized JSON.
* **Model APIs**: gRPC/HTTP endpoints for text/url/image/attachment models.
* **ThreatIntel API**: IOC lookup and enrichment.
* **Fusion API**: compute final risk score.
* **Policy API**: compute final action set.

### 8.3 Admin API

* Policy management (per-tenant sensitivity, action mapping).
* Model publish/rollback and traffic splitting.
* Threat-intel provider configuration and manual overrides.
* Audit log and decision trace retrieval.

---

## 9. Security & Compliance Requirements

### 9.1 Data Security

* AES-256 encryption at rest.
* TLS1.3 in transit.
* Strict least privilege IAM roles per service.
* No raw content in logs; fields masked or hashed where possible.

### 9.2 Compliance Targets

* SOC2 Type II.
* GDPR/CCPA compliant data lifecycle (retention, deletion, export).
* Full auditability of all decisions (who/what/when/why).

### 9.3 Runtime Hardening

* Pod Security Standards (PSS) enforced.
* Signed container images (Sigstore/Cosign) and admission controls.
* Secrets stored in Vault/KMS, never in code or env files.

---

## 10. Reliability, Monitoring & Observability

### 10.1 Metrics

* Model latency per modality.
* Queue latency & backlog.
* Error rate per service and per tool.
* ThreatIntel lookup latency & success rate.
* Agent-level metrics (average number of tool calls per email, fallback rate).

### 10.2 Logging & Tracing

* Structured JSON logs.
* Correlation-ID based tracing across microservices.
* Per-step agentic reasoning logs (tool calls, inputs/outputs, decisions) stored with redaction.

### 10.3 Resilience

* Circuit breakers around external ThreatIntel providers.
* Graceful degradation: rules-only mode if models unavailable.
* Retry with exponential backoff, idempotent operations for external APIs.

---

## 11. Deployment Topology (Cloud + On-Premise Option)

### 11.1 Cloud Deployment

* Kubernetes (AKS/EKS/GKE).
* Multi-AZ architecture.
* Managed storage (S3/Blob).
* GPU node pool for image models.
* Separate node pools for DeepAgents runtime vs model services.

### 11.2 On-Premise Deployment

* Air-gapped mode supported (offline TI synchronization window).
* Hardware: 1 GPU node + 2–4 CPU nodes for mid-size deployment.
* Uses same Helm charts with values overrides (no cloud-managed services).

---

## 12. Performance & Latency Targets

| Component        | Target Latency |
| ---------------- | -------------- |
| Preprocessing    | < 30 ms        |
| Text Model       | < 20 ms        |
| URL Model        | < 10 ms        |
| Image Model      | < 60 ms (GPU)  |
| Fusion + Policy  | < 10 ms        |
| DeepAgents Logic | < 20 ms        |
| **End-to-end**   | **< 150 ms**   |

Throughput target: **≥ 1M emails/day** per production cluster, horizontally scalable.

---

## 13. Future Extensions

* Deeper multimodal reasoning with larger LLMs (where latency budget allows).
* Graph-based sender and tenant reputation modeling.
* Federated learning to adapt to tenant-specific phishing patterns without raw data sharing.
* Autonomous TI ingestion and correlation with endpoint/network telemetry.
* User-behavior profiling (e.g., unusual reply/forward patterns) as an additional modality.

---

## 14. Appendix A — PhishDetectionAgent (DeepAgents) Config & Pseudo-code

> **Note:** The following is implementation-oriented pseudo-code and configuration.
> Actual DeepAgents API names may differ slightly and should be aligned with the version used by the team.

### 14.1 Conceptual Agent Topology

* **Agent:** `PhishDetectionAgent`
* **Inputs:** Normalized email JSON (after preprocessing service).
* **Outputs:** Final phishing decision, score, actions, explanation, correlation ID.
* **Tools (wrapped as DeepAgents tools):**

  * `call_text_model(email_features)`
  * `call_url_model(url_features)`
  * `call_image_model(image_features)`
  * `call_attachment_model(attachment_features)`
  * `call_threat_intel(ioc_list)`
  * `call_fusion(scores, ti_result)`
  * `call_policy_engine(fused_result, tenant_policy)`

### 14.2 High-Level DeepAgents Config (YAML-style)

```yaml
agents:
  phish_detection:
    description: >-
      Multi-modal phishing detection agent that calls internal model and
      threat-intel tools and returns a final decision, score, and actions.

    llm:
      provider: openai      # or other
      model: gpt-4.1        # example; choose according to latency/cost
      temperature: 0.1
      max_tokens: 512

    memory:
      short_term:
        type: in_memory
        max_items: 32
      long_term:
        type: disabled  # can be enabled in future for cross-email patterns

    tools:
      - name: text_model
        kind: http
        description: Call the text phishing classifier microservice.
        config:
          method: POST
          url: http://text-classifier-svc:8080/v1/predict
          timeout_ms: 100

      - name: url_model
        kind: http
        description: Call the URL risk scoring microservice.
        config:
          method: POST
          url: http://url-risk-svc:8080/v1/predict
          timeout_ms: 80

      - name: image_model
        kind: http
        description: Call the image phishing detection microservice.
        config:
          method: POST
          url: http://image-detector-svc:8080/v1/predict
          timeout_ms: 120

      - name: attachment_model
        kind: http
        optional: true
        description: Call the attachment risk microservice.
        config:
          method: POST
          url: http://attachment-risk-svc:8080/v1/predict
          timeout_ms: 100

      - name: threat_intel
        kind: http
        description: Query threat-intel enrichment service for URLs/domains/IPs.
        config:
          method: POST
          url: http://ti-svc:8080/v1/lookup
          timeout_ms: 150

      - name: fusion
        kind: http
        description: Combine modality scores and TI into a unified risk score.
        config:
          method: POST
          url: http://fusion-svc:8080/v1/fuse
          timeout_ms: 50

      - name: policy_engine
        kind: http
        description: Convert risk score and tenant policy into actions.
        config:
          method: POST
          url: http://policy-svc:8080/v1/evaluate
          timeout_ms: 50

    policies:
      routing:
        skip_image_if_no_attachments: true
        skip_url_if_no_urls: true
      timeouts:
        overall_ms: 200
      fallbacks:
        if_models_down: rules_only
```

### 14.3 Pseudo-code for PhishDetectionAgent

Below is a Python-style pseudo-code sketch showing how the agent might orchestrate tools. Adapt this to the actual DeepAgents SDK you use.

```python
from typing import Any, Dict, List

# Pseudo DeepAgents-like base classes
from deepagents import Agent, ToolContext, AgentContext


class PhishDetectionAgent(Agent):
    """DeepAgents-based agent for phishing detection."""

    name = "phish_detection"

    async def run(self, ctx: AgentContext, email: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Extract basic structure
        headers = email.get("headers", {})
        body = email.get("body", "")
        urls = email.get("urls", [])
        images = email.get("images", [])  # maybe base64 or references
        attachments = email.get("attachments", [])
        tenant_id = email.get("tenant_id")

        # 2) Decide which modalities to call
        use_text = bool(body)
        use_url = len(urls) > 0
        use_image = len(images) > 0
        use_attachment = len(attachments) > 0

        scores = {}

        # 3) Call tools based on available modalities
        if use_text:
            scores["text"] = await self.call_text_model(ctx.tools, headers, body)

        if use_url:
            scores["url"] = await self.call_url_model(ctx.tools, urls)

        if use_image:
            scores["image"] = await self.call_image_model(ctx.tools, images)

        if use_attachment:
            scores["attachment"] = await self.call_attachment_model(ctx.tools, attachments)

        # 4) Collect IOCs for ThreatIntel
        iocs = self.extract_iocs(urls, headers)
        ti_result = await self.call_threat_intel(ctx.tools, iocs) if iocs else None

        # 5) Call fusion service
        fused = await self.call_fusion(ctx.tools, scores, ti_result)

        # 6) Fetch tenant policy (could also be provided in email input)
        tenant_policy = await self.fetch_tenant_policy(tenant_id)

        # 7) Call policy engine to get final actions
        decision = await self.call_policy_engine(ctx.tools, fused, tenant_policy)

        # 8) Attach correlation ID and explanation
        result = {
            "is_phish": decision["is_phish"],
            "score": decision["score"],
            "actions": decision["actions"],
            "explanation": decision.get("explanation", []),
            "correlation_id": ctx.correlation_id,
        }

        return result

    # --- Tool wrappers -------------------------------------------------

    async def call_text_model(self, tools: ToolContext, headers, body: str) -> Dict[str, Any]:
        payload = {"headers": headers, "body": body}
        return await tools["text_model"].call(json=payload)

    async def call_url_model(self, tools: ToolContext, urls: List[str]) -> Dict[str, Any]:
        payload = {"urls": urls}
        return await tools["url_model"].call(json=payload)

    async def call_image_model(self, tools: ToolContext, images: List[str]) -> Dict[str, Any]:
        payload = {"images": images}
        return await tools["image_model"].call(json=payload)

    async def call_attachment_model(self, tools: ToolContext, attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {"attachments": attachments}
        return await tools["attachment_model"].call(json=payload)

    async def call_threat_intel(self, tools: ToolContext, iocs: Dict[str, Any]) -> Dict[str, Any]:
        return await tools["threat_intel"].call(json=iocs)

    async def call_fusion(self, tools: ToolContext, scores: Dict[str, Any], ti_result: Dict[str, Any] | None) -> Dict[str, Any]:
        payload = {"scores": scores, "threat_intel": ti_result}
        return await tools["fusion"].call(json=payload)

    async def call_policy_engine(self, tools: ToolContext, fused: Dict[str, Any], tenant_policy: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"fused": fused, "policy": tenant_policy}
        return await tools["policy_engine"].call(json=payload)

    # --- Helpers -------------------------------------------------------

    def extract_iocs(self, urls: List[str], headers: Dict[str, Any]) -> Dict[str, Any]:
        # Minimal example; extend to IPs, domains, etc.
        return {"urls": urls}

    async def fetch_tenant_policy(self, tenant_id: str | None) -> Dict[str, Any]:
        # In production, call policy service or config store; here, return default.
        if tenant_id is None:
            return {"sensitivity": "medium"}
        # TODO: implement real lookup
        return {"tenant_id": tenant_id, "sensitivity": "medium"}
```

---

## 15. Appendix B — Developer Guide

### 15.1 Implementing Tool Wrappers

Each DeepAgents tool is a thin wrapper around an internal microservice. Recommended pattern:

1. **Define the service contract** (request/response JSON schema) for each model and utility microservice.
2. **Implement an HTTP client wrapper** (with timeout, retries, metrics) that is reused by both:

   * DeepAgents tools.
   * Non-agent code paths (e.g., batch offline jobs).
3. **Register tools in DeepAgents** using these wrappers.

Example (pseudo-code for a tool implementation):

```python
from deepagents import Tool
import httpx


class HttpJsonTool(Tool):
    def __init__(self, name: str, url: str, timeout_ms: int = 100):
        super().__init__(name=name)
        self._url = url
        self._timeout = timeout_ms / 1000.0

    async def __call__(self, json: dict) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._url, json=json)
            resp.raise_for_status()
            return resp.json()


# Registration example
text_tool = HttpJsonTool("text_model", "http://text-classifier-svc:8080/v1/predict", timeout_ms=100)
```

Guidelines:

* All tools must **validate inputs/outputs** and log errors with correlation IDs.
* Treat tools as pure functions for a given request (idempotent as much as possible).
* Limit tool execution time and use retries with backoff for transient errors.

### 15.2 Local Testing of Agents

Recommended workflow for developers:

1. **Run services locally**

   * Start mock or real microservices via Docker Compose (`text`, `url`, `image`, `ti`, `fusion`, `policy`).
   * Expose consistent ports (e.g., `localhost:8081` for text, `8082` for url, etc.).

2. **Configure DeepAgents for local endpoints**

   * Provide a `deepagents.local.yaml` with URLs pointing to `localhost` instead of Kubernetes services.

3. **Write unit tests for PhishDetectionAgent**

   * Use a small set of representative email JSON fixtures: benign, obvious phish, borderline phish.
   * Assert:

     * `is_phish` and `score` values are in expected range.
     * Correct tools are called (e.g., no image tool call if no images).
     * Explanations contain expected modalities and reasons.

Example (pytest-style pseudo-code):

```python
import pytest
from deepagents import AgentRuntime
from my_agents import PhishDetectionAgent


@pytest.mark.asyncio
async def test_obvious_phish_flow():
    runtime = AgentRuntime(config_path="deepagents.local.yaml")
    agent = PhishDetectionAgent()

    email = {
        "headers": {"from": "it-support@fake-login.com"},
        "body": "Please login to update your password at http://fake-login.com",
        "urls": ["http://fake-login.com"],
        "images": [],
        "attachments": [],
        "tenant_id": "tenant-1",
    }

    result = await agent.run(runtime.new_context(), email=email)

    assert result["is_phish"] is True
    assert result["score"] > 0.9
    assert any(r["modality"] == "text" for r in result["explanation"])
    assert any(r["modality"] == "url" for r in result["explanation"])
```

### 15.3 Mocking Model Services

For fast and deterministic tests, mock the microservices instead of calling real models:

1. **Mock at the HTTP layer**

   * Use a library like `responses`/`pytest-httpx` to intercept HTTP calls from tools and return canned JSON.

2. **Mock at the tool layer**

   * Replace tools in the agent’s `ToolContext` with fake implementations that return fixed scores.

Example: mocking at tool layer (pseudo-code):

```python
class FakeTextTool:
    async def call(self, json: dict) -> dict:
        # simple heuristic: if body contains "password" and "login", mark high risk
        body = json.get("body", "").lower()
        score = 0.95 if ("password" in body and "login" in body) else 0.1
        return {"score": score, "modality": "text"}


@pytest.mark.asyncio
async def test_agent_with_fake_tools():
    agent = PhishDetectionAgent()

    class FakeTools(dict):
        pass

    tools = FakeTools(text_model=FakeTextTool(), url_model=FakeTextTool())  # etc.

    ctx = type("FakeCtx", (), {"tools": tools, "correlation_id": "test-123"})()

    email = {"headers": {}, "body": "Please login to reset your password", "urls": []}

    result = await agent.run(ctx, email=email)
    assert result["is_phish"] is True
```

Guidelines for mocking:

* **Keep mocks simple**, deterministic, and clearly documented.
* Use realistic ranges for scores (e.g., [0.0, 1.0]).
* Cover corner cases: no URLs, multiple attachments, very long body, etc.

### 15.4 CI Integration

* Add a **test suite** that runs agent tests with mocked services on every PR.
* Optionally add an **integration test job** that spins up Docker Compose with real model containers and runs a smaller set of high-level end-to-end tests.
* Fail the pipeline if:

  * Latency budget is exceeded in integration tests.
  * Contract changes break tool or API compatibility.

This Developer Guide should give the implementation team enough detail to:

* Implement DeepAgents tool wrappers.
* Build and test the `PhishDetectionAgent` end-to-end.
* Safely mock services for unit/integration tests and CI.

---

## 14. Appendix A — PhishDetectionAgent (DeepAgents) Config & Pseudo-code

> **Note:** The following is implementation-oriented pseudo-code and configuration.
> Actual DeepAgents API names may differ slightly and should be aligned with the version used by the team.

### 14.1 Conceptual Agent Topology

* **Agent:** `PhishDetectionAgent`
* **Inputs:** Normalized email JSON (after preprocessing service).
* **Outputs:** Final phishing decision, score, actions, explanation, correlation ID.
* **Tools (wrapped as DeepAgents tools):**

  * `call_text_model(email_features)`
  * `call_url_model(url_features)`
  * `call_image_model(image_features)`
  * `call_attachment_model(attachment_features)`
  * `call_threat_intel(ioc_list)`
  * `call_fusion(scores, ti_result)`
  * `call_policy_engine(fused_result, tenant_policy)`

### 14.2 High-Level DeepAgents Config (YAML-style)

```yaml
agents:
  phish_detection:
    description: >-
      Multi-modal phishing detection agent that calls internal model and
      threat-intel tools and returns a final decision, score, and actions.

    llm:
      provider: openai      # or other
      model: gpt-4.1        # example; choose according to latency/cost
      temperature: 0.1
      max_tokens: 512

    memory:
      short_term:
        type: in_memory
        max_items: 32
      long_term:
        type: disabled  # can be enabled in future for cross-email patterns

    tools:
      - name: text_model
        kind: http
        description: Call the text phishing classifier microservice.
        config:
          method: POST
          url: http://text-classifier-svc:8080/v1/predict
          timeout_ms: 100

      - name: url_model
        kind: http
        description: Call the URL risk scoring microservice.
        config:
          method: POST
          url: http://url-risk-svc:8080/v1/predict
          timeout_ms: 80

      - name: image_model
        kind: http
        description: Call the image phishing detection microservice.
        config:
          method: POST
          url: http://image-detector-svc:8080/v1/predict
          timeout_ms: 120

      - name: attachment_model
        kind: http
        optional: true
        description: Call the attachment risk microservice.
        config:
          method: POST
          url: http://attachment-risk-svc:8080/v1/predict
          timeout_ms: 100

      - name: threat_intel
        kind: http
        description: Query threat-intel enrichment service for URLs/domains/IPs.
        config:
          method: POST
          url: http://ti-svc:8080/v1/lookup
          timeout_ms: 150

      - name: fusion
        kind: http
        description: Combine modality scores and TI into a unified risk score.
        config:
          method: POST
          url: http://fusion-svc:8080/v1/fuse
          timeout_ms: 50

      - name: policy_engine
        kind: http
        description: Convert risk score and tenant policy into actions.
        config:
          method: POST
          url: http://policy-svc:8080/v1/evaluate
          timeout_ms: 50

    policies:
      routing:
        skip_image_if_no_attachments: true
        skip_url_if_no_urls: true
      timeouts:
        overall_ms: 200
      fallbacks:
        if_models_down: rules_only
```

### 14.3 Pseudo-code for PhishDetectionAgent

Below is a Python-style pseudo-code sketch showing how the agent might orchestrate tools. Adapt this to the actual DeepAgents SDK you use.

```python
from typing import Any, Dict, List

# Pseudo DeepAgents-like base classes
from deepagents import Agent, ToolContext, AgentContext


class PhishDetectionAgent(Agent):
    """DeepAgents-based agent for phishing detection."""

    name = "phish_detection"

    async def run(self, ctx: AgentContext, email: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Extract basic structure
        headers = email.get("headers", {})
        body = email.get("body", "")
        urls = email.get("urls", [])
        images = email.get("images", [])  # maybe base64 or references
        attachments = email.get("attachments", [])
        tenant_id = email.get("tenant_id")

        # 2) Decide which modalities to call
        use_text = bool(body)
        use_url = len(urls) > 0
        use_image = len(images) > 0
        use_attachment = len(attachments) > 0

        scores = {}

        # 3) Call tools based on available modalities
        if use_text:
            scores["text"] = await self.call_text_model(ctx.tools, headers, body)

        if use_url:
            scores["url"] = await self.call_url_model(ctx.tools, urls)

        if use_image:
            scores["image"] = await self.call_image_model(ctx.tools, images)

        if use_attachment:
            scores["attachment"] = await self.call_attachment_model(ctx.tools, attachments)

        # 4) Collect IOCs for ThreatIntel
        iocs = self.extract_iocs(urls, headers)
        ti_result = await self.call_threat_intel(ctx.tools, iocs) if iocs else None

        # 5) Call fusion service
        fused = await self.call_fusion(ctx.tools, scores, ti_result)

        # 6) Fetch tenant policy (could also be provided in email input)
        tenant_policy = await self.fetch_tenant_policy(tenant_id)

        # 7) Call policy engine to get final actions
        decision = await self.call_policy_engine(ctx.tools, fused, tenant_policy)

        # 8) Attach correlation ID and explanation
        result = {
            "is_phish": decision["is_phish"],
            "score": decision["score"],
            "actions": decision["actions"],
            "explanation": decision.get("explanation", []),
            "correlation_id": ctx.correlation_id,
        }

        return result

    # --- Tool wrappers -------------------------------------------------

    async def call_text_model(self, tools: ToolContext, headers, body: str) -> Dict[str, Any]:
        payload = {"headers": headers, "body": body}
        return await tools["text_model"].call(json=payload)

    async def call_url_model(self, tools: ToolContext, urls: List[str]) -> Dict[str, Any]:
        payload = {"urls": urls}
        return await tools["url_model"].call(json=payload)

    async def call_image_model(self, tools: ToolContext, images: List[str]) -> Dict[str, Any]:
        payload = {"images": images}
        return await tools["image_model"].call(json=payload)

    async def call_attachment_model(self, tools: ToolContext, attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {"attachments": attachments}
        return await tools["attachment_model"].call(json=payload)

    async def call_threat_intel(self, tools: ToolContext, iocs: Dict[str, Any]) -> Dict[str, Any]:
        return await tools["threat_intel"].call(json=iocs)

    async def call_fusion(self, tools: ToolContext, scores: Dict[str, Any], ti_result: Dict[str, Any] | None) -> Dict[str, Any]:
        payload = {"scores": scores, "threat_intel": ti_result}
        return await tools["fusion"].call(json=payload)

    async def call_policy_engine(self, tools: ToolContext, fused: Dict[str, Any], tenant_policy: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"fused": fused, "policy": tenant_policy}
        return await tools["policy_engine"].call(json=payload)

    # --- Helpers -------------------------------------------------------

    def extract_iocs(self, urls: List[str], headers: Dict[str, Any]) -> Dict[str, Any]:
        # Minimal example; extend to IPs, domains, etc.
        return {"urls": urls}

    async def fetch_tenant_policy(self, tenant_id: str | None) -> Dict[str, Any]:
        # In production, call policy service or config store; here, return default.
        if tenant_id is None:
            return {"sensitivity": "medium"}
        # TODO: implement real lookup
        return {"tenant_id": tenant_id, "sensitivity": "medium"}
```

---

## 15. Appendix B — Developer Guide

### 15.1 Implementing Tool Wrappers

Each DeepAgents tool is a thin wrapper around an internal microservice. Recommended pattern:

1. **Define the service contract** (request/response JSON schema) for each model and utility microservice.
2. **Implement an HTTP client wrapper** (with timeout, retries, metrics) that is reused by both:

   * DeepAgents tools.
   * Non-agent code paths (e.g., batch offline jobs).
3. **Register tools in DeepAgents** using these wrappers.

Example (pseudo-code for a tool implementation):

```python
from deepagents import Tool
import httpx


class HttpJsonTool(Tool):
    def __init__(self, name: str, url: str, timeout_ms: int = 100):
        super().__init__(name=name)
        self._url = url
        self._timeout = timeout_ms / 1000.0

    async def __call__(self, json: dict) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._url, json=json)
            resp.raise_for_status()
            return resp.json()


# Registration example
text_tool = HttpJsonTool("text_model", "http://text-classifier-svc:8080/v1/predict", timeout_ms=100)
```

Guidelines:

* All tools must **validate inputs/outputs** and log errors with correlation IDs.
* Treat tools as pure functions for a given request (idempotent as much as possible).
* Limit tool execution time and use retries with backoff for transient errors.

### 15.2 Local Testing of Agents

Recommended workflow for developers:

1. **Run services locally**

   * Start mock or real microservices via Docker Compose (`text`, `url`, `image`, `ti`, `fusion`, `policy`).
   * Expose consistent ports (e.g., `localhost:8081` for text, `8082` for url, etc.).

2. **Configure DeepAgents for local endpoints**

   * Provide a `deepagents.local.yaml` with URLs pointing to `localhost` instead of Kubernetes services.

3. **Write unit tests for PhishDetectionAgent**

   * Use a small set of representative email JSON fixtures: benign, obvious phish, borderline phish.
   * Assert:

     * `is_phish` and `score` values are in expected range.
     * Correct tools are called (e.g., no image tool call if no images).
     * Explanations contain expected modalities and reasons.

Example (pytest-style pseudo-code):

```python
import pytest
from deepagents import AgentRuntime
from my_agents import PhishDetectionAgent


@pytest.mark.asyncio
async def test_obvious_phish_flow():
    runtime = AgentRuntime(config_path="deepagents.local.yaml")
    agent = PhishDetectionAgent()

    email = {
        "headers": {"from": "it-support@fake-login.com"},
        "body": "Please login to update your password at http://fake-login.com",
        "urls": ["http://fake-login.com"],
        "images": [],
        "attachments": [],
        "tenant_id": "tenant-1",
    }

    result = await agent.run(runtime.new_context(), email=email)

    assert result["is_phish"] is True
    assert result["score"] > 0.9
    assert any(r["modality"] == "text" for r in result["explanation"])
    assert any(r["modality"] == "url" for r in result["explanation"])
```

### 15.3 Mocking Model Services

For fast and deterministic tests, mock the microservices instead of calling real models:

1. **Mock at the HTTP layer**

   * Use a library like `responses`/`pytest-httpx` to intercept HTTP calls from tools and return canned JSON.

2. **Mock at the tool layer**

   * Replace tools in the agent’s `ToolContext` with fake implementations that return fixed scores.

Example: mocking at tool layer (pseudo-code):

```python
class FakeTextTool:
    async def call(self, json: dict) -> dict:
        # simple heuristic: if body contains "password" and "login", mark high risk
        body = json.get("body", "").lower()
        score = 0.95 if ("password" in body and "login" in body) else 0.1
        return {"score": score, "modality": "text"}


@pytest.mark.asyncio
async def test_agent_with_fake_tools():
    agent = PhishDetectionAgent()

    class FakeTools(dict):
        pass

    tools = FakeTools(text_model=FakeTextTool(), url_model=FakeTextTool())  # etc.

    ctx = type("FakeCtx", (), {"tools": tools, "correlation_id": "test-123"})()

    email = {"headers": {}, "body": "Please login to reset your password", "urls": []}

    result = await agent.run(ctx, email=email)
    assert result["is_phish"] is True
```

Guidelines for mocking:

* **Keep mocks simple**, deterministic, and clearly documented.
* Use realistic ranges for scores (e.g., [0.0, 1.0]).
* Cover corner cases: no URLs, multiple attachments, very long body, etc.

### 15.4 CI Integration

* Add a **test suite** that runs agent tests with mocked services on every PR.
* Optionally add an **integration test job** that spins up Docker Compose with real model containers and runs a smaller set of high-level end-to-end tests.
* Fail the pipeline if:

  * Latency budget is exceeded in integration tests.
  * Contract changes break tool or API compatibility.

