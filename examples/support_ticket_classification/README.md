# Example 1: Support Ticket Classification (`support_ticket_classification`)

This example distills **Gemini 3.5 Flash** (`gemini-3.5-flash`) into **Gemma 3 4B** (`google/gemma-3-4b-it`) to classify enterprise technical support tickets into structured JSON outputs (`category`, `subcategory`, `severity`, `routing_queue`, `requires_escalation`, and `summary`).

## Dataset
- **File**: [`prompts.jsonl`](prompts.jsonl)
- **Size**: **150 realistic prompts** spanning 10 enterprise support categories (`BILLING_AND_INVOICING`, `AUTHENTICATION_AND_SSO`, `NETWORKING_AND_VPN`, `DATABASE_AND_STORAGE`, `API_AND_WEBHOOKS`, `SECURITY_AND_COMPLIANCE`, `KUBERNETES_AND_COMPUTE`, `DATA_PIPELINE_AND_ETL`, `UI_AND_DASHBOARD`), customer tiers (`Starter`, `Pro`, `Business`, `Enterprise`), and severity levels (`P0`–`P3`).

## Quickstart

```bash
# 1. Initialize the task workspace on GCS
distillfw init \
  --config examples/support_ticket_classification/config.yaml \
  --prompts examples/support_ticket_classification/prompts.jsonl

# 2. Run or resume the pipeline using only its GCS task URI
distillfw resume gs://my-distillfw-bucket/tasks/support-ticket-classification-v1
```
