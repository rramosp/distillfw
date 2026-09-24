# Example 2: Retail Product Search Query Expansion (`query_expansion`)

This example distills **Gemini 3.5 Flash** (`gemini-3.5-flash`) into **Gemma 3 4B** (`google/gemma-3-4b-it`) to perform low-latency retail product search query expansion (rewriting dense product queries, generating retail keyword synonyms, classifying query specificity, and extracting structured product attributes like brand, color, material, size, and style).

## Dataset & Prompt Construction
- **File**: [`prompts.jsonl`](prompts.jsonl)
- **Format**: Each JSONL line contains a `"data"` JSON object (`retail_category`, `query_detail_level`, `shopper_query`) and `"metadata"`, which are populated into `prompt_construction.prompt_template` (and paired with `prompt_construction.system_instructions`) in [`config.yaml`](config.yaml) for both the Teacher and Student models.
- **Size**: **150 retail shopper queries** spanning 10 retail categories (`footwear_athletic`, `apparel_fashion`, `consumer_electronics`, `home_kitchen_appliances`, `furniture_decor`, `beauty_personal_care`, `sports_outdoor_gear`, `luggage_bags`, `watches_jewelry`, `toys_baby_pet_supplies`) across three explicit levels of product detail (`specificity_level`):
  - **Broad (`broad`)**: Short, underspecified category searches (e.g., `"running shoes"`, `"denim jacket"`, `"espresso machine"`).
  - **Medium (`medium`)**: Multi-attribute functional/style searches (e.g., `"waterproof trail running shoes with wide toe box"`, `"oversized vintage wash blue denim trucker jacket"`).
  - **High (`high`)**: Highly specific product descriptions with brand, color, material, or model details (e.g., `"adidas pair of sporting shoes with red stripes"`, `"breville barista express brushed stainless steel espresso machine with conical burr grinder"`).

## Quickstart

```bash
# 1. Initialize the task workspace on GCS
distillfw init \
  --config examples/query_expansion/config.yaml \
  --prompts examples/query_expansion/prompts.jsonl

# 2. Run or resume the pipeline using only its GCS task URI
distillfw resume gs://my-distillfw-bucket/tasks/query-expansion-v1
```
