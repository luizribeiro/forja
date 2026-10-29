# SmolLM2 chat

SmolLM2-135M-Instruct uses ChatML with a default SmolLM system message and EOS
id 2. The 135M model declares no tool protocol; grammar requests are refused.
Select `[chat] component = "smollm2-chat"` in the engine profile.

The template is copied verbatim from `tokenizer_config.json` at Hugging Face
revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The render test fixes the
user-only prompt bytes and tests plain incremental text. System messages suppress
the default system insertion. The model and template are Apache-2.0 licensed.
