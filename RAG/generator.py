"""
generator.py
Generates answers from retrieved context using a language model.

Three backends are supported, auto-selected based on what's available:
  1. Anthropic API  -> used if ANTHROPIC_API_KEY is set
  2. OpenAI API      -> used if OPENAI_API_KEY is set
  3. Local HF model  -> flan-t5-base, fully offline, no key needed (default fallback)

This means the project runs out-of-the-box with zero API keys,
but automatically upgrades to a stronger model if you provide one.
"""
import os


PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the context below.
If the answer is not contained in the context, say "I don't have enough information in the document to answer that."

Context:
{context}

Question: {question}

Answer:"""


class LocalGenerator:
    """Offline generator using a small HF seq2seq model (no API key needed)."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        from transformers import pipeline
        self.pipe = pipeline("text2text-generation", model=model_name)

    def generate(self, context: str, question: str) -> str:
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        result = self.pipe(prompt, max_new_tokens=200, do_sample=False)
        return result[0]["generated_text"].strip()


class AnthropicGenerator:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, context: str, question: str) -> str:
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()


class OpenAIGenerator:
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def generate(self, context: str, question: str) -> str:
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()


def get_generator():
    """Pick the best available generator backend automatically."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("[Generator] Using Anthropic API")
        return AnthropicGenerator()
    if os.environ.get("OPENAI_API_KEY"):
        print("[Generator] Using OpenAI API")
        return OpenAIGenerator()
    print("[Generator] No API key found - using local flan-t5-base model")
    return LocalGenerator()
