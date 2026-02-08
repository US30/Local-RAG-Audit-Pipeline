from deepeval.models import DeepEvalBaseLLM
from langchain_ollama import OllamaLLM
import json

class OllamaEvaluator(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name
        # We set format="json" to force Ollama to be strict
        self.llm = OllamaLLM(model=model_name, format="json")

    def load_model(self):
        return self.llm

    def _clean_json(self, text: str) -> str:
        """Helper to strip markdown code blocks if the LLM adds them."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def generate(self, prompt: str) -> str:
        # We explicitly tell the model to use JSON in the prompt as a backup
        res = self.llm.invoke(prompt)
        return self._clean_json(res)

    async def a_generate(self, prompt: str) -> str:
        res = await self.llm.ainvoke(prompt)
        return self._clean_json(res)

    def get_model_name(self):
        return self.model_name