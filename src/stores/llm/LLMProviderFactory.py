from .LLMEnums import LLM_Enums
from .providers import CohereProvider, OpenAIProvider, GoogleAIProvider, LlamaCPPProvider

class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str, *, for_embeddings: bool = False):
        if provider == LLM_Enums.OPENAI.value:
            api_key = self.config.OPENAI_API_KEY
            api_url = self.config.OPENAI_BASE_URL
            if for_embeddings:
                api_key = getattr(self.config, "EMBEDDING_API_KEY", None) or api_key
                api_url = getattr(self.config, "EMBEDDING_BASE_URL", None) or api_url
            return OpenAIProvider(api_key=api_key, api_url=api_url)

        if provider == LLM_Enums.COHERE.value:
            return CohereProvider(
                api_key=self.config.COHERE_API_KEY
            )

        if provider == LLM_Enums.GOOGLE_AI.value:
            return GoogleAIProvider(
                api_key=self.config.GOOGLE_AI_API_KEY
            )

        if provider == LLM_Enums.LLAMA_CPP.value:
            return LlamaCPPProvider(
                n_gpu_layers=self.config.LLAMA_CPP_N_GPU_LAYERS,
                n_ctx=self.config.LLAMA_CPP_N_CTX
            )

        return None
