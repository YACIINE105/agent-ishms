from ..LLMinterface import LLM_Interface
from ..LLMEnums import GoogleAI_Enums, DocumentTypeEnum
import logging
from google import genai
from google.genai import types
from typing import List, Union

class GoogleAIProvider(LLM_Interface):
    def __init__(self , api_key:str, 
                            default_generation_max_output_characters:int=1000,
                            default_generation_max_output_token:int=1000, 
                            default_generation_temperature:float=0.2):     
        self.api_key = api_key
        self.default_generation_max_output_characters = default_generation_max_output_characters
        self.default_generation_max_output_token = default_generation_max_output_token
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None
        self.client = genai.Client(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id:str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id:str, embedding_size:int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text:str):
        return text[:self.default_generation_max_output_characters].strip()

    def generate_text(self, prompt:str, chat_history:list=None, 
                      max_output_token:int=None, temperature:float=None):

        if not self.client:
            self.logger.error("GoogleAI client was not set")
            return None

        if not self.generation_model_id:
                self.logger.error("Generation model for GoogleAI client wasn't set ")
                return None

        max_output_token = max_output_token if max_output_token else self.default_generation_max_output_token     
        temperature = temperature if temperature else self.default_generation_temperature
        user_message = self.construct_prompt(prompt=prompt, 
                                             role=GoogleAI_Enums.USER.value)
        chat_history = chat_history or []

        chat_history.append(user_message)

        try:
            response = self.client.models.generate_content(
                        model = self.generation_model_id,
                        contents = chat_history,
                        config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_token
                )
                    )

            if not response.text or not response.text.strip():
                self.logger.error("GoogleAI returned a successful response, but the text is empty.")
                return None

            chat_history.append(self.construct_response(response=response.text))

            return response.text

        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return None

    def embed_text(self, text: Union[str, List[str]], document_type: str = None, **kwargs):
        if not self.client:
            self.logger.error("GoogleAI client wasn't set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for GoogleAI client wasn't set")
            return None

        is_single = isinstance(text, str)
        contents = [text] if is_single else text

        task = "RETRIEVAL_DOCUMENT" if document_type == DocumentTypeEnum.DOCUMENT.value else "RETRIEVAL_QUERY"
        try:
            response = self.client.models.embed_content(
                model=self.embedding_model_id,
                contents=contents,
                config=types.EmbedContentConfig(
                    task_type=task,
                    output_dimensionality=self.embedding_size
                )
            )

            if not response or not response.embeddings:
                self.logger.error("GoogleAI returned an empty embedding response.")
                return None

            embeddings = [emb.values for emb in response.embeddings if hasattr(emb, "values")]

            if self.embedding_size and embeddings and len(embeddings[0]) != self.embedding_size:
                self.logger.warning(
                    f"Expected dimension {self.embedding_size}, but got {len(embeddings[0])}"
                )

            return embeddings

        except Exception as e:
            self.logger.error(f"Error during embedding generation: {e}")
            return None

    def construct_response(self, response):
        return {
            "role" : GoogleAI_Enums.ASSISTANT.value,
            "parts" : response      
        }

    def construct_prompt(self, prompt:str, role:str):
        return {"role":role, 
                "parts":[self.process_text(prompt)]
                }
