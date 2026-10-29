from ..LLMinterface import LLM_Interface
from openai import OpenAI
import logging
from ..LLMEnums import OpenAI_Enums
from typing import List, Union

class OpenAIProvider(LLM_Interface):
    def __init__(self , api_key:str, api_url:str=None, 
                        default_generation_max_output_characters:int=1000,
                        default_generation_max_output_token:int=1000, 
                        default_generation_temperature:float=0.2):

        self.api_key = api_key
        self.api_url = api_url
        self.default_generation_max_output_characters = default_generation_max_output_characters
        self.default_generation_max_output_token = default_generation_max_output_token
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None
        self.enums = OpenAI_Enums

        client_kwargs = {"api_key": self.api_key, "max_retries": 0}
        if self.api_url:
            client_kwargs["base_url"] = self.api_url if self.api_url and len(self.api_url)!=0 else None

        self.client = OpenAI(**client_kwargs)

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id:str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id:str, embedding_size:int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def generate_text(self, prompt:str, chat_history:list=None, 
                      max_output_token:int=None, temperature:float=None):
        if not self.client:
            self.logger.error("OpenAI client wasn't set ")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI client wasn't set ")
            return None

        max_output_token = max_output_token if max_output_token else self.default_generation_max_output_token     
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history = chat_history or []
        chat_history.append(self.construct_prompt(prompt=prompt, role=OpenAI_Enums.USER.value))

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=chat_history,
                max_tokens=max_output_token,
                temperature=temperature
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                self.logger.error("OpenAI returned a successful response, but the text is empty.")
                return None

            chat_history.append(self.construct_response(response=response))

            return content

        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return None

    def embed_text(self, text: Union[str, List[str]], input_type: str = None, document_type: str = None, **kwargs):
        if not self.client:
            self.logger.error("OpenAI client wasn't set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI client wasn't set")
            return None

        if isinstance(text, str):
            text = [text]

        extra_body = {"input_type": input_type} if input_type else None

        response = self.client.embeddings.create(
            model=self.embedding_model_id,
            input=text,
            encoding_format="float",
            extra_body=extra_body
        ) 

        if not response or not response.data or len(response.data) == 0:
            self.logger.error("Error while embedding text with OpenAI")
            return None

        return [r.embedding for r in response.data]

    def construct_prompt(self, prompt:str, role:str):
        return {"role":role, 
                "content":prompt
                }

    def process_text(self, text:str):
        return text[:self.default_generation_max_output_characters].strip()

    def construct_response(self, response):
        return {
            "role" : OpenAI_Enums.ASSISTANT.value,
            "content" : response.choices[0].message.content      
        }
