
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from dotenv import load_dotenv
from utils.config_loader import load_config
import os
import sys

#define logger
log=CustomLogger().get_logger(__name__)

class ModelLoader:
    def __init__(self):
        #load credentials
        load_dotenv(override=True)
        self.validate_env()
        self.config=load_config("config/config.yaml")
        log.info("load configration successfully",config_keys=list(self.config.keys()))
    
    def validate_env(self):
        "Ensure API keys exists"
        required_vars=['OPENAI_API_KEY','GOOGLE_API_KEY','GROQ_API_KEY']
        self.api_keys={key:os.getenv(key) for key in required_vars}
        missing=[k for k, v in self.api_keys.items() if not v ]

        if missing:
            log.error("missing environment variable",missing_vars=missing)
            raise DocumentPortalException("Missing environment variable",sys)
    
    def load_embedding_model(self):
        "load enbedding model"
        try:
            log.info("load Embedding model...")
            model_name=self.config['embedding_model']['model_name']
            return OpenAIEmbeddings(model=model_name)
        except Exception as e:
            log.error("error in loading embedding model",error=str(e))

    def load_llm(self):
        "load and return the model"
        llm_block = self.config["llm"]

        # set default provider
        provider_key = os.getenv("LLM_PROVIDER", "openai")

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider_key=provider_key)
            raise ValueError(f"provider {provider_key} not found in config file")
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get('model_name')
        temperature = llm_config.get('temperature', 0.2)
        max_output_tokens = llm_config.get('max_output_tokens', 2048)

        log.info("Load LLM Model Sucessfully", provider=provider, model=model_name)

        if provider == "google":
            return GoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_keys['GOOGLE_API_KEY'],
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )

        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_keys['GROQ_API_KEY'],
                temperature=temperature,
                max_tokens=max_output_tokens
            )

        elif provider == "openai":
            return ChatOpenAI(
                model=model_name,
                api_key=self.api_keys['OPENAI_API_KEY'],
                temperature=temperature,
                max_tokens=max_output_tokens
            )

        else:
            log.error("Unsupported llm provider", provider=provider)
            raise ValueError(f"Unsupported llm provider {provider}")


#if __name__=="__main__":
 #  ModelLoader().validate_env()
 #  embedding_model=ModelLoader().load_embedding_model()
  # result=embedding_model.embed_query("what is capital of india?")
  # print("embedding_model result",result)

  # llm=ModelLoader().load_llm()
  # result=llm.invoke("what is capital of india?")
  # print("llm result",result.content)

   
   



