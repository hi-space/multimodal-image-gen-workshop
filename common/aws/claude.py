import boto3
import json
from botocore.config import Config

from langchain.prompts import PromptTemplate
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

from langchain_aws.chat_models import ChatBedrock
from langchain.callbacks import StdOutCallbackHandler, StreamingStdOutCallbackHandler


class BedrockClaude():
    def __init__(self, region='us-west-2', modelId = 'anthropic.claude-3-5-sonnet-20240620-v1:0', **model_kwargs):
        self.region = region
        self.modelId = modelId
        self.bedrock = boto3.client(
            service_name = 'bedrock-runtime',
            region_name = self.region,
            config = Config(
                connect_timeout=120,
                read_timeout=120,
                retries={'max_attempts': 5}
            ),
        )

        # https://docs.aws.amazon.com/ko_kr/bedrock/latest/userguide/model-parameters.html?icmpid=docs_bedrock_help_panel_playgrounds
        self.model_kwargs = {
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 4096, # max tokens
            'temperature': 0.1, # [0, 1]
            'top_p': 0.9, # [0, 1]
            'top_k': 250, # [0, 500]
            'stop_sequences': ['Human:', 'H: ']
        }

        self.model_kwargs.update(model_kwargs)

    '''
    Langchain API: get ChatBedrock
    '''
    def get_chat_model(self, callback=StdOutCallbackHandler(), streaming=True):
        return ChatBedrock(
            model_id = self.modelId,
            client = self.bedrock,
            streaming = streaming,
            callbacks = [callback],
            model_kwargs = self.model_kwargs,
        )
    

    '''
    Bedrock API: invoke LLM model
    '''    
    def invoke_llm(self, text: str, image: str = None, system: str = None):
        '''
        Returns:
            dict: ['id', 'type', 'role', 'content', 'model', 'stop_reason', 'stop_sequence', 'usage']
        '''
        parameter = self.model_kwargs.copy()
      
        content = []
        # text
        if text:
            content.append({
                'type': 'text',
                'text': text,
            })
        
        # image
        if image:
            content.append({
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/webp',
                    'data': image,
                }
            })

        # system
        if system:
            parameter['system'] = system
        
        parameter.update({
            'messages': [{
                'role': 'user',
                'content': content
            }]
        })

        try:
            response = self.bedrock.invoke_model(
                body=json.dumps(parameter),
                modelId=self.modelId,
                accept='application/json',
                contentType='application/json'
            )
            return json.loads(response.get('body').read())
        except Exception as e:
            print(e)
            return None
        
    def invoke_llm_response(self, text: str, image: str = None, system: str = None):
        return self.invoke_llm(
            text=text, image=image, system=system).get('content', [])[0].get('text', '')

    def get_prompt(self, text: str = "그림을 상세히 묘사해줘", image: str = None):
        content = []

        if image:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/webp;base64,{image}",
                },
            })

        content.append({
            "type": "text",
            "text": text
        })

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(
                content=content
            )
        ]

        return messages
