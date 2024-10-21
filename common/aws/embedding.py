import boto3
import json


class BedrockEmbedding():
    def __init__(self, region='us-west-2'):
        self.region = region
        self.bedrock = boto3.client(
            service_name = 'bedrock-runtime',
            region_name = self.region,
            endpoint_url=f"https://bedrock-runtime.{self.region}.amazonaws.com"
        )
    
    '''
    Multimodal Embedding
    '''
    def embedding_multimodal(self, text=None, image=None):
        body = dict()
        if text is not None: body['inputText'] = text
        if image is not None: body['inputImage'] = image

        try:
            res = self.bedrock.invoke_model(
                body=json.dumps(body),
                modelId='amazon.titan-embed-image-v1',
                accept="application/json",
                contentType="application/json"
            )
            return json.loads(res.get("body").read()).get("embedding")
        except Exception as e:
            print(e)
            return []


    '''
    Text Embedding
    '''
    def embedding_text(self, text=None):
        body = dict()
        if text is not None: body['inputText'] = text
        
        try:
            res = self.bedrock.invoke_model(
                body=json.dumps(body),
                modelId='amazon.titan-embed-text-v2:0',
                accept="application/json",
                contentType="application/json"
            )
            return json.loads(res.get("body").read()).get("embedding")
        except Exception as e:
            print(e)
            return []
