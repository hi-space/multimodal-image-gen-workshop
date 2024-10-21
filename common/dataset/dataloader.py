import boto3
import json
import pandas as pd
from enum import Enum
from io import BytesIO
from PIL import Image
from common.utils.images import display_image, encode_image_base64


class LanguageTag(Enum):
    ENG = 'en_US'
    KOR = 'ko_KR'


class DataLoader():
    def __init__(self, index=0, language=LanguageTag.ENG):
        if index < 0 or index > 9:
            raise ValueError("Index must be between 0 and 9.")
        
        self.index = index
        self.language = language

        self.image_meta = self._get_image_meta()
        self.item_meta = self._get_item_meta()
        self.dataset = self._make_dataset_with_image()
        self.s3_client = boto3.client('s3')

    def show_item(self, item_id, detail=False) -> dict:
        item, img = self.get_item(item_id=item_id)

        print(f"[{item.get('item_id')}] {item.get('item_name')}")
        display_image(encode_image_base64(img))
        
        if detail:
            print(json.dumps(item, indent=4, ensure_ascii=False))
            for img in item['other_image_id']:
                display_image(encode_image_base64(self.get_image(img)))

        return item, img

    def get_random_id(self) -> str:
        return self.dataset['item_id'].sample(n=1).iloc[0]
                
    def get_item(self, item_id):
        item_row = self.dataset[self.dataset['item_id'] == item_id]
        
        if not item_row.empty:
            json_str = item_row.to_json(orient="records")
            item = json.loads(json_str)[0]
            img = self.get_image(image_id=item.get('main_image_id'))
            return item, img
        else:
            return None
        
    def get_image(self, image_id):
        try:
            row = self.image_meta[self.image_meta['image_id'] == image_id]
            
            if not row.empty:
                row = row.iloc[0]
                response = self.s3_client.get_object(
                    Bucket="amazon-berkeley-objects",
                    Key=f"images/original/{row['path']}"
                )
                image_content = response['Body'].read()
                return Image.open(BytesIO(image_content))
        except Exception as e:
            print(e)
            return None

    def get_image_info(self, image_id):
        try:
            row = self.image_meta[self.image_meta['image_id'] == image_id]
            
            if not row.empty:
                row = row.iloc[0]
                response = self.s3_client.get_object(
                    Bucket="amazon-berkeley-objects",
                    Key=f"images/small/{row['path']}"
                )
                image_content = response['Body'].read()
                image = Image.open(BytesIO(image_content))
                
                return {
                    'image': image,
                    'image_id': row['image_id'],
                    'height': int(row['height']),
                    'width': int(row['width']),
                    'path': row['path']
                }
        except Exception as e:
            print(e)
            return {}
        
    def get_id_list(self):
        return self.dataset['item_id'].tolist()

    def _make_dataset_with_image(self):
        # extract only english field using list comprehension
        processed_data = pd.DataFrame([self._extract_language_item(row) for _, row in self.item_meta.iterrows()])

        # Remove NaN items by replacing them with empty strings or appropriate defaults
        processed_data.fillna('', inplace=True)

        # Remove items if 'item_name' is empty
        processed_data = processed_data[processed_data['item_name'] != '']

        # Merge with image metadata
        dataset = processed_data.merge(self.image_meta, left_on="main_image_id", right_on="image_id", how="left")

        # Assign the image path based on the main_image_id
        dataset.rename(columns={'path': 'img_path'}, inplace=True)

        self.dataset = dataset
        return dataset
    
    def _extract_language_item(self, row):
        def process_language_field(field, default_value):
            if isinstance(field, list):
                return next((item['value'] for item in field if item.get('language_tag') == self.language.value), default_value)
            return field if field is not None else default_value
            
        return {
            'item_id': row.get('item_id', ''),
            'bullet_point': process_language_field(row.get('bullet_point'), ''),
            'item_name': process_language_field(row.get('item_name'), ''),
            'product_type': [item.get('value', '') for item in row.get('product_type', [])] if isinstance(row.get('product_type'), list) else [],
            'main_image_id': row.get('main_image_id', ''),
            'other_image_id': row.get('other_image_id', []),
            'node': [item.get('node_name', '') for item in row.get('node', [])] if isinstance(row.get('node'), list) else [],
        }

    def _get_item_meta(self):
        return pd.read_json(f"s3://amazon-berkeley-objects/listings/metadata/listings_{self.index}.json.gz", lines=True)

    def _get_image_meta(self):
        return pd.read_csv("s3://amazon-berkeley-objects/images/metadata/images.csv.gz")
    
    def _get_image_path(self, image_path, origin_size=True):
        if origin_size:
            return f"s3://amazon-berkeley-objects/images/original/{image_path}"
        return f"s3://amazon-berkeley-objects/images/small/{image_path}"