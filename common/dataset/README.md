# [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html)

## Description

Amazon Berkeley Objects is a collection of 147,702 product listings with multilingual metadata and 398,212 unique catalog images. 8,222 listings come with turntable photography (also referred as *spin* or *360º-View* images), as sequences of 24 or 72 images, for a total of 586,584 images in 8,209 unique sequences. For 7,953 products, the collection also provides high-quality 3d models, as glTF 2.0 files.

## How to Use

It is a collection of product listings with multilingual metadata, catalog imagery, high-quality 3d models with materials and parts, and benchmarks derived from that data.

The `DataLoader` class provides a convenient way to extract necessary metadata and images. In particular, multilingual metadata is structured in nested JSON format with values corresponding to country codes. To facilitate easy data usage, we process it to extract values for specific languages only.

You can load datasets using the [`DataLoader`](./dataloader.py) class.

### Load Dataset

Through the `dataloader`, you can retrieve the metadata of products and product images in a single DataFrame format. You can load a dataset by selecting its index, which is provided as a value between 0 and 9. It provides an average of 1653 items for `ENG` and 554 for `KOR` per index. It takes less than 10 seconds to load the initial data.

```python
from dataloader import DataLoader, LanguageTag

loader = DataLoader(index=0, language=LanguageTag.KOR)
```

This dataloader processes the existing dataset items through the following steps

1. Originally, multiple language values are nested in a single field. It removes data in other languages and sets only specific language tag values as representative values. You can select language tags as follows, and add more if needed

    ```python
    class LanguageTag(Enum):
        ENG = 'en_US'
        KOR = 'ko_KR'
    ```

2. It removes the nested structure where JSON objects are included in a list and represents them as a list string.

3. Items without an `item_name` are removed.

4. It joins the metadata of product images using the corresponding image ID.
   - Image metadata includes `image_id`, `height`, `width`, and `img_path`.

### Get a item by ID

```py
# item (dict), img (PIL.Image)
item, img = dataloader.get_item(item_id="B07F3F8B3Y")
print(json.dumps(item, indent=4))
```

```json
// Example of item
{
    "item_id": "B082VSXML3",
    "bullet_point": "66.14\"W x 36\"D x 28.94\"H",
    "item_name": "Amazon Brand - Ravenna Home Archer Outdoor Patio Steel Dining Table with Panel Top, 66.14\"W, Gray",
    "product_type": [
        "TABLE"
    ],
    "main_image_id": "616rLGcketL",
    "other_image_id": [
        "71eyLZCzLxL",
        "61DzrseC8bL",
        "817ondBaLoL",
        "61nXRGnI9iL",
        "81GT7PvQLtL",
        "61QdXiH+iBL",
        "61di7+0FpzL"
    ],
    "node": [
        "/Categories/Patio Furniture & Accessories/Tables/Dining Tables"
    ],
    "image_id": "616rLGcketL",
    "height": 2000.0,
    "width": 2000.0,
    "img_path": "3d/3d7aaf17.jpg"
}
```
