import pandas as pd
from PIL import Image
from torchvision.transforms.functional import to_tensor

from inid.data import DataFrameDataset


def test_data_frame_dataset(tmp_path):
    df = pd.DataFrame([
        {'my_image': str(tmp_path / 'A_0000.jpg'), 'my_label': 'A'},
        {'my_image': str(tmp_path / 'A_0001.jpg'), 'my_label': 'A'},
        {'my_image': str(tmp_path / 'B_0000.jpg'), 'my_label': 'B'},
        {'my_image': str(tmp_path / 'B_0001.jpg'), 'my_label': 'B'},
    ])

    for i, img_path in enumerate(df['my_image']):
        c = int(i * 255 / len(df))
        Image.new('RGB', (2, 2), (c, c, c)).save(img_path)

    label_to_int = {
        'A': 0,
        'B': 1,
    }

    ds = DataFrameDataset(df, label_key='my_label', image_key='my_image',
                          label_to_int=label_to_int, transform=None)

    assert ds.unique_labels == {'A', 'B'}

    assert len(ds) == len(df)

    for i in range(len(df)):
        ret_im, ret_label = ds[i]

        exp_im = Image.open(df.loc[i, 'my_image'])
        exp_label = label_to_int[df.loc[i, 'my_label']]

        assert (to_tensor(ret_im) == to_tensor(exp_im)).all()
        assert ret_label == exp_label
