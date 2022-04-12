from xml.dom import ValidationErr
from torch.utils.data import Dataset
from pathlib import Path


class TripletData(Dataset):
    def __init__(self, path):
        super(TripletData, self).__init__()
        self.data = [txt for txt in Path(path).glob('*.txt')]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class custom_collate(object):
    def __init__(self, ):
        pass

    def __call__(self, batch):
        """
        preprocess batch
        
        ret (list[triplet])
        """
        
        ret = []
        for txt_file in batch:
            with txt_file.open('r', encoding='utf8') as f:
                data = f.read()
            # triplet: (query, positive, negative)
            triplet = data.split('\n\n\n')
            if len(triplet) != 3:
                print(f'[Warning] len(triplet) != 3, skipped this data [txt_file]') 
                #raise ValidationErr
            ret.append(triplet)
        
        return ret
