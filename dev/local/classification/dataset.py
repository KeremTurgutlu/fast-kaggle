#AUTOGENERATED! DO NOT EDIT! File to edit: dev/20_classification_dataset.ipynb (unless otherwise specified).

__all__ = ['ImageClassificationData']

#Cell
from fastai.vision import *

#Cell
class ImageClassificationData:
    "Creates image classification dataset from fastai datablock API"
    def __init__(self,PATH,IMAGES,LABELS,TRAIN,VALID,TEST,
                 is_multilabel,sample_size,bs,size,**dl_kwargs):
        # input params
        self.path, self.sample_size, self.bs, self.size  = \
        PATH, sample_size, bs, size
        self.label_cls = partial(MultiCategoryList, label_delim=";") if is_multilabel else CategoryList
        self.VALID, self.TEST = VALID, TEST
        self.dl_kwargs = dl_kwargs

        # read training
        self.train_df = pd.read_csv(self.path/TRAIN, header=None)
        if sample_size: self.train_df = self.train_df.sample(sample_size)

        # read validation and test
        if (VALID is not None) and (type(VALID) is str):
            self.valid_file = True
        else:
            self.valid_file = False
        if self.valid_file: self.valid_df = pd.read_csv(self.path/VALID, header=None)
        if TEST is not None: self.test_df = pd.read_csv(self.path/TEST, header=None)

        # read labels
        self.labels_df = pd.read_csv(self.path/LABELS)

        # image folder
        self.path_img = self.path/IMAGES

    def get_data(self):
        if self.valid_file:
            self.train_valid_df = pd.concat([self.train_df, self.valid_df])
            self.train_valid_df.columns = ["images"]
            self.train_valid_df["is_valid"] = len(self.train_df)*[False] + len(self.valid_df)*[True]
        else:
            self.train_valid_df = self.train_df

        # get
        il = SegmentationItemList.from_df(self.train_valid_df, self.path_img)
        # split
        if self.valid_file: ill = il.split_from_df("is_valid")
        else: ill = il.split_by_rand_pct(ifnone(self.VALID, 0.2))
        # label
        labels_dict = dict(zip(self.labels_df.iloc[:,0], self.labels_df.iloc[:,1]))
        ll = ill.label_from_func(lambda o: labels_dict[Path(o).name], label_cls=self.label_cls)
        # databunch
        data = (ll.transform(get_transforms(),
                             size=self.size,
                             tfm_y=False,
                             resize_method=ResizeMethod.SQUISH)
                    .databunch(bs=self.bs, **self.dl_kwargs))
        # test
        if self.TEST:
            il = ImageList.from_df(self.test_df, self.path_img) # get
            data.add_test(il, tfm_y=False)
        return data

    def __repr__(self):
        return f"""___repr__"""

    def __str__(self):
        return f"""___str___"""