from fastai.vision import *

__all__ = ["SemanticSegmentationData"]


class SemanticSegmentationData:
    """
    Creates semantic segmentation data,masks should be already numericalized.
    Normalization is not done here, can be different in different TL settings.
    Test set is always assumed to be without labels, otherwise use valid.
    """
    def __init__(self, PATH, CODES, TRAIN, VALID, TEST, sample_size, bs, size, has_test_labels=True):
        """
        path: path to data folder
        codes: txt file which has segmentation pixel codes
        sample_size: training sample size, None for all
        bs: batch size
        size: image size
        """
        
        self.path, self.sample_size, self.bs, self.size, self.has_test_labels  =\
                                            PATH, sample_size, bs, size, has_test_labels
        self.codes = np.loadtxt(self.path/CODES, dtype=str)
        
        self.train_df = pd.read_csv(self.path/TRAIN, header=None)
        if VALID is not None: self.valid_df = pd.read_csv(self.path/VALID, header=None)
        if TEST is not None: self.test_df = pd.read_csv(self.path/TEST, header=None)
        
        self.path_img = self.path/"images"
        self.path_lbl = self.path/"masks"
        
        self.VALID, self.TEST = VALID, TEST
        
    def get_y_fn(self, x): return self.path_lbl/f'{Path(x).stem}.png'
        
    def get_data(self):        
        if self.VALID: 
            self.train_valid_df = pd.concat([self.train_df, self.valid_df])
            self.train_valid_df.columns = ["images"]
            self.train_valid_df["is_valid"] = len(self.train_df)*[False] + len(self.valid_df)*[True]
        else:
            self.train_valid_df = self.train_df
        
        il = SegmentationItemList.from_df(self.train_valid_df, self.path, folder="images") # get
        if self.VALID: ill = il.split_from_df("is_valid") # split
        else: ill = il.split_by_rand_pct() # split
        ll = ill.label_from_func(self.get_y_fn, classes=self.codes) # label
            
        data = (ll.transform(get_transforms(), size=(self.size, self.size), tfm_y=True,
                             resize_method=ResizeMethod.SQUISH)
                    .databunch(bs=self.bs))
        # add_test
        if self.TEST:
            il = SegmentationItemList.from_df(self.test_df, self.path, folder="images") # get
            data.add_test(il, tfm_y=False)
        return data
        
    def __repr__(self):
        return f"""___repr__"""
    
    def __str__(self):
        return f"""___str___"""
    