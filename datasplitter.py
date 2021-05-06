'''
In this code I am assuming that I have data from multiple sources. All the dataframe (data from multiple sources) should have x_row_name,y_row_name,dataset_name. In case of segmentation or some classes where we don't have exact classes we can create a new column in all the dataframe name as mask which tells whether the class is true or positive.(It will act as label.) We are adding it because we will be doing stratified split.


##Future works
> We can add data prevalance option in the splitter. Right now the stratified split itself picks up the class prevalance and based on it does the train,test and val split. But in future we can add it.
'''


class DataSplitter:

    def __init__(self,
                list_of_dataframe,
                x_col_name,
                y_col_name,
                dataset_col_name,
                train,
                val,
                test):
        '''
        list_of_dataframe --> List containing all the dataframes across different datasets.
        Here we are assuming that one dataframe refers to one dataset.
        x_col_name --> Colname for the sample (It should be similar across all the dataframes)
        y_col_name --> Binary class (Since we are planning to use it in segmentation, column name should be similar to all the dataframe)
        dataset_col_name --> Column name in which dataset name is mentioned.
        train --> Dictionary Ratio of training data for all the dataframe
        val --> Dictionary Ratio of validation data to keep
        test --> Ratio of test data to keep.
        '''

        self.df_list = dic_of_dataframe
        self.x_name = x_col_name
        self.y_name = y_col_name
        self.dataset_col_name = dataset_col_name
        self.train_ratio_dic = train
        self.val_ratio_dic = val
        self.test_ratio_dic = test


    def split_the_data(self):
        '''
        All the data split are going to be stratified splits.
        In this we will add a column name dataset split. That column will tell us whether the following image lies
        in the training set or the test set or the val set(based on the ratio we have given)
        So we will have a for loop which will iterate over the list of dataframe.
        Now after that we want to do stratified data split.
        '''
        '''
        test_size=0.2, random_state=0,
                               stratify=df_nih['pnuemothorax_mask'])

        '''
        self.splits_df_list = {}
        for dataset_name,df in self.df_list.items():
            temp_df = {}
            temp_df['train'], test = train_test_split(df,test_size=1-self.train_ratio_dic[dataset_name],shuffle=True,random_state=7,stratify=df[self.y_name])
            print(test)
            temp_df['test'],temp_df['val'] = train_test_split(test,test_size=self.test_ratio_dic[dataset_name]/(self.test_ratio_dic[dataset_name] + self.val_ratio_dic[dataset_name]),shuffle=True,random_state=7,stratify=test[self.y_name])
            self.splits_df_list[dataset_name] = temp_df


    def merge_data_frame(self):
        '''

        updated_list_of_dataframe with splits will be created in the split the data function.
        Once that is done, we can merge the dataframe and we can create train,val and test data.
        '''
        self.df_train = None
        self.df_val = None
        self.df_test = None
        count = 0
        for dataset_name, df in self.splits_df_list.items():
            if count == 0:
                self.df_train = pd.DataFrame()
                self.df_val = pd.DataFrame()
                self.df_test = pd.DataFrame()

            self.df_train = pd.concat([self.df_train,df['train']])
            self.df_val = pd.concat([self.df_val,df['val']])
            self.df_test = pd.concat([self.df_test,df['test']])

            self.df_train = self.df_train.reset_index(drop=True)
            self.df_val = self.df_val.reset_index(drop=True)
            self.df_test = self.df_test.reset_index(drop=True)

            count += 1



'''
Sample way of sending input to the class =>
dic_of_dataframe = {
    'NIH' : df_nih[['image_path','dataset','pnuemothorax_mask','pneumothorax']],
    'MIMIC':df_mimic[['image_path','dataset','pnuemothorax_mask','pneumothorax']],
    'CHEXPERT':df_chexpert[['image_path','dataset','pnuemothorax_mask','pneumothorax']],
    'KAGGLE_TRAIN':df_kaggle_train[['image_path','dataset','pnuemothorax_mask','pneumothorax']],
    'KAGGLE_TEST':df_kaggle_test[['image_path','dataset','pnuemothorax_mask','pneumothorax']]
}

x_col_name = 'image_path'
y_col_name = 'pnuemothorax_mask'
dataset_col_name = None

train = {
    'NIH':0.70,
    'MIMIC':0.70,
    'CHEXPERT':0.70,
    'KAGGLE_TRAIN':0.70,
    'KAGGLE_TEST':0.70
}

val = {
    'NIH':0.15,
    'MIMIC':0.15,
    'CHEXPERT':0.15,
    'KAGGLE_TRAIN':0.15,
    'KAGGLE_TEST':0.15
}

test = {
    'NIH':0.15,
    'MIMIC':0.15,
    'CHEXPERT':0.15,
    'KAGGLE_TRAIN':0.15,
    'KAGGLE_TEST':0.15
}

dd = DataSplitter(dic_of_dataframe,x_col_name,y_col_name,dataset_col_name,train,val,test)
'''
