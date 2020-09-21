# 工业数据的RUL时间序列预测建模实践

### &diams; 项目描述
### &diams; 数据说明
### &diams; 基于keras的lstm的时间序列建模


### 1. 项目描述


在工业制造过程中，RUL（Remaining Useful Life，剩余使用寿命），主要指系统运行一段时间后，剩余的使用寿命，准确地预测系统的剩余使用寿命，可以大大地减少因系统当机引起的损失，提高系统的运行可靠性。为了根据历史的RUL变化情况，准确预测未来系统的RUL值，本文通过使用LSTM深度学习网络，建立时间序列模型，显示生产制程系统的RUL预测。

数据与代码：https://github.com/EwdAger/Industrial-Data-Time-Servies-Prediction

### 2. 数据说明

通过整理历史数据，得到如下的时间序列数据(展示部分)，其中，id列为时间序列ID，RUL列为需要预测建模的列，共计有连续的20631个时间序列样本点：

![blockchain](https://github.com/EwdAger/Industrial-Data-Time-Servies-Prediction/blob/master/docs/picture/%E5%8E%9F%E5%A7%8B%E6%95%B0%E6%8D%AE%E5%BA%8F%E5%88%97.png)

绘制时序图如下所示：

![时间序列图](https://github.com/EwdAger/Industrial-Data-Time-Servies-Prediction/blob/master/docs/picture/%E5%8E%9F%E5%A7%8B%E5%BA%8F%E5%88%97.png)


### 3. 基于keras的lstm的时间序列建模

#### 3.0  lstm算法原理

[参见链接](https://www.gvoidy.cn/posts/e4e448be/)

#### 3.1 数据样本制作
时间样本对制作，时间序列预测使用通过历史数据来预测将来的数据序列，通过将原始数据制作为X-Y的样本对，用来输入到深度学习模型中，样本制作变换形式如下所示：

![样本对制作](https://github.com/EwdAger/Industrial-Data-Time-Servies-Prediction/blob/master/docs/picture/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%B6%E4%BD%9C%E5%9B%BE.png)
```
class DataLoader():
    """A class for loading and transforming data for the lstm_check_point model"""

    def __init__(self, filename, split, cols_X, cols_Y):
        # read csv
        dataframe = pd.read_csv(filename)
        # 数据集
        data_train_X = dataframe.get(cols_X).values
        data_train_Y = dataframe.get(cols_Y).values
        # X规范化
        scaler_X = StandardScaler().fit(data_train_X)
        data_train_X = scaler_X.transform(data_train_X)
        # Y规范化
        self.scaler_Y = StandardScaler().fit(data_train_Y)
        data_train_Y = self.scaler_Y.transform(data_train_Y)
        # 数据集，供方法使用
        i_split = int(len(dataframe) * split)
        # 自身列预测自身
        if len(cols_X)==1 and cols_X[0] == cols_Y[0]:
            self.data_train = data_train_X[:i_split]
            self.data_test = data_train_X[i_split:]
        else:
            self.data_train = np.hstack((data_train_Y, data_train_X))[:i_split]
            self.data_test = np.hstack((data_train_Y, data_train_X))[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_test - seq_len):
            x, y = self._next_window(i, seq_len, "test")
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def get_train_data(self, seq_len):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, "train")
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)
```

#### 3.2  模型建立

##### 3.2.1 代码
```
class Model():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def build_model(self, configs):
		timer = Timer()
		timer.start()

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')
		timer.stop()
```
##### 3.2.2.tensorboard模型图

![tensorboard模型图](https://github.com/EwdAger/Industrial-Data-Time-Servies-Prediction/blob/master/docs/picture/tensorboard.png)


#### 3.3  模型结果

运行代码，等到如下预测效果图（r2 score=0.956）：

![结果图](https://github.com/EwdAger/Industrial-Data-Time-Servies-Prediction/blob/master/docs/picture/%E7%BB%93%E6%9E%9C%E5%9B%BE.png)

### &diams; 基于keras的GRU的时间序列建模

#### GRU 算法原理

[参见链接](https://www.gvoidy.cn/posts/7394ddd0/)

数据样本制作、预处理与LSTM同理，仅将网络结构改为GRU。预测效果如下图

![结果图](https://github.com/EwdAger/Industrial-Data-Time-Servies-Prediction/blob/master/docs/picture/GRU_result.png)