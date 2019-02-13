# DropBlock

- Dropout: 完全隨機丟棄 neuron
- Sparital Dropout: 按 channel 隨機丟棄
- Stochastic Depth: 按 res block 隨機丟棄
- DropBlock: 每個 feature map 上按 spatial square 隨機丟棄
- Cutout: 在 input layer 按 spatial square 隨機丟棄
- DropConnect: 只在連接處丟，不丟 neuron
- [DropBlock](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650751601&idx=5&sn=6ba09bea3acb116eb9f4902af5261e72&chksm=871a860fb06d0f194c4c0452e53d21cc6c537b4e33a5aea4e3c0a067db0c46d41168afabcc0c&scene=21#wechat_redirect)
    - Idea
        - 一般的 Dropout 都是用在 fully connection layer，而在 convolutional network 上使用 dropout 的意義並不大，該文章則認為因為在每一個 feature maps 的位置都具有一個 [receptive field](https://zhuanlan.zhihu.com/p/28492837)，僅對單一像素位置進行 dropout 並不能降低 feature maps 學習特徵範圍，也就是說，network 能夠特過**相鄰位置**的特徵值去學習，也不會特別加強去學習保留下來的訊息。既然對於單獨的對每個位置進行 dropout 並無法提高 network 本身的泛化能力，那就以區塊的概念來進行 dropout，反而更能讓 network 去學習保留下來的訊息，而加重特徵的權重。
    - Method
        - 不同 feature maps 共享相同的 dropblock mask，在相同的位置丟棄訊息
        - 每一層的 feature maps 使用各自的 dropblock mask
    - Parameters
        - block size: 控制要讓 value of feature maps 歸為 0 的區塊大小
        - $ \gamma $: 用來控制要丟棄特徵的數量
        - keep_prob: 與 dropout 的參數相同
    - Code implement 
        - https://github.com/DHZS/tf-dropblock/blob/master/nets/dropblock.py
        - https://github.com/shenmbsw/tensorflow-dropblock/blob/master/dropblock.py 	

Bernoulli distrubtion:

```{python}
import tensorflow as tf
tf.reset_default_graph()
with tf.Graph().as_default() as g: 
    mean = tf.placeholder(tf.float32, [None])
    input_shape = tf.placeholder(tf.float32, [None, 4, 4, 3])
    shape = tf.stack(tf.shape(input_shape))
    # method 1
    # 用 uniform distributions 產生值，再透過 sign 轉為 [-1, 1], 最後透過 relu 將 -1 轉換為 0
    uniform_dist = tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    sign_dist = tf.sign(mean - uniform_dist)
    bernoulli = tf.nn.relu(sign_dist)
    # method 2
    # probs 可以為多個 p, 對應 shape, 產生 n of p 的 bernoulli distributions
    noise_dist = tf.distributions.Bernoulli(probs=[0.1])
    mask = noise_dist.sample(shape)
    
with tf.Session(graph=g) as sess:
    tmp_array = np.zeros([4, 4, 3], dtype=np.uint8) 
    tmp_array[:,:100] = [255, 0, 0] #Orange left side array[:,100:] = [0, 0, 255] #Blue right side
    batch_array = np.array([tmp_array]*3)
    uniform, sign, bern = sess.run([uniform_dist, sign_dist, bernoulli], feed_dict={mean: [1.], input_shape:batch_array})
    
```

DropBlock implement:

```{python}
# DropBlock
import tensorflow as tf
from tensorflow.python.keras import backend as K
class DropBlock(tf.keras.layers.Layer) :
    def __init__(self, keep_prob, block_size, **kwargs):
        super(DropBlock, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def build(self, input_shape):
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        bottom = right = (self.block_size -1) // 2
        top = left = (self.block_size -1) - bottom
        self.padding = [[0, 0], [top, bottom], [left, right], [0, 0]]
        self.set_keep_prob()
        super(DropBlock, self).build(input_shape)
        
    def set_keep_prob(self, keep_prob=None):
        """This method only support Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.to_float(self.w), tf.to_float(self.h)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0], 
                                        self.h - self.block_size + 1, 
                                        self.w - self.block_size + 1,
                                        self.channel])
        mask = DropBlock._bernoulli(sampling_mask_shape, self.gamma)
        # 擴充行列，並給予0值，依據 paddings 參數給予的上下左右值來做擴充，mode有三種模式可選，可參考 document
        mask = tf.pad(tensor=mask, paddings=self.padding, mode='CONSTANT') 
        mask = tf.nn.max_pool(value=mask, 
                              ksize=[1, self.block_size, self.block_size, 1], 
                              strides=[1, 1, 1, 1], 
                              padding='SAME')
        mask = 1 - mask
        return mask
        
    @staticmethod    
    def _bernoulli(shape, mean):
        return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)))
    
    # The call function is a built-in function in 'tf.keras'.
    def call(self, inputs, training=None, scale=True, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale,
                             true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output
        
        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output
```

```{python}
# Testing
a = tf.placeholder(tf.float32, [None, 5, 5, 3])
keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

drop_block = DropBlock(keep_prob=keep_prob, block_size=3)
b = drop_block(inputs=a, training=training)

sess = tf.Session()
feed_dict = {a: np.ones([2, 5, 5, 3]), keep_prob: 0.8, training: True}
c = sess.run(b, feed_dict=feed_dict)

print(c[0, :, :, 0])
```

- [Targeted Dropout](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650752571&idx=1&sn=8417645148afd8eebdb79c91b37a7409&chksm=871a8245b06d0b53115d79f1ce42bc5a03aad5d038fe51c2f237c5848c41c51c5b756aaa8937&scene=21#wechat_redirect)

Reference:

1. https://cloud.tencent.com/developer/article/1367373