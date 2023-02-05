from datetime import datetime


class Config:
    def __init__(self):
        self.seed = 1

        """Data"""
        self.data_path = r"../../Data/bag_level/"
        self.train_path = self.data_path + "train.txt"
        self.max_bag_size = 8
        self.test_path = self.data_path + "test.txt"
        self.label_path = self.data_path + "label.txt"
        self.PN_test_path = self.data_path + "PN_test.txt"

        """Embeddings"""
        # pretrained
        self.use_plm = True
        self.pretrained_path = r"../../PLMs/bert_based_chinese/"
        self.pretrained_vocab_path = self.pretrained_path + "vocab.txt"
        # random
        self.vocab_dict_path = r""
        self.vocab_size = -9999

        """Process"""
        self.limit_size = 80
        self.max_len = 80
        self.classes_num = 33

        """Model"""
        if self.use_plm:
            self.embedding_size = 768
        else:
            self.embedding_size = 300
        self.pos_dim = 50
        self.filter_size = 3
        self.num_filters = 128
        self.dropout_rate = 0.5

        """Train and Evaluate"""
        self.bert_lr = 2e-5
        self.lr = 2e-3
        self.weight_decay = 1e-5
        self.epochs = 64
        self.batch_size = 64
        self.patients = 5

        """checkpoints and result"""
        self.model_name = "model" + \
                          "_" + \
                          datetime.now().strftime('%Y-%m-%d-%H:%M:%S').replace('-', '_').replace(':', '_') + ".ckpt"
        self.save_model_path = r"checkpoints/" + self.model_name

        self.pr_curve_result_path = r"result/" + self.model_name + '.txt'

    def __str__(self):
        ans = ""
        for n, v in vars(self).items():
            ans += n + "=" + str(v) + "\n"
        return ans
