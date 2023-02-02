from datetime import datetime


class Config:
    def __init__(self):
        self.seed = 1

        """Data"""
        self.data_path = r"../../Data/sentence_level/"
        self.train_path = self.data_path + "aug_train.json"
        self.test_path = self.data_path + "test.json"
        self.label_path = self.data_path + "label.txt"
        self.max_bag_size = 12

        """Embeddings"""
        self.pretrained_path = r"../../PLMs/bert_based_chinese/"
        self.pretrained_vocab_path = self.pretrained_path + "vocab.txt"

        """Process"""
        self.limit_size = 80
        self.max_len = 80
        self.classes_num = 33

        """Model"""
        self.embedding_size = 768
        self.hidden_size = 256
        self.pos_dim = 50
        self.dropout_rate = 0.5

        """Train and Evaluate"""
        self.bert_lr = 2e-5
        self.lr = 2e-3
        self.weight_decay = 1e-5
        self.epochs = 64
        self.batch_size = 128
        self.patients = 8

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
