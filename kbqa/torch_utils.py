import torch
import time
import numpy as np
import six


class TrainHandler:
    def __init__(self,
                 train_loader, valid_loader, model, criterion, optimizer,
                 model_path, batch_size=32, epochs=5, scheduler=None, gpu_num=0):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_path = model_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.scheduler = scheduler
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_num}')
            print(f'training device is gpu:{gpu_num}')
        else:
            self.device = torch.device('cpu')
            print('traning device is cpu')
        self.model = model.to(self.device)

    def _train_func(self):
        train_loss = 0
        train_correct = 0
        for i, (x, y) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            x, y = x.to(self.device).long(), y.to(self.device).long()
            output = self.model(x)
            loss = self.criterion(output, y)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            train_correct += (output.argmax(1) == y).sum().item()

        if self.scheduler is not None:
            self.scheduler.step()
        # 这两个长度有区别吗？
        return train_loss / len(self.train_loader), train_correct / len(self.train_loader.dataset)

    def _test_func(self):
        valid_loss = 0
        valid_correct = 0
        for x, y in self.valid_loader:
            x, y = x.to(self.device).long(), y.to(self.device)
            with torch.no_grad():
                output = self.model(x)
                loss = self.criterion(output, y)
                valid_loss += loss.item()
                valid_correct += (output.argmax(1) == y).sum().item()

        return valid_loss / len(self.valid_loader), valid_correct / len(self.valid_loader.dataset)

    def train(self):
        min_valid_loss = float('inf')

        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss, train_acc = self._train_func()
            valid_loss, valid_acc = self._test_func()

            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss
                torch.save(self.model, self.model_path)
                print(f'\tSave model done, valid loss:{valid_loss}')

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60
            print('Epoch:%d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
            print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


def torch_text_process():
    from torchtext import data

    def tokenizer(text):
        import jieba
        return list(jieba.cut(text))

    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=20)
    LABEL = data.Field(sequential=False, use_vocab=False)
    all_dataset = data.TabularDataset.split(path='',
                                            train='LCOMC.csv',
                                            fields=[('sentence1', TEXT),('sentence2', TEXT), ('label',LABEL)])[0]
    TEXT.build_vocab(all_dataset)
    train, valid = all_dataset.split(0.1)
    # batchh_sizes?
    (train_iter, valid_iter) = data.BucketIterator.splits(datasets=(train, valid),
                                                          batch_sizes=(64,128),
                                                          sort_key=lambda x: len(x.sentence1))
    return train_iter, valid_iter


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='pre', value=0.):
    """
    pads sequences to the same length.

    this function transforms a list of 'num_smaples' sequences
    into a 2D NUmpy array of shape `(num_samples, num_timesteps)`.
    :param sequences: List of lists, whehre each element is a sequence.
    :param maxlen: Int, maximun length of all sequences
    :param dtype: Type of the output sequences.
    :param padding: String, 'pre' or 'post':
                pad either before or after each sequence.
    :param truncating: String, 'pre' or 'post':
                remove values from sequences larger than `maxlen`,either at the begging
                or at the end of the sequences.
    :param value:Float or String, padding values
    :return:
            x: numpy array with shape `(len(sequences),maxlen)`
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables.'
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))
    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-maxlen]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

    trunc = np.array(trunc, dtype=dtype)
    if trunc.shape[1:] != sample_shape:
        raise ValueError('Shape of sample %s of sequence at position %s '
                         'is different from expected shape %s' %
                         (trunc.shape[1:], idx, sample_shape))

    if padding == 'post':
        x[idx, :len(trunc)] = trunc
    elif padding == 'pre':
        x[idx, -len(trunc)] = trunc
    else:
        raise ValueError('Padding type "%s" not understood' % padding)
    return x


if __name__ == '__main__':
    torch_text_process()



