import  numpy as np


def gen_batch(data, batch_size, num_steps):
    data = np.asarray(data)
    n_batch = (data.shape[0] - num_steps + 2) // (batch_size) + 1
    print('number of batches = ', n_batch)
    for i in range(n_batch):
        data_batch = []
        for j in range(batch_size * i, batch_size * (i + 1)):
            begin = j
            end = j + num_steps
            if end > data.shape[0]:
                break
            # end = end if end < data.shape[0] else data.shape[0]
            data_batch.append(data[begin:end])
        yield data_batch

if __name__ == '__main__':
    batch_size = 32
    num_steps = 11
    data = range(1, 159)
    for idx, batch in enumerate(gen_batch(data, batch_size, num_steps)):
        print('batch {} = {}'.format(idx, batch))
        print('batch {}.size={}'.format(idx, len(batch)))
