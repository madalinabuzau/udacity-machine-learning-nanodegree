import numpy as np
import matplotlib.pyplot as plt

def generate_sequences(X_train, y_train, img_height, img_width, max_digits=5,
        n_synthetic = 150000, random_seed=2017):

    # Set random seed for reproducible results
    np.random.seed(random_seed)

    # Generate 150000 random numbers of sequence lengths (1, max_digits)
    len_synt = np.random.choice(np.arange(1,max_digits+1), size=150000)

    # See the distribution of sequences
    dist_sequences = np.unique(len_synt, return_counts=True)

    print('Distribution of sequence length: ',dist_sequences)

    # Initialize empty tensor for our synthetic input dataset
    X_synt = np.zeros((n_synthetic, img_height, img_width*max_digits))

    # Initialize array for our synthetic target. We fill it with '10' to represent 'blank' characters
    y_synt = np.zeros((n_synthetic, max_digits))
    y_synt.fill(10)

    # Concatenate images according to their sequence length
    for i in range(max_digits):
        num_samples = np.sum(len_synt==(i+1))
        for k in range(i+1):
            np.random.seed(2017 + i + 10*k)
            rand_idx = np.random.choice(np.arange(X_train.shape[0]), size=num_samples)
            X_synt[len_synt==(i+1),:,img_width*k:(img_width*k+img_width)] = X_train[rand_idx,:,:]
            y_synt[len_synt==(i+1), k] = y_train[rand_idx]

    return X_synt, y_synt, len_synt

def save_ex_sequences(X_synt, y_synt, len_synt, max_digits=5, num_display=2, figsize=(16,4)):
    for i in range(max_digits):
        fig = plt.figure(figsize=figsize)
        for k in range(num_display):
            plt.subplot(1, num_display, k + 1)
            plt.axis('off')
            plt.imshow(X_synt[len_synt==i+1,:,:][k], cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('Target: '+str(y_synt[len_synt==i+1][k]))
        plt.savefig('Example_concatenated_digits_%i.png' %i)
        plt.close()


def normalize_split_data(X_synt, y_synt, img_height, img_width,
                         test_size=0.2, max_digits=5, n_synthetic=150000,
                         random_seed=2017):

    np.random.seed(random_seed)

    # Normalize data by max. I have tried standardizing (with 0 mean and equal variance)
    # the input first but it worked very poorly
    X_synt = X_synt/X_synt.max().max()

    # Reshape data from 3D to 4D
    X_synt = np.reshape(X_synt, (n_synthetic, img_height, img_width*max_digits, 1))

    # Create a test set representing 20% of the entire dataset
    test_size = 0.2
    rand_idx = np.random.choice(np.arange(0, n_synthetic), size=int(test_size*n_synthetic), replace=False)

    # Get the test set
    X_test = X_synt[rand_idx,:]
    y_test = y_synt[rand_idx]

    # Get the training set
    X_train = np.delete(X_synt, rand_idx, 0)
    y_train = np.delete(y_synt, rand_idx, 0)

    # One-hot encode the target labels in the training and testing set
    y_train_ohe = {}
    y_test_ohe = {}
    for i in range(max_digits):
        y_train_ohe[i] = np.zeros((y_train.shape[0], 11))
        y_test_ohe[i] = np.zeros((y_test.shape[0], 11))
        y_train_ohe[i][np.arange(0,y_train.shape[0]), y_train[:,i].astype(int)] = 1
        y_test_ohe[i][np.arange(0,y_test.shape[0]), y_test[:,i].astype(int)] = 1

    return X_train, X_test, y_train_ohe, y_test_ohe
