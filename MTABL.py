import Call as Cal
import ModelLib as MLib
import numpy as np
from keras import constraints
from keras.utils import to_categorical
import pickle as pickle
from sklearn import metrics


def make_history(train_history, test_history, horizon, maxnorm, dropout, r, batch_size, LR, Schedule):

    data = {'train': train_history, 'test': test_history,
            'horizon': horizon,
            'maxnorm': maxnorm, 'dropout': dropout, 'r': r,
            'batch_size': batch_size, 'LR': LR, 'Schedule': Schedule}

    return data


horizon = 0
Maxnorm = (7.0, 7.0)
dropout = 0.1
r = 1.0
batch_size = 256

# Some random data
aprice_train = np.random.rand(100, 10, 10)
avol_train = np.random.rand(100, 10, 10)
bprice_train = np.random.rand(100, 10, 10)
bvol_train = np.random.rand(100, 10, 10)
aprice_test = np.random.rand(100, 10, 10)
avol_test = np.random.rand(100, 10, 10)
bprice_test = np.random.rand(100, 10, 10)
bvol_test = np.random.rand(100, 10, 10)

# ==============================================================================
y_test_tmp = np.random.randint(0, 3, (100,))
y_train_tmp = np.random.randint(0, 3, (100,))
y_train = to_categorical(y_train_tmp, 3)
y_test = to_categorical(y_test_tmp, 3)

result_dir = '../results'

# ==============================================================================

x_AP0 = aprice_train[np.where(y_train_tmp == 0)[0]]
x_AP1 = aprice_train[np.where(y_train_tmp == 1)[0]]
x_AP2 = aprice_train[np.where(y_train_tmp == 2)[0]]

x_AV0 = avol_train[np.where(y_train_tmp == 0)[0]]
x_AV1 = avol_train[np.where(y_train_tmp == 1)[0]]
x_AV2 = avol_train[np.where(y_train_tmp == 2)[0]]

x_BV0 = bvol_train[np.where(y_train_tmp == 0)[0]]
x_BV1 = bvol_train[np.where(y_train_tmp == 1)[0]]
x_BV2 = bvol_train[np.where(y_train_tmp == 2)[0]]

x_BP0 = bprice_train[np.where(y_train_tmp == 0)[0]]
x_BP1 = bprice_train[np.where(y_train_tmp == 1)[0]]
x_BP2 = bprice_train[np.where(y_train_tmp == 2)[0]]

for maxnorm in Maxnorm:

    model_name = 'name'
    kernel_constraint = constraints.max_norm(maxnorm)
    kernel_regularizer = None
    model = MLib.get_subattention(kernel_constraint=kernel_constraint, kernel_regularizer=kernel_regularizer)
    model.summary()

    # Training parameters

    class_weight = {0: 1e6 / (np.size(np.where(y_train_tmp == 0))**(1 / r)),
                    1: 1e6 / (np.size(np.where(y_train_tmp == 1))**(1 / r)),
                    2: 1e6 / (np.size(np.where(y_train_tmp == 2))**(1 / r))}

    batch_size = 256
    LR = (0.01, ) * 15 + (0.005, ) * 20 + (0.001, ) * 50 + (0.0005, ) * 20 + (0.0001, ) * 50 + (0.00001, ) * 50
    Schedule = (15, 35, 85, 105, 155, 205)

    train_history = np.zeros((len(LR), 4))
    test_history = np.zeros((len(LR), 4))

    activations = []
    derivatives = []

    for index, lr in enumerate(LR):
        if index in Schedule:

            # Get mask output of three different Y:s for all inputs
            histories = Cal.MaskHistories(data=[x_AP0, x_AP1, x_AP2,
                                                x_AV0, x_AV1, x_AV2,
                                                x_BP0, x_BP1, x_BP2,
                                                x_BV0, x_BV1, x_BV2])

            model = MLib.get_subattention(kernel_constraint=kernel_constraint,
                                          kernel_regularizer=kernel_regularizer)

            model.load_weights(result_dir + model_name + '.h5')

            model.optimizer.lr = lr

            model.fit([aprice_train, avol_train, bprice_train, bvol_train], y_train,
                      batch_size=batch_size,
                      epochs=1, class_weight=class_weight,
                      verbose=1,
                      callbacks=[histories],
                      validation_data=([aprice_test, avol_test, bprice_test, bvol_test], y_test))

            y_pred = np.argmax(model.predict([aprice_train, avol_train,
                                              bprice_train, bvol_train], batch_size=batch_size), axis=1)

            train_history[index, :] = [metrics.accuracy_score(y_train_tmp, y_pred),
                                       metrics.precision_score(y_train_tmp, y_pred, average='macro'),
                                       metrics.recall_score(y_train_tmp, y_pred, average='macro'),
                                       metrics.f1_score(y_train_tmp, y_pred, average='macro')]

            y_pred = np.argmax(model.predict([aprice_test, avol_test,
                                              bprice_test, bvol_test], batch_size=batch_size), axis=1)

            test_history[index, :] = [metrics.accuracy_score(y_test_tmp, y_pred),
                                      metrics.precision_score(y_test_tmp, y_pred, average='macro'),
                                      metrics.recall_score(y_test_tmp, y_pred, average='macro'),
                                      metrics.f1_score(y_test_tmp, y_pred, average='macro')]

            activations += histories.all_outputs
            derivatives += histories.derivatives

            # For information during running
            print('R=%.2f, Epoch %d' % tuple([r, index]))
            print('Train Result %.4f\t %.4f\t %.4f\t %.4f' % tuple(train_history[index, :]))
            print('Test Result %.4f\t %.4f\t %.4f\t %.4f' % tuple(test_history[index, :]))

        else:

            histories = Cal.MaskHistories(data=[x_AP0, x_AP1, x_AP2,
                                                x_AV0, x_AV1, x_AV2,
                                                x_BP0, x_BP1, x_BP2,
                                                x_BV0, x_BV1, x_BV2])
            model.optimizer.lr = lr

            model.fit([aprice_train, avol_train, bprice_train, bvol_train], y_train,
                      batch_size=batch_size,
                      epochs=1,
                      class_weight=class_weight,
                      verbose=1,
                      callbacks=[histories],
                      validation_data=([aprice_test, avol_test, bprice_test, bvol_test], y_test))

            y_pred = np.argmax(model.predict([aprice_train, avol_train,
                                              bprice_train, bvol_train], batch_size=batch_size), axis=1)

            train_history[index, :] = [metrics.accuracy_score(y_train_tmp, y_pred),
                                       metrics.precision_score(y_train_tmp, y_pred, average='macro'),
                                       metrics.recall_score(y_train_tmp, y_pred, average='macro'),
                                       metrics.f1_score(y_train_tmp, y_pred, average='macro')]

            y_pred = np.argmax(model.predict([aprice_test, avol_test,
                                              bprice_test, bvol_test], batch_size=batch_size), axis=1)

            test_history[index, :] = [metrics.accuracy_score(y_test_tmp, y_pred),
                                      metrics.precision_score(y_test_tmp, y_pred, average='macro'),
                                      metrics.recall_score(y_test_tmp, y_pred, average='macro'),
                                      metrics.f1_score(y_test_tmp, y_pred, average='macro')]

            print('R=%.2f, Epoch %d' % tuple([r, index]))
            print('Train Result %.4f\t %.4f\t %.4f\t %.4f' % tuple(train_history[index, :]))
            print('Test Result %.4f\t %.4f\t %.4f\t %.4f' % tuple(test_history[index, :]))

            model.save_weights(result_dir + model_name +'.h5')
            activations += histories.all_outputs
            derivatives += histories.derivatives

    """save training history"""

    history = make_history(train_history, test_history, horizon,
                           maxnorm, dropout, r, batch_size, LR, Schedule)

    file_handle = open('../../performance.txt', 'a')

    f1 = np.max(test_history[:, 3])
    idx = np.argmax(test_history[:, 3])

    text_result = str(horizon) + ', ' + str(r) + ', ' + str(dropout) + ', ' + str(maxnorm) + ', ' \
                  + '{0:.4f}'.format(f1) + ',' \
                  + '{0:.4f}'.format(test_history[idx, 0]) + ', ' \
                  + '{0:.4f}'.format(test_history[idx, 1]) + ', ' \
                  + '{0:.4f}'.format(test_history[idx, 2]) + ', ' \
                  + '\n'

    file_handle.write(text_result)
    file_handle.close()

    history['activations'] = activations
    history['derivatives'] = derivatives

    with open(result_dir + model_name + '.pickle', 'wb') as h:
        pickle.dump(history, h, protocol=pickle.HIGHEST_PROTOCOL)
