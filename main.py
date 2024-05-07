import numpy as np
import argparse
import preprocess
import utils
import train
from torchsummary import summary

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-m', '--mode', default="test", type=str,
                        help="mode: train \ test")

    parser.add_argument('-n', '--model_name', default='model', type=str,
                        help="model name")


    args = parser.parse_args()

    mode = args.mode
    name = args.model_name

    fs = 128
    dir_name = "learning-set"
    segment_time = 4
    seg_data, s_size = preprocess.load_preprocess(dir_name, fs, segment_time)

    utils.plot_example(seg_data)

    print(f"size of train set: {seg_data.shape[0]}")

    # train and test
    train_df, test_df, valid_df = utils.divide_sets(seg_data)
    X_train, X_test, X_valid, y_train, y_test, y_valid = utils.extract_samples(train_df, test_df, valid_df)

    model = train.CNNModel(s_size).float()

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters:", total_params)

    batch_size = 512
    epochs = 600

    if mode == 'train':
        train_loss, val_loss_arr, lr = train.train_model(model=model, x_train=X_train, y_train=y_train,
                                                         x_valid=X_valid, y_valid=y_valid, batch_size=batch_size,
                                                         epochs=epochs, name=name)
        #     plot results
        summary(model, input_size=X_train.size())
        utils.plot_loss(train_loss, val_loss_arr, lr)

    elif mode == 'test':
        nn_predict = train.test_model(model=model, x_test=X_test, y_test=y_test, name=name)
        # nn_predict = train.test_model(model=model, x_test=X_train, y_test=y_train, name=name)
        # utils.plot_results(nn_predict, y_train)
        utils.plot_results(nn_predict, y_test)
        sustain = 2

        # True_label = train_df['True_label'].values
        True_label = test_df['True_label'].values
        predict = nn_predict.copy()
        predict[nn_predict <= 60] = 0
        predict[(nn_predict > 60) & (nn_predict <= 120)] = 1
        predict[nn_predict > 120] = sustain

        bin_pred = predict.copy()
        bin_pred[bin_pred < sustain] = 0
        bin_label = True_label.copy()
        bin_label[bin_label < sustain] = 0

        utils.calc_confusion(bin_label, bin_pred)
        # print(nn_predict)
        # print(f"prediction: {predict}")
        # print(f"True label {True_label}")
        # print(f"bin classification {np.sum(bin_label == bin_pred) / bin_pred.shape[1]}")
        # print(f"full classification {np.sum(predict == True_label) / predict.shape[1]}")
