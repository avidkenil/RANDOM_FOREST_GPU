curl http://zphang.net/data_fol/mnist/train_x.npy -o mnist_train_x.npy
curl http://zphang.net/data_fol/mnist/train_y.npy -o mnist_train_y.npy
curl http://zphang.net/data_fol/mnist/test_x.npy -o mnist_test_x.npy
curl http://zphang.net/data_fol/mnist/test_y.npy -o mnist_test_y.npy

python save_iris.py