# JavaCNN
一个卷积神经网络的java实现. 仿Matlab toolbox(https://github.com/rasmusbergpalm/DeepLearnToolbox )实现的，同时进行了部分改进，使得卷积核和采样块可以为矩形而不仅仅是正方形。更多细节，请查看http://www.cnblogs.com/fengfenggirl/p/cnn_implement.html
## 创建一个卷积神经网络

	LayerBuilder builder = new LayerBuilder();
	builder.addLayer(Layer.buildInputLayer(new Size(28, 28)));
	builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)));
	builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
	builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)));
	builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
	builder.addLayer(Layer.buildOutputLayer(10));
	CNN cnn = new CNN(builder, 50);
	
## 在 MNIST 数据集上测试
	
	String fileName = "data/train.format";
	Dataset dataset = Dataset.load(fileName, ",", 784);
	cnn.train(dataset, 100);
	Dataset testset = Dataset.load("data/test.format", ",", -1);
	cnn.predict(testset, "data/test.predict");

迭代100次，四核CPU大约需要运行一个小时后，正确率97.8%

##Lisence
	MIT