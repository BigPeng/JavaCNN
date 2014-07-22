# JavaCNN
A Java implement of Convolutional Neural Network. Learn from DeepLearnToolbox(https://github.com/rasmusbergpalm/DeepLearnToolbox) more detail. see here(http://www.cnblogs.com/fengfenggirl/p/cnn_implement.html)
## Build a CNN

	LayerBuilder builder = new LayerBuilder();
	builder.addLayer(Layer.buildInputLayer(new Size(28, 28)));
	builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)));
	builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
	builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)));
	builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
	builder.addLayer(Layer.buildOutputLayer(10));
	CNN cnn = new CNN(builder, 50);
	
## Run on MNIST dataset
	
	String fileName = "data/train.format";
	Dataset dataset = Dataset.load(fileName, ",", 784);
	cnn.train(dataset, 100);
	Dataset testset = Dataset.load("data/test.format", ",", -1);
	cnn.predict(testset, "data/test.predict");

It takes a about an hour to complete 100 iteration and get a precison of 97.8%

##Lisence
	MIT