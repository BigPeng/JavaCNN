package edu.hitsz.c102c.cnn;

import edu.hitsz.c102c.cnn.CNN.LayerBuilder;
import edu.hitsz.c102c.cnn.Layer.Size;
import edu.hitsz.c102c.data.Dataset;

public class RunCNN {

	public static void runCnn() {
		LayerBuilder builder = new LayerBuilder();
		builder.addLayer(Layer.buildInputLayer(new Size(28, 28)));
		builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildOutputLayer(10));
		CNN cnn = new CNN(builder, 20);
		String fileName = "data/train.format";
		Dataset dataset = Dataset.load(fileName, ",", 784);
		cnn.train(dataset, 3);
		// dataset.clear();
		// dataset = null;
		// Dataset testset =
		// Dataset.load("data/test.format", ",", -1);
		// cnn.predict(testset, "data/test.predict");
	}

	public static void tinyTest() {
		LayerBuilder builder = new LayerBuilder();
		builder.addLayer(Layer.buildInputLayer(new Size(4, 4)));
		Layer c = Layer.buildConvLayer(2, new Size(3, 3));
		builder.addLayer(c);
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		Layer out = Layer.buildOutputLayer(1);
		builder.addLayer(out);
		CNN cnn = new CNN(builder, 1);
		double[][] k1 = { { 1, 1, 1 }, { 0, 1, 0 }, { 1, 0, 0 } };
		double[][] k2 = { { 1, 0, 1 }, { 0, 1, 0 }, { 1, 0, 1 } };
		c.setKernel(0, 0, k1);
		c.setKernel(0, 1, k2);
		double[][] k11 = { { 0.5 } };
		double[][] k22 = { { -0.5 } };
		out.setKernel(0, 0, k11);
		out.setKernel(1, 0, k22);
		String fileName = "data/train.tiny";
		Dataset dataset = Dataset.load(fileName, ",", 16);
		cnn.train(dataset, 1000);
	}

	public static void main(String[] args) {
		//runCnn();
		tinyTest();
	}

}
