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
		cnn.train(dataset, 2);
	}
	public static void main(String[] args) {
		runCnn();
	}

}
