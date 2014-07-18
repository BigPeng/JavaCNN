package edu.hitsz.c102c.cnn;

import edu.hitsz.c102c.cnn.CNN.LayerBuilder;
import edu.hitsz.c102c.cnn.Layer.Size;
import edu.hitsz.c102c.data.Dataset;
import edu.hitsz.c102c.util.ConcurenceRunner;
import edu.hitsz.c102c.util.TimedTest;
import edu.hitsz.c102c.util.TimedTest.TestTask;

public class RunCNN {

	public static void runCnn() {
		LayerBuilder builder = new LayerBuilder();
		builder.addLayer(Layer.buildInputLayer(new Size(28, 28)));
		builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildOutputLayer(2));
		CNN cnn = new CNN(builder, 2);
		String fileName = "data/train.format";
		Dataset dataset = Dataset.load(fileName, ",", 784);
		cnn.train(dataset, 3);//
		// String modelName = "model/model.cnn";
		// cnn.saveModel(modelName);
		// String fileName = "data/train.format";
		// String modelName = "model/model.cnn";
		// CNN cnn = CNN.loadModel(modelName);
		// Dataset dataset = Dataset.load(fileName,
		// ",", 784);
		// cnn.train(dataset, 400);
		// cnn.saveModel(modelName);
		// dataset.clear();
		// dataset = null;
		// Dataset testset =
		// Dataset.load("data/test.format", ",", -1);
		// cnn.predict(testset, "data/test.predict");
	}

	public static void testArData() {
		// LayerBuilder builder = new LayerBuilder();
		// builder.addLayer(Layer.buildInputLayer(new
		// Size(3, 256)));
		// builder.addLayer(Layer.buildConvLayer(12,
		// new Size(3, 17)));
		// builder.addLayer(Layer.buildSampLayer(new
		// Size(1, 2)));
		// builder.addLayer(Layer.buildConvLayer(12,
		// new Size(1, 21)));
		// builder.addLayer(Layer.buildSampLayer(new
		// Size(1, 2)));
		// builder.addLayer(Layer.buildOutputLayer(5));
		// CNN cnn = new CNN(builder, 20);
		String fileName = "data/ar_data.shuffle";
		Dataset dataset = Dataset.load(fileName, ",", 768);
		// cnn.train(dataset, 40);
		String modelName = "model/ar.cnn";
		CNN cnn = CNN.loadModel(modelName);
		cnn.train(dataset, 200);
		cnn.saveModel(modelName);
		dataset.clear();
		dataset = null;
		Dataset testset = Dataset.load("data/ar_data.shuffle.test", ",", 768);
		cnn.test(testset);

	}

	public static void tinyTest() {
		LayerBuilder builder = new LayerBuilder();
		builder.addLayer(Layer.buildInputLayer(new Size(6, 6)));
		Layer c = Layer.buildConvLayer(2, new Size(3, 3));
		builder.addLayer(c);
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		Layer out = Layer.buildOutputLayer(1);
		builder.addLayer(out);
		CNN cnn = new CNN(builder, 2);
		double[][] k1 = { { 1, 0, 1 }, { 1, 1, 0 }, { 1, 0, 0 } };
		double[][] k2 = { { 1, 1, 1 }, { 1, 1, 0 }, { 1, 0, 0 } };
		c.setKernel(0, 0, k1);
		c.setKernel(0, 1, k2);
		double[][] k11 = { { 0.5, -0.5 }, { -0.5, 0.5 } };
		double[][] k22 = { { -0.5, 0.5 }, { 0.5, -0.5 } };
		out.setKernel(0, 0, k11);
		out.setKernel(1, 0, k22);
		String fileName = "data/train.tiny";
		Dataset dataset = Dataset.load(fileName, ",", 16);
		cnn.train(dataset, 20);
		// String modelName = "model/model.cnn";
		// cnn.saveModel(modelName);

//		CNN cnn2 = CNN.loadModel(modelName);
//		cnn2.test(dataset);

	}

	public static void main(String[] args) {
		//runCnn();
		tinyTest();
		// new TimedTest(new TestTask() {
		//
		// @Override
		// public void process() {
		// runCnn();
		// //tinyTest();
		// //testArData();
	
		// }
		// }, 1).test();
		ConcurenceRunner.stop();

	}

}
