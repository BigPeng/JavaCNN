package edu.hitsz.c102c.cnn;

import java.io.IOException;

import edu.hitsz.c102c.cnn.CNN.LayerBuilder;
import edu.hitsz.c102c.cnn.Layer.Size;
import edu.hitsz.c102c.dataset.Dataset;
import edu.hitsz.c102c.dataset.DatasetLoader;
import edu.hitsz.c102c.util.ConcurenceRunner;
import edu.hitsz.c102c.util.TimedTest;
import edu.hitsz.c102c.util.TimedTest.TestTask;

public class RunCNN {

	public static void runCnn() throws IOException {
		//����һ�����������
		LayerBuilder builder = new LayerBuilder();
		builder.addLayer(Layer.buildInputLayer(new Size(28, 28)));
		builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildOutputLayer(10));
		CNN cnn = new CNN(builder, 50);

		//�������ݼ�
		String fileName = "dataset/train.format";
		Dataset dataset = DatasetLoader.load(fileName, ",", 784);
		cnn.train(dataset, 3);//
		String modelName = "model/model.cnn";
		cnn.saveModel(modelName);
		dataset.clear();
		dataset = null;

		//Ԥ��
		// CNN cnn = CNN.loadModel(modelName);	
		Dataset testset = DatasetLoader.load("dataset/test.format", ",", -1);
		cnn.predict(testset, "dataset/test.predict");
	}

	public static void main(String[] args) {

		new TimedTest(new TestTask() {

			@Override
			public void process() {
				try {
					runCnn();
				} catch (IOException e) {
					throw new RuntimeException(e);
				}
			}
		}, 1).test();
		ConcurenceRunner.stop();

	}

}
