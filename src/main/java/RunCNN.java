import java.io.IOException;

import javacnn.cnn.CNN;
import javacnn.cnn.CNN.LayerBuilder;
import javacnn.cnn.CNNLoader;
import javacnn.cnn.Layer;
import javacnn.cnn.Layer.Size;
import javacnn.dataset.Dataset;
import javacnn.dataset.DatasetLoader;
import javacnn.util.ConcurenceRunner;
import javacnn.util.TimedTest;

public class RunCNN {

	public static void runCnn() throws IOException {

		final LayerBuilder builder = new LayerBuilder();

		builder.addLayer(Layer.buildInputLayer(new Size(28, 28)));
		builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildOutputLayer(10));

		final CNN cnn = new CNN(builder, 50);

		final String fileName = "dataset/train.format";
		final Dataset dataset = DatasetLoader.load(fileName, ",", 784);
		cnn.train(dataset, 100);

		CNNLoader.saveModel("model.cnn", cnn);
		dataset.clear();

		// CNN cnn = CNNLoader.loadModel(modelName);
		final Dataset testset = DatasetLoader.load("dataset/test.format", ",", -1);
		cnn.predict(testset, "dataset/test.predict");
	}

	public static void main(String[] args) {
		new TimedTest(() -> {
			try {
				runCnn();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}, 1).test();
		ConcurenceRunner.stop();
	}

}
