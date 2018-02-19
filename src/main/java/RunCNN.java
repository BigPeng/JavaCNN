import java.io.IOException;

import javacnn.cnn.CNN;
import javacnn.cnn.CNNLoader;
import javacnn.cnn.Layer;
import javacnn.dataset.Dataset;
import javacnn.dataset.DatasetLoader;
import javacnn.util.ConcurenceRunner;

public class RunCNN {

	public static void main(String[] args) throws IOException, ClassNotFoundException {

		final ConcurenceRunner concurenceRunner = new ConcurenceRunner();

		try {

			final CNN.LayerBuilder builder = new CNN.LayerBuilder();

			builder.addLayer(Layer.buildInputLayer(new Layer.Size(28, 28)));
			builder.addLayer(Layer.buildConvLayer(6, new Layer.Size(5, 5)));
			builder.addLayer(Layer.buildSampLayer(new Layer.Size(2, 2)));
			builder.addLayer(Layer.buildConvLayer(12, new Layer.Size(5, 5)));
			builder.addLayer(Layer.buildSampLayer(new Layer.Size(2, 2)));
			builder.addLayer(Layer.buildOutputLayer(10));

			final CNN cnn = new CNN(builder, 50, concurenceRunner);

			final String fileName = "dataset/train.format";
			final Dataset dataset = DatasetLoader.load(fileName, ",", 784);
			cnn.train(dataset, 5);

			CNNLoader.saveModel("model.cnn", cnn);
			dataset.clear();

			/*
			final CNN cnn = CNNLoader.loadModel("model.cnn");
			cnn.setRunner(concurenceRunner);
			*/

			final Dataset testset = DatasetLoader.load("dataset/test.format", ",", -1);
			cnn.predict(testset, "dataset/test.predict");

		} finally {
			concurenceRunner.shutdown();
		}
	}

}
