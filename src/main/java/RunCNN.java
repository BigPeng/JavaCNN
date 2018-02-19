import java.io.IOException;

import javacnn.cnn.CNN;
import javacnn.cnn.CNN.LayerBuilder;
import javacnn.cnn.CNNLoader;
import javacnn.cnn.Layer;
import javacnn.cnn.Layer.Size;
import javacnn.dataset.Dataset;
import javacnn.dataset.DatasetLoader;
import javacnn.util.ConcurenceRunner;

public class RunCNN {

	public static void main(String[] args) throws IOException {

		final ConcurenceRunner concurenceRunner = new ConcurenceRunner();

		final LayerBuilder builder = new LayerBuilder();

		builder.addLayer(Layer.buildInputLayer(new Size(28, 28)));
		builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildOutputLayer(10));

		final CNN cnn = new CNN(builder, 50, concurenceRunner);

		final String fileName = "dataset/train.format";
		final Dataset dataset = DatasetLoader.load(fileName, ",", 784);
		cnn.train(dataset, 5);

		CNNLoader.saveModel("model.cnn", cnn);
		dataset.clear();

		// CNN cnn = CNNLoader.loadModel(modelName);
		final Dataset testset = DatasetLoader.load("dataset/test.format", ",", -1);
		cnn.predict(testset, "dataset/test.predict");

		concurenceRunner.stop();
	}

}
