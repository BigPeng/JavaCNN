package javacnn.cnn;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import javacnn.dataset.Dataset;
import javacnn.util.DotProgressIndicator;
import javacnn.util.Log;
import javacnn.util.ProgressIndicator;
import javacnn.util.Runner;
import javacnn.util.Util;

public class CNN implements Serializable {

	private static final long serialVersionUID = 3L;

	private static final double LAMBDA = 0;

	private double ALPHA = 0.85;

	private final List<Layer> layers;
	private final int layerNum;
	private final int batchSize;

	private final Util.Operator divide_batchSize;
	private final Util.Operator multiply_alpha;
	private final Util.Operator multiply_lambda;

	private transient Runner runner;

	private transient ProgressIndicator progressIndicator = new DotProgressIndicator();

	public CNN(LayerBuilder layerBuilder, final int batchSize, final Runner runner) {

		this.layers = layerBuilder.layerList;
		this.layerNum = layers.size();
		this.batchSize = batchSize;

		this.runner = runner;

		setup(batchSize);

		// ---

		final double _1_batchSize = 1. / batchSize;

		divide_batchSize = new Util.Operator() {
			private static final long serialVersionUID = 7424011281732651055L;

			@Override
			public double process(double value) {
				return value * _1_batchSize;
			}

		};

		multiply_alpha = new Util.Operator() {
			private static final long serialVersionUID = 5761368499808006552L;

			@Override
			public double process(double value) {
				return value * ALPHA;
			}

		};

		multiply_lambda = new Util.Operator() {
			private static final long serialVersionUID = 4499087728362870577L;

			@Override
			public double process(double value) {
				return value * (1 - LAMBDA * ALPHA);
			}
		};
	}

	// === simple getters and setters ===

	public void setRunner(final Runner runner) {
		this.runner = runner;
	}

	private Runner getRunner() {
		if (runner == null) throw new NullPointerException("'runner' is null.  Set runner before start training or test!");
		return runner;
	}

	public ProgressIndicator getProgressIndicator() {
		return progressIndicator;
	}

	public void setProgressIndicator(final ProgressIndicator progressIndicator) {
		this.progressIndicator = progressIndicator;
	}

	// === business logic ===

	public void train(final Dataset trainset, final int iterationCount) {
		for (int iteration = 0; iteration < iterationCount; iteration++) {

			progressIndicator.start();

			// separate trainset in batches of batchsize ... and round up the result
			final int epochsNum = (trainset.size() + batchSize - 1) / batchSize;

			Log.info("");
			Log.info(iteration + "th iter epochsNum:" + epochsNum);

			int right = 0;
			int count = 0;

			for (int epoch = 0; epoch < epochsNum; epoch++) {

				int[] randPerm = Util.randomPerm(trainset.size(), batchSize);

				Layer.prepareForNewBatch();

				for (int index : randPerm) {
					final boolean isRight = train(trainset.getRecord(index));
					if (isRight) right++;
					count++;
					Layer.prepareForNewRecord();
				}

				// After finishing a batch update weight
				updateParas();

				progressIndicator.progress();
			}

			progressIndicator.finished();

			final double precision = ((double) right) / count;

			if (iteration % 10 == 1 && precision > 0.96) {
				ALPHA = 0.001 + ALPHA * 0.9; // Adjust the quasi-learning rate dynamically
				Log.info("Set alpha = " + ALPHA);
			}

			Log.info("precision " + right + "/" + count + "=" + precision);
		}
	}

	public double test(final Dataset dataset) {
		Layer.prepareForNewBatch();

		final Iterator<Dataset.Record> iterator = dataset.iterator();

		int right = 0;
		while (iterator.hasNext()) {
			final Dataset.Record record = iterator.next();

			final double[] out = propagate(record);

			if (record.getLabel().intValue() == Util.getMaxIndex(out)) {
				right++;
			}
		}

		double p = 1.0 * right / dataset.size();

		Log.info("precision", p + "");

		return p;
	}

	private double[] getOutput() {
		final Layer outputLayer = layers.get(layerNum - 1);

		final int mapNum = outputLayer.getOutMapNum();
		final double[] out = new double[mapNum];
		for (int m = 0; m < mapNum; m++) {
			final double[][] outmap = outputLayer.getMap(m);
			out[m] = outmap[0][0];
		}
		return out;
	}

	// TODO: Move this method to other/new class (reduce CNN-class to the minimal CNN-logic)
	public void predict(Dataset testset, String fileName) {
		Log.info("begin predict");
		try {
			// final int max = layers.get(layerNum - 1).getClassNum();
			final PrintWriter writer = new PrintWriter(new File(fileName));

			Layer.prepareForNewBatch();

			final Iterator<Dataset.Record> iter = testset.iterator();
			while (iter.hasNext()) {
				final Dataset.Record record = iter.next();
				final double[] out = propagate(record);
				// int label =
				// Util.binaryArray2int(out);
				final int label = Util.getMaxIndex(out);
				// if (label >= max)
				// label = label - (1 << (out.length -
				// 1));
				writer.write(label + "\n");
			}
			writer.flush();
			writer.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		Log.info("end predict");
	}

	private boolean train(Dataset.Record record) {
		forward(record);

		return backPropagation(record);
	}

	private boolean backPropagation(Dataset.Record record) {
		boolean result = setOutLayerErrors(record);
		setHiddenLayerErrors();
		return result;
	}

	private void updateParas() {
		for (int l = 1; l < layerNum; l++) {
			Layer layer = layers.get(l);
			Layer lastLayer = layers.get(l - 1);
			switch (layer.getType()) {
				case conv:
				case output:
					updateKernels(layer, lastLayer);
					updateBias(layer);
					break;
				default:
					break;
			}
		}
	}

	private void updateBias(final Layer layer) {
		final double[][][][] errors = layer.getErrors();
		int mapNum = layer.getOutMapNum();

		final Process processor =
				new Process() {
					@Override
					public void process(int start, int end) {
						for (int j = start; j < end; j++) {
							final double[][] error = Util.sum(errors, j);
							// update offset
							final double deltaBias = Util.sum(error) / batchSize;
							final double bias = layer.getBias(j) + ALPHA * deltaBias;
							layer.setBias(j, bias);
						}
					}
				};

		getRunner().startProcess(mapNum, processor);
	}

	private void updateKernels(final Layer layer, final Layer lastLayer) {
		final int mapNum = layer.getOutMapNum();
		final int lastMapNum = lastLayer.getOutMapNum();

		final Process process = new Process() {
			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					for (int i = 0; i < lastMapNum; i++) {
						double[][] deltaKernel = null;
						for (int r = 0; r < batchSize; r++) {
							final double[][] error = layer.getError(r, j);
							if (deltaKernel == null)
								deltaKernel = Util.convnValid(lastLayer.getMap(r, i), error);
							else {
								deltaKernel = Util.matrixOp(Util.convnValid(lastLayer.getMap(r, i), error), deltaKernel, null, null, Util.plus);
							}
						}

						deltaKernel = Util.matrixOp(deltaKernel, divide_batchSize);
						final double[][] kernel = layer.getKernel(i, j);
						deltaKernel = Util.matrixOp(kernel, deltaKernel, multiply_lambda, multiply_alpha, Util.plus);
						layer.setKernel(i, j, deltaKernel);
					}
				}

			}
		};

		getRunner().startProcess(mapNum, process);
	}

	private void setHiddenLayerErrors() {
		for (int l = layerNum - 2; l > 0; l--) {
			final Layer layer = layers.get(l);
			final Layer nextLayer = layers.get(l + 1);
			switch (layer.getType()) {
				case samp:
					setSampErrors(layer, nextLayer);
					break;
				case conv:
					setConvErrors(layer, nextLayer);
					break;
				default:
					break;
			}
		}
	}

	private void setSampErrors(final Layer layer, final Layer nextLayer) {
		final int mapNum = layer.getOutMapNum();
		final int nextMapNum = nextLayer.getOutMapNum();

		final Process process = new Process() {
			@Override
			public void process(int start, int end) {
				for (int i = start; i < end; i++) {
					double[][] sum = null;// ��ÿһ������������
					for (int j = 0; j < nextMapNum; j++) {
						final double[][] nextError = nextLayer.getError(j);
						final double[][] kernel = nextLayer.getKernel(i, j);
						if (sum == null)
							sum = Util.convnFull(nextError, Util.rot180(kernel));
						else
							sum = Util.matrixOp(Util.convnFull(nextError, Util.rot180(kernel)), sum, null, null, Util.plus);
					}
					layer.setError(i, sum);
				}
			}

		};

		getRunner().startProcess(mapNum, process);
	}

	private void setConvErrors(final Layer layer, final Layer nextLayer) {
		final int mapNum = layer.getOutMapNum();

		final Process process = new Process() {
			@Override
			public void process(int start, int end) {
				for (int m = start; m < end; m++) {
					final Layer.Size scale = nextLayer.getScaleSize();
					final double[][] nextError = nextLayer.getError(m);
					final double[][] map = layer.getMap(m);
					double[][] outMatrix = Util.matrixOp(map, Util.cloneMatrix(map), null, Util.one_value, Util.multiply);
					outMatrix = Util.matrixOp(outMatrix, Util.kronecker(nextError, scale), null, null, Util.multiply);
					layer.setError(m, outMatrix);
				}
			}
		};

		getRunner().startProcess(mapNum, process);
	}

	private boolean setOutLayerErrors(final Dataset.Record record) {

		final Layer outputLayer = layers.get(layerNum - 1);
		final int mapNum = outputLayer.getOutMapNum();
		final double[] target = new double[mapNum];
		final double[] outmaps = new double[mapNum];

		for (int m = 0; m < mapNum; m++) {
			final double[][] outmap = outputLayer.getMap(m);
			outmaps[m] = outmap[0][0];
		}

		final int label = record.getLabel().intValue();

		target[label] = 1;

		for (int m = 0; m < mapNum; m++) {
			outputLayer.setError(m, 0, 0, outmaps[m] * (1 - outmaps[m]) * (target[m] - outmaps[m]));
		}

		return label == Util.getMaxIndex(outmaps);
	}

	/**
	 * Propagate given values through the network.
	 * Returns the results.
	 * For each input it returns the set of output values.
	 *
	 * @param inputs A list of vectors of input values
	 * @return A list of results of the network corresponding to each input vector
	 */
	public double[][] propagate(final double[][] inputs) {
		final double[][] results = new double[inputs.length][];

		int index = 0;
		for (final double[] input : inputs) {
			final Dataset.Record record = new Dataset.Record(input, -1.);
			results[index] = propagate(record);
			index++;
		}

		return results;
	}

	/**
	 * Propagate given values through the network.
	 * Returns the result.
	 *
	 * @param inputs A vector of input values
	 * @return The result of the network
	 */
	public double[] propagate(final double[] inputs) {
		final Dataset.Record record = new Dataset.Record(inputs, -1.);

		return propagate(record);
	}

	/**
	 * Propagate given Record through the network.
	 * Returns the result.
	 *
	 * @param record A Record
	 * @return The result of the network
	 */
	public double[] propagate(final Dataset.Record record) {
		forward(record);

		return getOutput();
	}

	private void forward(final Dataset.Record record) {
		setInLayerOutput(record);

		for (int l = 1; l < layers.size(); l++) {
			final Layer layer = layers.get(l);
			final Layer lastLayer = layers.get(l - 1);

			switch (layer.getType()) {
				case conv:
					setConvOutput(layer, lastLayer);
					break;

				case samp:
					setSampOutput(layer, lastLayer);
					break;

				case output:
					setConvOutput(layer, lastLayer);
					break;

				default:
					break;
			}
		}
	}

	private void setInLayerOutput(final Dataset.Record record) {
		final Layer inputLayer = layers.get(0);
		final Layer.Size mapSize = inputLayer.getMapSize();

		final double[] attr = record.getAttrs();

		if (attr.length != mapSize.x * mapSize.y) {
			throw new RuntimeException("The size of the data record does not match the size of the map defined!");
		}

		for (int i = 0; i < mapSize.x; i++) {
			for (int j = 0; j < mapSize.y; j++) {
				inputLayer.setMapValue(0, i, j, attr[mapSize.x * i + j]);
			}
		}
	}

	/**
	 * Compute the output of the convolutional layer, each thread is responsible for part of the map
	 */
	private void setConvOutput(final Layer layer, final Layer lastLayer) {
		final int mapNum = layer.getOutMapNum();
		final int lastMapNum = lastLayer.getOutMapNum();

		final Process process = new Process() {
			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					double[][] sum = null;
					for (int i = 0; i < lastMapNum; i++) {
						double[][] lastMap = lastLayer.getMap(i);
						double[][] kernel = layer.getKernel(i, j);
						if (sum == null) {
							sum = Util.convnValid(lastMap, kernel);
						} else {
							sum = Util.matrixOp(Util.convnValid(lastMap, kernel), sum, null, null, Util.plus);
						}
					}
					final double bias = layer.getBias(j);
					sum = Util
							.matrixOp(
									sum,
									new Util.Operator() {
										private static final long serialVersionUID = 2469461972825890810L;

										@Override
										public double process(double value) {
											return Util.sigmod(value + bias);
										}
									}
							);

					layer.setMapValue(j, sum);
				}
			}
		};

		getRunner().startProcess(mapNum, process);
	}

	private void setSampOutput(final Layer layer, final Layer lastLayer) {
		final int lastMapNum = lastLayer.getOutMapNum();

		final Process process = new Process() {
			@Override
			public void process(int start, int end) {
				for (int i = start; i < end; i++) {
					final double[][] lastMap = lastLayer.getMap(i);
					final Layer.Size scaleSize = layer.getScaleSize();
					final double[][] sampMatrix = Util.scaleMatrix(lastMap, scaleSize);
					layer.setMapValue(i, sampMatrix);
				}
			}
		};

		getRunner().startProcess(lastMapNum, process);
	}

	private void setup(final int batchSize) {
		final Layer inputLayer = layers.get(0);

		inputLayer.initOutmaps(batchSize);

		for (int i = 1; i < layers.size(); i++) {

			final Layer layer = layers.get(i);
			final Layer frontLayer = layers.get(i - 1);

			final int frontMapNum = frontLayer.getOutMapNum();
			switch (layer.getType()) {
				case input:
					break;
				case conv:
					layer.setMapSize(frontLayer.getMapSize().subtract(layer.getKernelSize(), 1));
					layer.initKernel(frontMapNum);
					layer.initBias(frontMapNum);
					layer.initErros(batchSize);
					layer.initOutmaps(batchSize);
					break;

				case samp:
					layer.setOutMapNum(frontMapNum);
					layer.setMapSize(frontLayer.getMapSize().divide(layer.getScaleSize()));
					layer.initErros(batchSize);
					layer.initOutmaps(batchSize);
					break;

				case output:
					layer.initOutputKerkel(frontMapNum, frontLayer.getMapSize());
					layer.initBias(frontMapNum);
					layer.initErros(batchSize);
					layer.initOutmaps(batchSize);
					break;
			}
		}
	}

	// === inner classes ===

	public static class LayerBuilder {
		private List<Layer> layerList;

		public LayerBuilder() {
			layerList = new ArrayList<>();
		}

		public LayerBuilder(Layer layer) {
			this();
			layerList.add(layer);
		}

		public LayerBuilder addLayer(Layer layer) {
			layerList.add(layer);
			return this;
		}
	}

}
