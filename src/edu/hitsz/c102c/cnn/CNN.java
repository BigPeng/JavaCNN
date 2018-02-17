package edu.hitsz.c102c.cnn;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import edu.hitsz.c102c.cnn.Layer.Size;
import edu.hitsz.c102c.dataset.Dataset;
import edu.hitsz.c102c.dataset.Dataset.Record;
import edu.hitsz.c102c.util.ConcurenceRunner.TaskManager;
import edu.hitsz.c102c.util.Log;
import edu.hitsz.c102c.util.Util;
import edu.hitsz.c102c.util.Util.Operator;

public class CNN implements Serializable {

	private static final long serialVersionUID = 337920299147929932L;

	private static final double LAMBDA = 0;

	private static double ALPHA = 0.85;


	private List<Layer> layers;

	private int layerNum;

	private int batchSize;

	private Operator divide_batchSize;

	private Operator multiply_alpha;

	private Operator multiply_lambda;


	public CNN(LayerBuilder layerBuilder, final int batchSize) {
		layers = layerBuilder.mLayers;
		layerNum = layers.size();
		this.batchSize = batchSize;
		setup(batchSize);
		initPerator();
	}

	private void initPerator() {
		divide_batchSize = new Operator() {
			private static final long serialVersionUID = 7424011281732651055L;

			@Override
			public double process(double value) {
				return value / batchSize; // TODO: Remove division
			}

		};

		multiply_alpha = new Operator() {
			private static final long serialVersionUID = 5761368499808006552L;

			@Override
			public double process(double value) {
				return value * ALPHA;
			}

		};

		multiply_lambda = new Operator() {
			private static final long serialVersionUID = 4499087728362870577L;

			@Override
			public double process(double value) {
				return value * (1 - LAMBDA * ALPHA);
			}
		};
	}


	public void train(final Dataset trainset, final int iterationCount) {
		new Listener().start();

		for (int iteration = 0; iteration < iterationCount && !stopTrain.get(); iteration++) {

			int epochsNum = trainset.size() / batchSize;

			if (trainset.size() % batchSize != 0) {
				epochsNum++;
			}

			Log.info("");
			Log.info(iteration + "th iter epochsNum:" + epochsNum);

			int right = 0;
			int count = 0;

			for (int epoch = 0; epoch < epochsNum; epoch++) {

				int[] randPerm = Util.randomPerm(trainset.size(), batchSize);

				Layer.prepareForNewBatch();

				for (int index : randPerm) {
					boolean isRight = train(trainset.getRecord(index));
					if (isRight)
						right++;
					count++;
					Layer.prepareForNewRecord();
				}

				updateParas();

				if (epoch % 50 == 0) {
					System.out.print(".");
					if (epoch + 50 > epochsNum) {
						System.out.println();
					}
				}
			}

			final double p = 1.0 * right / count;

			if (iteration % 10 == 1 && p > 0.96) {
				ALPHA = 0.001 + ALPHA * 0.9;
				Log.info("Set alpha = " + ALPHA);
			}

			Log.info("precision " + right + "/" + count + "=" + p);
		}
	}

	private static AtomicBoolean stopTrain;

	static class Listener extends Thread {

		Listener() {
			setDaemon(true);
			stopTrain = new AtomicBoolean(false);
		}

		@Override
		public void run() {

			System.out.println("Input & to stop train.");

			while (true) {
				try {
					final int a = System.in.read();
					if (a == '&') {
						stopTrain.compareAndSet(false, true);
						break;
					}
				} catch (IOException e) {
					throw new RuntimeException(e);
				}
			}

			System.out.println("Listener stopped");
		}
	}

	public double test(final Dataset dataset) {
		Layer.prepareForNewBatch();

		final Iterator<Record> iterator = dataset.iter();

		int right = 0;
		while (iterator.hasNext()) {
			final Record record = iterator.next();

			forward(record);

			final Layer outputLayer = layers.get(layerNum - 1);
			final int mapNum = outputLayer.getOutMapNum();

			final double[] out = new double[mapNum];
			for (int m = 0; m < mapNum; m++) {
				final double[][] outmap = outputLayer.getMap(m);
				out[m] = outmap[0][0];
			}

			if (record.getLabel().intValue() == Util.getMaxIndex(out)) {
				right++;
			}
		}
		double p = 1.0 * right / dataset.size();

		Log.info("precision", p + "");

		return p;
	}

	// TODO: Move this method to other/new class (reduce CNN-class to the minimal CNN-logic)
	public void predict(Dataset testset, String fileName) {
		Log.info("begin predict");
		try {
			// final int max = layers.get(layerNum - 1).getClassNum();
			final PrintWriter writer = new PrintWriter(new File(fileName));

			Layer.prepareForNewBatch();

			final Iterator<Record> iter = testset.iter();
			while (iter.hasNext()) {
				final Record record = iter.next();
				forward(record);
				final Layer outputLayer = layers.get(layerNum - 1);

				int mapNum = outputLayer.getOutMapNum();
				double[] out = new double[mapNum];
				for (int m = 0; m < mapNum; m++) {
					double[][] outmap = outputLayer.getMap(m);
					out[m] = outmap[0][0];
				}
				// int label =
				// Util.binaryArray2int(out);
				int label = Util.getMaxIndex(out);
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

	private boolean isSame(double[] output, double[] target) {
		boolean r = true;
		for (int i = 0; i < output.length; i++)
			if (Math.abs(output[i] - target[i]) > 0.5) {
				r = false;
				break;
			}

		return r;
	}

	private boolean train(Record record) {
		forward(record);
		boolean result = backPropagation(record);
		return result;
		// System.exit(0);
	}

	private boolean backPropagation(Record record) {
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
					updateBias(layer, lastLayer);
					break;
				default:
					break;
			}
		}
	}

	private void updateBias(final Layer layer, Layer lastLayer) {
		final double[][][][] errors = layer.getErrors();
		int mapNum = layer.getOutMapNum();

		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					double[][] error = Util.sum(errors, j);
					// ����ƫ��
					double deltaBias = Util.sum(error) / batchSize;
					double bias = layer.getBias(j) + ALPHA * deltaBias;
					layer.setBias(j, bias);
				}
			}
		}.start();

	}

	private void updateKernels(final Layer layer, final Layer lastLayer) {
		final int mapNum = layer.getOutMapNum();
		final int lastMapNum = lastLayer.getOutMapNum();
		new TaskManager(mapNum) {

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
		}.start();

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
		new TaskManager(mapNum) {

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

		}.start();

	}

	private void setConvErrors(final Layer layer, final Layer nextLayer) {
		final int mapNum = layer.getOutMapNum();
		new TaskManager(mapNum) {
			@Override
			public void process(int start, int end) {
				for (int m = start; m < end; m++) {
					final Size scale = nextLayer.getScaleSize();
					final double[][] nextError = nextLayer.getError(m);
					final double[][] map = layer.getMap(m);
					double[][] outMatrix = Util.matrixOp(map, Util.cloneMatrix(map), null, Util.one_value, Util.multiply);
					outMatrix = Util.matrixOp(outMatrix, Util.kronecker(nextError, scale), null, null, Util.multiply);
					layer.setError(m, outMatrix);
				}
			}
		}.start();
	}

	private boolean setOutLayerErrors(Record record) {

		Layer outputLayer = layers.get(layerNum - 1);
		int mapNum = outputLayer.getOutMapNum();
		// double[] target =
		// record.getDoubleEncodeTarget(mapNum);
		// double[] outmaps = new double[mapNum];
		// for (int m = 0; m < mapNum; m++) {
		// double[][] outmap = outputLayer.getMap(m);
		// double output = outmap[0][0];
		// outmaps[m] = output;
		// double errors = output * (1 - output) *
		// (target[m] - output);
		// outputLayer.setError(m, 0, 0, errors);
		// }
		// // ��ȷ
		// if (isSame(outmaps, target))
		// return true;
		// return false;

		final double[] target = new double[mapNum];
		final double[] outmaps = new double[mapNum];

		for (int m = 0; m < mapNum; m++) {
			final double[][] outmap = outputLayer.getMap(m);
			outmaps[m] = outmap[0][0];
		}

		final int label = record.getLabel().intValue();

		target[label] = 1;

		// Log.i(record.getLable() + "outmaps:" +
		// Util.fomart(outmaps)
		// + Arrays.toString(target));

		for (int m = 0; m < mapNum; m++) {
			outputLayer.setError(m, 0, 0, outmaps[m] * (1 - outmaps[m]) * (target[m] - outmaps[m]));
		}

		return label == Util.getMaxIndex(outmaps);
	}

	private void forward(Record record) {
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

	private void setInLayerOutput(Record record) {
		final Layer inputLayer = layers.get(0);
		final Size mapSize = inputLayer.getMapSize();

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
		new TaskManager(mapNum) {

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
									new Operator() {
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
		}.start();
	}

	private void setSampOutput(final Layer layer, final Layer lastLayer) {
		final int lastMapNum = lastLayer.getOutMapNum();

		new TaskManager(lastMapNum) {
			@Override
			public void process(int start, int end) {
				for (int i = start; i < end; i++) {
					final double[][] lastMap = lastLayer.getMap(i);
					final Size scaleSize = layer.getScaleSize();
					final double[][] sampMatrix = Util.scaleMatrix(lastMap, scaleSize);
					layer.setMapValue(i, sampMatrix);
				}
			}
		}.start();

	}

	private void setup(int batchSize) {
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
		private List<Layer> mLayers;

		public LayerBuilder() {
			mLayers = new ArrayList<>();
		}

		public LayerBuilder(Layer layer) {
			this();
			mLayers.add(layer);
		}

		public LayerBuilder addLayer(Layer layer) {
			mLayers.add(layer);
			return this;
		}
	}

}
