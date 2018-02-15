package edu.hitsz.c102c.cnn;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
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
	/**
	 *
	 */
	private static final long serialVersionUID = 337920299147929932L;
	private static double ALPHA = 0.85;
	protected static final double LAMBDA = 0;
	// ����ĸ���
	private List<Layer> layers;
	// ����
	private int layerNum;

	// �������µĴ�С
	private int batchSize;
	// �������������Ծ����ÿһ��Ԫ�س���һ��ֵ
	private Operator divide_batchSize;

	// �������������Ծ����ÿһ��Ԫ�س���alphaֵ
	private Operator multiply_alpha;

	// �������������Ծ����ÿһ��Ԫ�س���1-labmda*alphaֵ
	private Operator multiply_lambda;

	/**
	 * ��ʼ������
	 *
	 * @param layerBuilder �����
	 * @param inputMapSize ����map�Ĵ�С
	 * @param classNum     ���ĸ�����Ҫ�����ݼ������ת��Ϊ0-classNum-1����ֵ
	 */
	public CNN(LayerBuilder layerBuilder, final int batchSize) {
		layers = layerBuilder.mLayers;
		layerNum = layers.size();
		this.batchSize = batchSize;
		setup(batchSize);
		initPerator();
	}

	/**
	 * ��ʼ��������
	 */
	private void initPerator() {
		divide_batchSize = new Operator() {

			private static final long serialVersionUID = 7424011281732651055L;

			@Override
			public double process(double value) {
				return value / batchSize;
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

	/**
	 * ��ѵ������ѵ������
	 *
	 * @param trainset
	 * @param repeat   �����Ĵ���
	 */
	public void train(Dataset trainset, int repeat) {
		// ����ֹͣ��ť
		new Lisenter().start();
		for (int t = 0; t < repeat && !stopTrain.get(); t++) {
			int epochsNum = trainset.size() / batchSize;
			if (trainset.size() % batchSize != 0)
				epochsNum++;// ���ȡһ�Σ�������ȡ��
			Log.i("");
			Log.i(t + "th iter epochsNum:" + epochsNum);
			int right = 0;
			int count = 0;
			for (int i = 0; i < epochsNum; i++) {
				int[] randPerm = Util.randomPerm(trainset.size(), batchSize);
				Layer.prepareForNewBatch();

				for (int index : randPerm) {
					boolean isRight = train(trainset.getRecord(index));
					if (isRight)
						right++;
					count++;
					Layer.prepareForNewRecord();
				}

				// ����һ��batch�����Ȩ��
				updateParas();
				if (i % 50 == 0) {
					System.out.print("..");
					if (i + 50 > epochsNum)
						System.out.println();
				}
			}
			double p = 1.0 * right / count;
			if (t % 10 == 1 && p > 0.96) {//��̬����׼ѧϰ����
				ALPHA = 0.001 + ALPHA * 0.9;
				Log.i("Set alpha = " + ALPHA);
			}
			Log.i("precision " + right + "/" + count + "=" + p);
		}
	}

	private static AtomicBoolean stopTrain;

	static class Lisenter extends Thread {
		Lisenter() {
			setDaemon(true);
			stopTrain = new AtomicBoolean(false);
		}

		@Override
		public void run() {
			System.out.println("Input & to stop train.");
			while (true) {
				try {
					int a = System.in.read();
					if (a == '&') {
						stopTrain.compareAndSet(false, true);
						break;
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			System.out.println("Lisenter stop");
		}

	}

	/**
	 * ��������
	 *
	 * @param trainset
	 * @return
	 */
	public double test(Dataset trainset) {
		Layer.prepareForNewBatch();
		Iterator<Record> iter = trainset.iter();
		int right = 0;
		while (iter.hasNext()) {
			Record record = iter.next();
			forward(record);
			Layer outputLayer = layers.get(layerNum - 1);
			int mapNum = outputLayer.getOutMapNum();
			double[] out = new double[mapNum];
			for (int m = 0; m < mapNum; m++) {
				double[][] outmap = outputLayer.getMap(m);
				out[m] = outmap[0][0];
			}
			if (record.getLabel().intValue() == Util.getMaxIndex(out))
				right++;
		}
		double p = 1.0 * right / trainset.size();
		Log.i("precision", p + "");
		return p;
	}

	/**
	 * Ԥ����
	 *
	 * @param testset
	 * @param fileName
	 */
	public void predict(Dataset testset, String fileName) {
		Log.i("begin predict");
		try {
			int max = layers.get(layerNum - 1).getClassNum();
			PrintWriter writer = new PrintWriter(new File(fileName));
			Layer.prepareForNewBatch();
			Iterator<Record> iter = testset.iter();
			while (iter.hasNext()) {
				Record record = iter.next();
				forward(record);
				Layer outputLayer = layers.get(layerNum - 1);

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
		Log.i("end predict");
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

	/**
	 * ѵ��һ����¼��ͬʱ�����Ƿ�Ԥ����ȷ��ǰ��¼
	 *
	 * @param record
	 * @return
	 */
	private boolean train(Record record) {
		forward(record);
		boolean result = backPropagation(record);
		return result;
		// System.exit(0);
	}

	/*
	 * ������
	 */
	private boolean backPropagation(Record record) {
		boolean result = setOutLayerErrors(record);
		setHiddenLayerErrors();
		return result;
	}

	/**
	 * ���²���
	 */
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

	/**
	 * ����ƫ��
	 *
	 * @param layer
	 * @param lastLayer
	 */
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

	/**
	 * ����layer��ľ���ˣ�Ȩ�أ���ƫ��
	 *
	 * @param layer     ��ǰ��
	 * @param lastLayer ǰһ��
	 */
	private void updateKernels(final Layer layer, final Layer lastLayer) {
		int mapNum = layer.getOutMapNum();
		final int lastMapNum = lastLayer.getOutMapNum();
		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					for (int i = 0; i < lastMapNum; i++) {
						// ��batch��ÿ����¼delta���
						double[][] deltaKernel = null;
						for (int r = 0; r < batchSize; r++) {
							double[][] error = layer.getError(r, j);
							if (deltaKernel == null)
								deltaKernel = Util.convnValid(
										lastLayer.getMap(r, i), error);
							else {// �ۻ����
								deltaKernel = Util.matrixOp(Util.convnValid(
										lastLayer.getMap(r, i), error),
										deltaKernel, null, null, Util.plus);
							}
						}

						// ����batchSize
						deltaKernel = Util.matrixOp(deltaKernel,
								divide_batchSize);
						// ���¾����
						double[][] kernel = layer.getKernel(i, j);
						deltaKernel = Util.matrixOp(kernel, deltaKernel,
								multiply_lambda, multiply_alpha, Util.plus);
						layer.setKernel(i, j, deltaKernel);
					}
				}

			}
		}.start();

	}

	/**
	 * �����н�����Ĳв�
	 */
	private void setHiddenLayerErrors() {
		for (int l = layerNum - 2; l > 0; l--) {
			Layer layer = layers.get(l);
			Layer nextLayer = layers.get(l + 1);
			switch (layer.getType()) {
				case samp:
					setSampErrors(layer, nextLayer);
					break;
				case conv:
					setConvErrors(layer, nextLayer);
					break;
				default:// ֻ�в�����;������Ҫ����в�����û�вв������Ѿ������
					break;
			}
		}
	}

	/**
	 * ���ò�����Ĳв�
	 *
	 * @param layer
	 * @param nextLayer
	 */
	private void setSampErrors(final Layer layer, final Layer nextLayer) {
		int mapNum = layer.getOutMapNum();
		final int nextMapNum = nextLayer.getOutMapNum();
		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int i = start; i < end; i++) {
					double[][] sum = null;// ��ÿһ������������
					for (int j = 0; j < nextMapNum; j++) {
						double[][] nextError = nextLayer.getError(j);
						double[][] kernel = nextLayer.getKernel(i, j);
						// �Ծ���˽���180����ת��Ȼ�����fullģʽ�µþ��
						if (sum == null)
							sum = Util
									.convnFull(nextError, Util.rot180(kernel));
						else
							sum = Util.matrixOp(
									Util.convnFull(nextError,
											Util.rot180(kernel)), sum, null,
									null, Util.plus);
					}
					layer.setError(i, sum);
				}
			}

		}.start();

	}

	/**
	 * ���þ����Ĳв�
	 *
	 * @param layer
	 * @param nextLayer
	 */
	private void setConvErrors(final Layer layer, final Layer nextLayer) {
		// ��������һ��Ϊ�����㣬�������map������ͬ����һ��mapֻ����һ���һ��map���ӣ�
		// ���ֻ�轫��һ��Ĳв�kronecker��չ���õ������
		int mapNum = layer.getOutMapNum();
		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int m = start; m < end; m++) {
					Size scale = nextLayer.getScaleSize();
					double[][] nextError = nextLayer.getError(m);
					double[][] map = layer.getMap(m);
					// ������ˣ����Եڶ��������ÿ��Ԫ��value����1-value����
					double[][] outMatrix = Util.matrixOp(map,
							Util.cloneMatrix(map), null, Util.one_value,
							Util.multiply);
					outMatrix = Util.matrixOp(outMatrix,
							Util.kronecker(nextError, scale), null, null,
							Util.multiply);
					layer.setError(m, outMatrix);
				}

			}

		}.start();

	}

	/**
	 * ���������Ĳв�ֵ,�������񾭵�Ԫ�������٣��ݲ����Ƕ��߳�
	 *
	 * @param record
	 * @return
	 */
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

		double[] target = new double[mapNum];
		double[] outmaps = new double[mapNum];
		for (int m = 0; m < mapNum; m++) {
			double[][] outmap = outputLayer.getMap(m);
			outmaps[m] = outmap[0][0];

		}
		int label = record.getLabel().intValue();
		target[label] = 1;
		// Log.i(record.getLable() + "outmaps:" +
		// Util.fomart(outmaps)
		// + Arrays.toString(target));
		for (int m = 0; m < mapNum; m++) {
			outputLayer.setError(m, 0, 0, outmaps[m] * (1 - outmaps[m])
					* (target[m] - outmaps[m]));
		}
		return label == Util.getMaxIndex(outmaps);
	}

	/**
	 * ǰ�����һ����¼
	 *
	 * @param record
	 */
	private void forward(Record record) {
		// ����������map
		setInLayerOutput(record);
		for (int l = 1; l < layers.size(); l++) {
			Layer layer = layers.get(l);
			Layer lastLayer = layers.get(l - 1);
			switch (layer.getType()) {
				case conv:// ������������
					setConvOutput(layer, lastLayer);
					break;
				case samp:// �������������
					setSampOutput(layer, lastLayer);
					break;
				case output:// �������������,�������һ������ľ����
					setConvOutput(layer, lastLayer);
					break;
				default:
					break;
			}
		}
	}

	/**
	 * ���ݼ�¼ֵ���������������ֵ
	 *
	 * @param record
	 */
	private void setInLayerOutput(Record record) {
		final Layer inputLayer = layers.get(0);
		final Size mapSize = inputLayer.getMapSize();
		final double[] attr = record.getAttrs();
		if (attr.length != mapSize.x * mapSize.y)
			throw new RuntimeException("���ݼ�¼�Ĵ�С�붨���map��С��һ��!");
		for (int i = 0; i < mapSize.x; i++) {
			for (int j = 0; j < mapSize.y; j++) {
				// ����¼���Ե�һά����Ū�ɶ�ά����
				inputLayer.setMapValue(0, i, j, attr[mapSize.x * i + j]);
			}
		}
	}

	/*
	 * �����������ֵ,ÿ���̸߳���һ����map
	 */
	private void setConvOutput(final Layer layer, final Layer lastLayer) {
		int mapNum = layer.getOutMapNum();
		final int lastMapNum = lastLayer.getOutMapNum();
		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					double[][] sum = null;// ��ÿһ������map�ľ���������
					for (int i = 0; i < lastMapNum; i++) {
						double[][] lastMap = lastLayer.getMap(i);
						double[][] kernel = layer.getKernel(i, j);
						if (sum == null)
							sum = Util.convnValid(lastMap, kernel);
						else
							sum = Util.matrixOp(
									Util.convnValid(lastMap, kernel), sum,
									null, null, Util.plus);
					}
					final double bias = layer.getBias(j);
					sum = Util.matrixOp(sum, new Operator() {
						private static final long serialVersionUID = 2469461972825890810L;

						@Override
						public double process(double value) {
							return Util.sigmod(value + bias);
						}

					});

					layer.setMapValue(j, sum);
				}
			}

		}.start();

	}

	/**
	 * ���ò���������ֵ���������ǶԾ����ľ�ֵ����
	 *
	 * @param layer
	 * @param lastLayer
	 */
	private void setSampOutput(final Layer layer, final Layer lastLayer) {
		int lastMapNum = lastLayer.getOutMapNum();
		new TaskManager(lastMapNum) {

			@Override
			public void process(int start, int end) {
				for (int i = start; i < end; i++) {
					double[][] lastMap = lastLayer.getMap(i);
					Size scaleSize = layer.getScaleSize();
					// ��scaleSize������о�ֵ����
					double[][] sampMatrix = Util
							.scaleMatrix(lastMap, scaleSize);
					layer.setMapValue(i, sampMatrix);
				}
			}

		}.start();

	}

	/**
	 * ����cnn�����ÿһ��Ĳ���
	 *
	 * @param batchSize    * @param classNum
	 * @param inputMapSize
	 */
	public void setup(int batchSize) {
		Layer inputLayer = layers.get(0);
		// ÿһ�㶼��Ҫ��ʼ�����map
		inputLayer.initOutmaps(batchSize);
		for (int i = 1; i < layers.size(); i++) {
			Layer layer = layers.get(i);
			Layer frontLayer = layers.get(i - 1);
			int frontMapNum = frontLayer.getOutMapNum();
			switch (layer.getType()) {
				case input:
					break;
				case conv:
					// ����map�Ĵ�С
					layer.setMapSize(frontLayer.getMapSize().subtract(
							layer.getKernelSize(), 1));
					// ��ʼ������ˣ�����frontMapNum*outMapNum�������

					layer.initKernel(frontMapNum);
					// ��ʼ��ƫ�ã�����frontMapNum*outMapNum��ƫ��
					layer.initBias(frontMapNum);
					// batch��ÿ����¼��Ҫ����һ�ݲв�
					layer.initErros(batchSize);
					// ÿһ�㶼��Ҫ��ʼ�����map
					layer.initOutmaps(batchSize);
					break;
				case samp:
					// �������map��������һ����ͬ
					layer.setOutMapNum(frontMapNum);
					// ������map�Ĵ�С����һ��map�Ĵ�С����scale��С
					layer.setMapSize(frontLayer.getMapSize().divide(
							layer.getScaleSize()));
					// batch��ÿ����¼��Ҫ����һ�ݲв�
					layer.initErros(batchSize);
					// ÿһ�㶼��Ҫ��ʼ�����map
					layer.initOutmaps(batchSize);
					break;
				case output:
					// ��ʼ��Ȩ�أ�����ˣ��������ľ���˴�СΪ��һ���map��С
					layer.initOutputKerkel(frontMapNum, frontLayer.getMapSize());
					// ��ʼ��ƫ�ã�����frontMapNum*outMapNum��ƫ��
					layer.initBias(frontMapNum);
					// batch��ÿ����¼��Ҫ����һ�ݲв�
					layer.initErros(batchSize);
					// ÿһ�㶼��Ҫ��ʼ�����map
					layer.initOutmaps(batchSize);
					break;
			}
		}
	}

	/**
	 * ������ģʽ�������,Ҫ�����ڶ������Ϊ�����������Ϊ�����
	 *
	 * @author jiqunpeng
	 * <p>
	 * ����ʱ�䣺2014-7-8 ����4:54:29
	 */
	public static class LayerBuilder {
		private List<Layer> mLayers;

		public LayerBuilder() {
			mLayers = new ArrayList<Layer>();
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

	/**
	 * ���л�����ģ��
	 *
	 * @param fileName
	 */
	public void saveModel(String fileName) {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(
					new FileOutputStream(fileName));
			oos.writeObject(this);
			oos.flush();
			oos.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	/**
	 * �����л�����ģ��
	 *
	 * @param fileName
	 * @return
	 */
	public static CNN loadModel(String fileName) {
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
					fileName));
			CNN cnn = (CNN) in.readObject();
			in.close();
			return cnn;
		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
		}
		return null;
	}
}
