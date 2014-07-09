package edu.hitsz.c102c.cnn;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;

import edu.hitsz.c102c.cnn.Layer.Size;
import edu.hitsz.c102c.data.Dataset;
import edu.hitsz.c102c.data.Dataset.Record;
import edu.hitsz.c102c.util.ConcurenceRunner;
import edu.hitsz.c102c.util.ConcurenceRunner.Task;
import edu.hitsz.c102c.util.TimedTest;
import edu.hitsz.c102c.util.TimedTest.TestTask;
import edu.hitsz.c102c.util.Util;

public class CNN {
	// 网络的各层
	private List<Layer> layers;
	// 层数
	private int layerNum;
	// 并行工具
	private static ConcurenceRunner runner = new ConcurenceRunner();

	/**
	 * 初始化网络
	 * 
	 * @param layerBuilder
	 *            网络层
	 * @param inputMapSize
	 *            输入map的大小
	 * @param classNum
	 *            类别的个数，要求数据集将类标转化为0-classNum-1的数值
	 */
	public CNN(LayerBuilder layerBuilder, int batchSize) {
		layers = layerBuilder.mLayers;
		layerNum = layers.size();
		setup(batchSize);
	}

	/**
	 * 在训练集上训练网络
	 * 
	 * @param trainset
	 */
	public void train(Dataset trainset, int batchSize) {
		int epochsNum = 1 + trainset.size() / batchSize;// 多抽取一次，即向上取整
		for (int i = 0; i < epochsNum; i++) {
			int[] randPerm = Util.randomPerm(trainset.size(), batchSize);
			Layer.prepareForNewBatch();
			for (int index : randPerm) {
				train(trainset.getRecord(index));
			}

		}
	}

	private void train(Record record) {
		forward(record);
		backPropagation(record);
	}

	/*
	 * 反向传输
	 */
	private void backPropagation(Record record) {
		setOutLayerErrors(record);
		setHiddenLayerErros();

	}

	/**
	 * 设置中将各层的残差
	 */
	private void setHiddenLayerErros() {
		for (int l = layerNum - 2; l > 0; l--) {
			Layer layer = layers.get(l);
			int mapNum = layer.getOutMapNum();
			Layer nextLayer = layers.get(l + 1);
			switch (layer.getType()) {
			case samp:

				break;
			case conv:
				// 卷积层的下一层为采样层，即两层的map个数相同，且一个map只与令一层的一个map连接，
				// 因此只需将下一层的残差kronecker扩展在用点积即可
				for (int m = 0; m < mapNum; m++) {
					Size scale = nextLayer.getScaleSize();
					double[][] nextError = nextLayer.getError(m);
					double[][] map = layer.getMap(m);
					// 矩阵相乘，但对第二个矩阵的每个元素value进行1-value操作
					matrixProduct(map, cloneMatrix(map), null, new Operator() {

						@Override
						public double process(double value) {
							return 1 - value;
						}

					});
					double[][] outMatrix = matrixProduct(map,
							kronecker(nextError, scale), null, null);

					layer.setError(m, outMatrix);

				}
				break;
			default:
				break;
			}
		}
	}

	/**
	 * 设置输出层的残差值
	 * 
	 * @param record
	 */
	private void setOutLayerErrors(Record record) {
		Layer outputLayer = layers.get(layerNum - 1);
		int mapNum = outputLayer.getOutMapNum();
		double[] target = record.getDoubleEncodeTarget(mapNum);
		for (int m = 0; m < mapNum; m++) {
			double[][] outmap = outputLayer.getMap(m);
			double output = outmap[0][0];
			double errors = output * (1 - output) * (target[m] - output);
			outputLayer.setError(m, 0, 0, errors);
		}
	}

	/**
	 * 前向计算一条记录
	 * 
	 * @param record
	 */
	private void forward(Record record) {
		// 设置输入层的map
		Layer inputLayer = layers.get(0);
		Size mapSize = inputLayer.getMapSize();
		double[] attr = record.getAttrs();
		if (attr.length != mapSize.x * mapSize.y)
			throw new RuntimeException("数据记录的大小与定义的map大小不一致!");
		int index = 0;
		for (int i = 0; i < mapSize.x; i++)
			for (int j = 0; j < mapSize.y; j++) {
				double value = attr[index++];
				inputLayer.setMapValue(0, i, j, value);
			}
		for (int l = 1; l < layers.size(); l++) {
			Layer layer = layers.get(l);
			Layer lastLayer = layers.get(l - 1);
			int mapNum = layer.getOutMapNum();
			int lastMapNum = lastLayer.getOutMapNum();
			switch (layer.getType()) {
			case conv:// 计算卷积层的输出
				for (int j = 0; j < mapNum; j++)
					for (int i = 0; i < lastMapNum; i++) {
						double[][] lastMap = lastLayer.getMap(i);
						double[][] kernel = layer.getKernel(i, j);
						double[][] outMatrix = convnValid(lastMap, kernel);
						layer.setMapValue(j, outMatrix);
					}
				break;
			case samp:// 计算采样层的输出
				for (int i = 0; i < lastMapNum; i++) {
					double[][] lastMap = lastLayer.getMap(i);
					Size scaleSize = layer.getScaleSize();
					double[][] sampMatrix = scaleMatrix(lastMap, scaleSize);
					layer.setMapValue(i, sampMatrix);
				}
				break;
			case output:// 计算输出层的输出
				for (int j = 0; j < mapNum; j++)
					for (int i = 0; i < lastMapNum; i++) {
						double[][] lastMap = lastLayer.getMap(i);
						double[][] kernel = layer.getKernel(i, j);
						double[][] outMatrix = convnValid(lastMap, kernel);
						layer.setMapValue(j, outMatrix);
					}
				break;
			default:
				break;
			}
		}
	}

	/**
	 * 设置cnn网络的每一层的参数
	 * 
	 * @param batchSize
	 * 
	 * @param classNum
	 * @param inputMapSize
	 */
	public void setup(int batchSize) {
		for (int i = 1; i < layers.size(); i++) {
			Layer layer = layers.get(i);
			Layer frontLayer = layers.get(i - 1);
			int frontMapNum = frontLayer.getOutMapNum();
			switch (layer.getType()) {
			case input:
				break;
			case conv:
				// 设置map的大小
				layer.setMapSize(frontLayer.getMapSize().subtract(
						layer.getKernelSize(), 1));
				// 初始化卷积核，共有frontMapNum*outMapNum个卷积核
				layer.initKerkel(frontMapNum);
				// 初始化偏置，共有frontMapNum*outMapNum个偏置
				layer.initBias(frontMapNum);
				break;
			case samp:
				// 采样层的map数量与上一层相同
				layer.setOutMapNum(frontMapNum);
				// 采样层map的大小是上一层map的大小除以scale大小
				layer.setMapSize(frontLayer.getMapSize().divide(
						layer.getScaleSize()));
				break;
			case output:
				// 初始化权重（卷积核），共有frontMapNum*outMapNum个1*1卷积核
				layer.initKerkel(frontMapNum);
				// 初始化偏置，共有frontMapNum*outMapNum个偏置
				layer.initBias(frontMapNum);
				break;
			}
			// 每一层都需要初始化输出map
			layer.initOutmaps(batchSize);
		}
	}

	/**
	 * 构造者模式构造各层,要求倒数第二层必须为采样层而不能为卷积层
	 * 
	 * @author jiqunpeng
	 * 
	 *         创建时间：2014-7-8 下午4:54:29
	 */
	class LayerBuilder {
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

	public static double[][] cloneMatrix(final double[][] matrix) {
		final int m = matrix.length;
		int n = matrix[0].length;
		final double[][] outMatrix = new double[m][n];
		int cpuNum = ConcurenceRunner.cpuNum;
		cpuNum = cpuNum < n ? cpuNum : 1;// 比cpu的个数小时，只用一个线程
		int fregLength = (n + cpuNum - 1) / cpuNum;// 向上取整
		final CountDownLatch gate = new CountDownLatch(cpuNum);
		for (int cpu = 0; cpu < cpuNum; cpu++) {
			int start = cpu * fregLength;
			int tmp = (cpu + 1) * fregLength;
			int end = tmp <= n ? tmp : n;
			Task task = new Task(start, end) {

				@Override
				public void process(int start, int end) {
					for (int i = 0; i < m; i++) {
						for (int j = start; j < end; j++) {
							outMatrix[i][j] = matrix[i][j];
						}
					}
					gate.countDown();
				}

			};
			runner.run(task);
		}
		await(gate);
		return outMatrix;
	}

	/**
	 * 两个维度相同的矩阵对应元素相乘,得到的结果方法mb中，即mb[i][j] =
	 * ma[i][j]*mb[i][j]
	 * 
	 * @param ma
	 * @param mb
	 * @param operatorB
	 *            在第mb矩阵上的操作
	 * @param operatorA
	 *            在ma矩阵元素上的操作
	 * @return
	 * @deprecated 会对mb矩阵进行修改，请注意
	 */
	private static double[][] matrixProduct(final double[][] ma,
			final double[][] mb, final Operator operatorA,
			final Operator operatorB) {
		final int m = ma.length;
		int n = ma[0].length;
		if (m != mb.length || n != mb[0].length)
			throw new RuntimeException("两个矩阵大小不一致");
		int cpuNum = ConcurenceRunner.cpuNum;
		cpuNum = cpuNum < n ? cpuNum : 1;// 比cpu的个数小时，只用一个线程
		int fregLength = (n + cpuNum - 1) / cpuNum;// 向上取整
		final CountDownLatch gate = new CountDownLatch(cpuNum);
		for (int cpu = 0; cpu < cpuNum; cpu++) {
			int start = cpu * fregLength;
			int tmp = (cpu + 1) * fregLength;
			int end = tmp <= n ? tmp : n;
			Task task = new Task(start, end) {

				@Override
				public void process(int start, int end) {
					for (int i = 0; i < m; i++) {
						for (int j = start; j < end; j++) {
							double a = ma[i][j];
							if (operatorA != null)
								a = operatorA.process(a);
							double b = mb[i][j];
							if (operatorB != null)
								b = operatorB.process(b);
							mb[i][j] = a * b;
						}
					}
					gate.countDown();
				}

			};
			runner.run(task);
		}
		return mb;
	}

	/**
	 * 克罗内克积,对矩阵进行扩展
	 * 
	 * @param matrix
	 * @param scale
	 * @return
	 */
	private static double[][] kronecker(final double[][] matrix,
			final Size scale) {
		final int m = matrix.length;
		int n = matrix[0].length;
		final double[][] outMatrix = new double[m * scale.x][n * scale.y];
		int cpuNum = ConcurenceRunner.cpuNum;
		cpuNum = cpuNum < n ? cpuNum : 1;// 比cpu的个数小时，只用一个线程
		int fregLength = (n + cpuNum - 1) / cpuNum;// 向上取整
		final CountDownLatch gate = new CountDownLatch(cpuNum);
		for (int cpu = 0; cpu < cpuNum; cpu++) {
			int start = cpu * fregLength;
			int tmp = (cpu + 1) * fregLength;
			int end = tmp <= n ? tmp : n;
			Task task = new Task(start, end) {

				@Override
				public void process(int start, int end) {
					for (int i = 0; i < m; i++) {
						for (int j = start; j < end; j++) {
							for (int ki = i * scale.x; ki < (i + 1) * scale.x; ki++) {
								for (int kj = j * scale.y; kj < (j + 1)
										* scale.y; kj++) {
									outMatrix[ki][kj] = matrix[i][j];
								}
							}
						}
					}
					gate.countDown();
				}

			};
			runner.run(task);
		}
		await(gate);
		return outMatrix;
	}

	/**
	 * 对矩阵进行缩小
	 * 
	 * @param matrix
	 * @param scaleSize
	 * @return
	 */
	private static double[][] scaleMatrix(final double[][] matrix,
			final Size scale) {
		int m = matrix.length;
		int n = matrix[0].length;
		final int sm = m / scale.x;
		final int sn = n / scale.y;
		final double[][] outMatrix = new double[sm][sn];
		if (sm * scale.x != m || sn * scale.y != n)
			throw new RuntimeException("scale不能整除matrix");
		// 并发运行
		int cpuNum = ConcurenceRunner.cpuNum;
		cpuNum = cpuNum < sn ? cpuNum : 1;// 比cpu的个数小时，只用一个线程
		int fregLength = (sn + cpuNum - 1) / cpuNum;// 想上取整
		final CountDownLatch gate = new CountDownLatch(cpuNum);
		final int size = scale.x * scale.y;
		for (int cpu = 0; cpu < cpuNum; cpu++) {
			int start = cpu * fregLength;
			int tmp = (cpu + 1) * fregLength;
			int end = tmp <= sn ? tmp : sn;
			Task task = new Task(start, end) {
				@Override
				public void process(int start, int end) {
					for (int i = 0; i < sm; i++) {
						for (int j = start; j < end; j++) {
							double sum = 0.0;
							for (int si = i * scale.x; si < (i + 1) * scale.x; si++) {
								for (int sj = j * scale.y; sj < (j + 1)
										* scale.y; sj++) {
									sum += matrix[si][sj];
								}
							}
							outMatrix[i][j] = sum / size;
						}
					}
					gate.countDown();
				}
			};
			runner.run(task);

		}
		await(gate);
		return outMatrix;
	}

	/**
	 * 计算full模式的卷积
	 * 
	 * @param matrix
	 * @param kernel
	 * @return
	 */
	public static double[][] convnFull(double[][] matrix,
			final double[][] kernel) {
		int m = matrix.length;
		int n = matrix[0].length;
		final int km = kernel.length;
		final int kn = kernel[0].length;
		// 扩展矩阵
		final double[][] extendMatrix = new double[m + 2 * (km - 1)][n + 2
				* (kn - 1)];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++)
				extendMatrix[i + km - 1][j + kn - 1] = matrix[i][j];
		}
		return convnValid(extendMatrix, kernel);
	}

	/**
	 * 等待
	 * 
	 * @param gate
	 */
	private static void await(CountDownLatch gate) {
		try {
			gate.await();
		} catch (InterruptedException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}

	/**
	 * 计算valid模式的卷积
	 * 
	 * @param matrix
	 * @param kernel
	 * @return
	 */
	public static double[][] convnValid(final double[][] matrix,
			final double[][] kernel) {
		int m = matrix.length;
		int n = matrix[0].length;
		final int km = kernel.length;
		final int kn = kernel[0].length;
		// 需要做卷积的列数
		int kns = n - kn + 1;
		// 需要做卷积的行数
		final int kms = m - km + 1;
		// 结果矩阵
		final double[][] outMatrix = new double[kms][kns];
		// 并发运行
		int cpuNum = ConcurenceRunner.cpuNum;
		cpuNum = cpuNum < kns ? cpuNum : 1;// 比cpu的个数小时，只用一个线程
		int fregLength = (kns + cpuNum - 1) / cpuNum;
		// Log.i("kns:" + kns);
		// Log.i("fregLength:" + fregLength);
		final CountDownLatch gate = new CountDownLatch(cpuNum);
		for (int cpu = 0; cpu < cpuNum; cpu++) {
			int start = cpu * fregLength;
			int tmp = (cpu + 1) * fregLength;
			int end = tmp <= kns ? tmp : kns;
			Task task = new Task(start, end) {

				@Override
				public void process(int start, int end) {

					for (int i = 0; i < kms; i++) {
						for (int j = start; j < end; j++) {
							double sum = 0.0;
							for (int ki = 0; ki < km; ki++) {
								for (int kj = 0; kj < kn; kj++)
									sum += matrix[i + ki][j + kj]
											* kernel[ki][kj];
							}
							outMatrix[i][j] = sum;

						}
					}
					gate.countDown();
				}

			};
			runner.run(task);

		}
		await(gate);
		return outMatrix;

	}

	/**
	 * 测试卷积,测试结果：4核下并发并行的卷积提高不到2倍
	 */
	private static void testConvn() {
		int count = 1;
		double[][] m = new double[5000][500];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = count++;
		double[][] k = new double[1][1];
		for (int i = 0; i < k.length; i++)
			for (int j = 0; j < k[0].length; j++)
				k[i][j] = 1.5;
		double[][] out;
		// out= convnValid(m, k);
		// Util.printMatrix(m);
		out = convnFull(m, k);
		// Util.printMatrix(out);
		// System.out.println();
		// out = convnFull(m, Util.rot180(k));
		// Util.printMatrix(out);

	}

	private static void testScaleMatrix() {
		int count = 1;
		double[][] m = new double[20000][200];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = 1;
		double[][] out = scaleMatrix(m, new Size(4, 4));
		// Util.printMatrix(m);
		// System.out.println();
		// Util.printMatrix(out);
	}

	private static void testKronecker() {
		int count = 1;
		double[][] m = new double[5][5];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = count++;
		double[][] out = kronecker(m, new Size(1, 1));
		Util.printMatrix(m);
		System.out.println();
		Util.printMatrix(out);
	}

	private static void testMatrixProduct() {
		int count = 1;
		double[][] m = new double[5][5];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = count++;
		double[][] k = new double[5][5];
		for (int i = 0; i < k.length; i++)
			for (int j = 0; j < k[0].length; j++)
				k[i][j] = j;

		Util.printMatrix(m);
		Util.printMatrix(k);
		double[][] out = matrixProduct(m, k, new Operator() {

			@Override
			public double process(double value) {

				return value - 1;
			}
		}, new Operator() {

			@Override
			public double process(double value) {

				return -1 * value;
			}
		});
		Util.printMatrix(out);
	}

	private static void testCloneMatrix() {
		int count = 1;
		double[][] m = new double[5][5];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = count++;
		double[][] out = cloneMatrix(m);
		Util.printMatrix(m);

		Util.printMatrix(out);
	}

	public static void main(String[] args) {
		new TimedTest(new TestTask() {

			@Override
			public void process() {
				// testConvn();
				// testScaleMatrix();
				// testKronecker();
				testMatrixProduct();
				// testCloneMatrix();
			}
		}, 1).test();
		ConcurenceRunner.stop();
	}

	/**
	 * 矩阵对应元素相乘时在每个元素上的操作
	 * 
	 * @author jiqunpeng
	 * 
	 *         创建时间：2014-7-9 下午9:28:35
	 */
	interface Operator {
		public double process(double value);
	}
}
