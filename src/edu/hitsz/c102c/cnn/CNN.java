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
	// 网络的各层
	private List<Layer> layers;
	// 层数
	private int layerNum;

	// 批量更新的大小
	private int batchSize;
	// 除数操作符，对矩阵的每一个元素除以一个值
	private Operator divide_batchSize;

	// 乘数操作符，对矩阵的每一个元素乘以alpha值
	private Operator multiply_alpha;

	// 乘数操作符，对矩阵的每一个元素乘以1-labmda*alpha值
	private Operator multiply_lambda;

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
	public CNN(LayerBuilder layerBuilder, final int batchSize) {
		layers = layerBuilder.mLayers;
		layerNum = layers.size();
		this.batchSize = batchSize;
		setup(batchSize);
		initPerator();
	}

	/**
	 * 初始化操作符
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
	 * 在训练集上训练网络
	 * 
	 * @param trainset
	 * @param repeat
	 *            迭代的次数
	 */
	public void train(Dataset trainset, int repeat) {
		// 监听停止按钮
		new Lisenter().start();
		for (int t = 0; t < repeat && !stopTrain.get(); t++) {
			int epochsNum = trainset.size() / batchSize;
			if (trainset.size() % batchSize != 0)
				epochsNum++;// 多抽取一次，即向上取整
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

				// 跑完一个batch后更新权重
				updateParas();
				if (i % 50 == 0) {
					System.out.print("..");
					if (i + 50 > epochsNum)
						System.out.println();
				}
			}
			double p = 1.0 * right / count;
			if (t % 10 == 1 && p > 0.96) {//动态调整准学习速率
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
	 * 测试数据
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
			if (record.getLable().intValue() == Util.getMaxIndex(out))
				right++;		
		}
		double p = 1.0 * right / trainset.size();
		Log.i("precision", p + "");
		return p;
	}

	/**
	 * 预测结果
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
				// int lable =
				// Util.binaryArray2int(out);
				int lable = Util.getMaxIndex(out);
				// if (lable >= max)
				// lable = lable - (1 << (out.length -
				// 1));
				writer.write(lable + "\n");
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
	 * 训练一条记录，同时返回是否预测正确当前记录
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
	 * 反向传输
	 */
	private boolean backPropagation(Record record) {
		boolean result = setOutLayerErrors(record);
		setHiddenLayerErrors();
		return result;
	}

	/**
	 * 更新参数
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
	 * 更新偏置
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
					// 更新偏置
					double deltaBias = Util.sum(error) / batchSize;
					double bias = layer.getBias(j) + ALPHA * deltaBias;
					layer.setBias(j, bias);
				}
			}
		}.start();

	}

	/**
	 * 更新layer层的卷积核（权重）和偏置
	 * 
	 * @param layer
	 *            当前层
	 * @param lastLayer
	 *            前一层
	 */
	private void updateKernels(final Layer layer, final Layer lastLayer) {
		int mapNum = layer.getOutMapNum();
		final int lastMapNum = lastLayer.getOutMapNum();
		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					for (int i = 0; i < lastMapNum; i++) {
						// 对batch的每个记录delta求和
						double[][] deltaKernel = null;
						for (int r = 0; r < batchSize; r++) {
							double[][] error = layer.getError(r, j);
							if (deltaKernel == null)
								deltaKernel = Util.convnValid(
										lastLayer.getMap(r, i), error);
							else {// 累积求和
								deltaKernel = Util.matrixOp(Util.convnValid(
										lastLayer.getMap(r, i), error),
										deltaKernel, null, null, Util.plus);
							}
						}

						// 除以batchSize
						deltaKernel = Util.matrixOp(deltaKernel,
								divide_batchSize);
						// 更新卷积核
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
	 * 设置中将各层的残差
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
			default:// 只有采样层和卷积层需要处理残差，输入层没有残差，输出层已经处理过
				break;
			}
		}
	}

	/**
	 * 设置采样层的残差
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
					double[][] sum = null;// 对每一个卷积进行求和
					for (int j = 0; j < nextMapNum; j++) {
						double[][] nextError = nextLayer.getError(j);
						double[][] kernel = nextLayer.getKernel(i, j);
						// 对卷积核进行180度旋转，然后进行full模式下得卷积
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
	 * 设置卷积层的残差
	 * 
	 * @param layer
	 * @param nextLayer
	 */
	private void setConvErrors(final Layer layer, final Layer nextLayer) {
		// 卷积层的下一层为采样层，即两层的map个数相同，且一个map只与令一层的一个map连接，
		// 因此只需将下一层的残差kronecker扩展再用点积即可
		int mapNum = layer.getOutMapNum();
		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int m = start; m < end; m++) {
					Size scale = nextLayer.getScaleSize();
					double[][] nextError = nextLayer.getError(m);
					double[][] map = layer.getMap(m);
					// 矩阵相乘，但对第二个矩阵的每个元素value进行1-value操作
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
	 * 设置输出层的残差值,输出层的神经单元个数较少，暂不考虑多线程
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
		// // 正确
		// if (isSame(outmaps, target))
		// return true;
		// return false;

		double[] target = new double[mapNum];
		double[] outmaps = new double[mapNum];
		for (int m = 0; m < mapNum; m++) {
			double[][] outmap = outputLayer.getMap(m);
			outmaps[m] = outmap[0][0];

		}
		int lable = record.getLable().intValue();
		target[lable] = 1;
		// Log.i(record.getLable() + "outmaps:" +
		// Util.fomart(outmaps)
		// + Arrays.toString(target));
		for (int m = 0; m < mapNum; m++) {
			outputLayer.setError(m, 0, 0, outmaps[m] * (1 - outmaps[m])
					* (target[m] - outmaps[m]));
		}
		return lable == Util.getMaxIndex(outmaps);
	}

	/**
	 * 前向计算一条记录
	 * 
	 * @param record
	 */
	private void forward(Record record) {
		// 设置输入层的map
		setInLayerOutput(record);
		for (int l = 1; l < layers.size(); l++) {
			Layer layer = layers.get(l);
			Layer lastLayer = layers.get(l - 1);
			switch (layer.getType()) {
			case conv:// 计算卷积层的输出
				setConvOutput(layer, lastLayer);
				break;
			case samp:// 计算采样层的输出
				setSampOutput(layer, lastLayer);
				break;
			case output:// 计算输出层的输出,输出层是一个特殊的卷积层
				setConvOutput(layer, lastLayer);
				break;
			default:
				break;
			}
		}
	}

	/**
	 * 根据记录值，设置输入层的输出值
	 * 
	 * @param record
	 */
	private void setInLayerOutput(Record record) {
		final Layer inputLayer = layers.get(0);
		final Size mapSize = inputLayer.getMapSize();
		final double[] attr = record.getAttrs();
		if (attr.length != mapSize.x * mapSize.y)
			throw new RuntimeException("数据记录的大小与定义的map大小不一致!");
		for (int i = 0; i < mapSize.x; i++) {
			for (int j = 0; j < mapSize.y; j++) {
				// 将记录属性的一维向量弄成二维矩阵
				inputLayer.setMapValue(0, i, j, attr[mapSize.x * i + j]);
			}
		}
	}

	/*
	 * 计算卷积层输出值,每个线程负责一部分map
	 */
	private void setConvOutput(final Layer layer, final Layer lastLayer) {
		int mapNum = layer.getOutMapNum();
		final int lastMapNum = lastLayer.getOutMapNum();
		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					double[][] sum = null;// 对每一个输入map的卷积进行求和
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
	 * 设置采样层的输出值，采样层是对卷积层的均值处理
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
					// 按scaleSize区域进行均值处理
					double[][] sampMatrix = Util
							.scaleMatrix(lastMap, scaleSize);
					layer.setMapValue(i, sampMatrix);
				}
			}

		}.start();

	}

	/**
	 * 设置cnn网络的每一层的参数
	 * 
	 * @param batchSize
	 *            * @param classNum
	 * @param inputMapSize
	 */
	public void setup(int batchSize) {
		Layer inputLayer = layers.get(0);
		// 每一层都需要初始化输出map
		inputLayer.initOutmaps(batchSize);
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

				layer.initKernel(frontMapNum);
				// 初始化偏置，共有frontMapNum*outMapNum个偏置
				layer.initBias(frontMapNum);
				// batch的每个记录都要保持一份残差
				layer.initErros(batchSize);
				// 每一层都需要初始化输出map
				layer.initOutmaps(batchSize);
				break;
			case samp:
				// 采样层的map数量与上一层相同
				layer.setOutMapNum(frontMapNum);
				// 采样层map的大小是上一层map的大小除以scale大小
				layer.setMapSize(frontLayer.getMapSize().divide(
						layer.getScaleSize()));
				// batch的每个记录都要保持一份残差
				layer.initErros(batchSize);
				// 每一层都需要初始化输出map
				layer.initOutmaps(batchSize);
				break;
			case output:
				// 初始化权重（卷积核），输出层的卷积核大小为上一层的map大小
				layer.initOutputKerkel(frontMapNum, frontLayer.getMapSize());
				// 初始化偏置，共有frontMapNum*outMapNum个偏置
				layer.initBias(frontMapNum);
				// batch的每个记录都要保持一份残差
				layer.initErros(batchSize);
				// 每一层都需要初始化输出map
				layer.initOutmaps(batchSize);
				break;
			}
		}
	}

	/**
	 * 构造者模式构造各层,要求倒数第二层必须为采样层而不能为卷积层
	 * 
	 * @author jiqunpeng
	 * 
	 *         创建时间：2014-7-8 下午4:54:29
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
	 * 序列化保存模型
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
	 * 反序列化导入模型
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
