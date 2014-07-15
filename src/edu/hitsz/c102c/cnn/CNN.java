package edu.hitsz.c102c.cnn;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CountDownLatch;

import edu.hitsz.c102c.cnn.Layer.Size;
import edu.hitsz.c102c.data.Dataset;
import edu.hitsz.c102c.data.Dataset.Record;
import edu.hitsz.c102c.util.ConcurenceRunner;
import edu.hitsz.c102c.util.ConcurenceRunner.Task;
import edu.hitsz.c102c.util.Log;
import edu.hitsz.c102c.util.Util;
import edu.hitsz.c102c.util.Util.Operator;

public class CNN {
	private static final double ALPHA = 1;
	protected static final double LAMBDA = 0;
	// 网络的各层
	private List<Layer> layers;
	// 层数
	private int layerNum;
	// 并行工具
	private static ConcurenceRunner runner = new ConcurenceRunner();
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

			@Override
			public double process(double value) {
				return value / batchSize;
			}

		};
		multiply_alpha = new Operator() {

			@Override
			public double process(double value) {

				return value * ALPHA;
			}

		};
		multiply_lambda = new Operator() {

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
		for (int t = 0; t < repeat; t++) {
			int epochsNum = 1 + trainset.size() / batchSize;// 多抽取一次，即向上取整
			for (int i = 0; i < epochsNum; i++) {
				int[] randPerm = Util.randomPerm(trainset.size(), batchSize);				
				Layer.prepareForNewBatch();
				for (int index : randPerm) {
					train(trainset.getRecord(index));
					Layer.prepareForNewRecord();
				}
				// if (0 == 0)
				// return;
				// 跑完一个batch后更新权重
				updateParas();
				if (i % 50 == 0)
					Log.i("epochsNum " + epochsNum + ":" + i);
			}
			Log.i("begin test");
			Layer.prepareForNewBatch();
			double precision = test(trainset);
			Log.i("precision " + precision);
		}
	}

	/**
	 * 测试数据
	 * 
	 * @param trainset
	 * @return
	 */
	private double test(Dataset trainset) {
		Iterator<Record> iter = trainset.iter();
		int right = 0;
		int count = 0;
		while (iter.hasNext()) {
			Record record = iter.next();
			forward(record);
			Layer outputLayer = layers.get(layerNum - 1);
			int mapNum = outputLayer.getOutMapNum();
			double[] target = record.getDoubleEncodeTarget(mapNum);
			double[] out = new double[mapNum];
			for (int m = 0; m < mapNum; m++) {
				double[][] outmap = outputLayer.getMap(m);
				out[m] = outmap[0][0];
			}
			// if (record.getLable().intValue() ==
			// Util.getMaxIndex(out))
			// right++;
			if (isSame(out, target)) {
				right++;
				// if (right % 1000 == 0)
				// Log.i("out:" + Arrays.toString(out)
				// + " \n target:"
				// + Arrays.toString(target));
			}

			if (count++ % 1000 == 0)
				Log.i("out:" + Arrays.toString(out) + " \n target:"
						+ Arrays.toString(target));
		}
		return 1.0 * right / trainset.size();
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
			int max = Layer.getClassNum();
			PrintWriter writer = new PrintWriter(new File(fileName));
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
				int lable = Util.binaryArray2int(out);
				if (lable > max)
					lable = lable - (1 << (out.length - 1));
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

	private void train(Record record) {
		forward(record);
		backPropagation(record);
		// System.exit(0);
	}

	/*
	 * 反向传输
	 */
	private void backPropagation(Record record) {
		setOutLayerErrors(record);
		setHiddenLayerErrors();
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
	private void updateBias(Layer layer, Layer lastLayer) {
		double[][][][] errors = layer.getErrors();
		int mapNum = layer.getOutMapNum();
		for (int j = 0; j < mapNum; j++) {
			double[][] error = Util.sum(errors, j);
			// 更新偏置
			double deltaBias = Util.sum(error) / batchSize;
			double bias = layer.getBias(j) + ALPHA * deltaBias;
			layer.setBias(j, bias);
		}
	}

	/**
	 * 更新layer层的卷积核（权重）和偏置
	 * 
	 * @param layer
	 *            当前层
	 * @param lastLayer
	 *            前一层
	 */
	private void updateKernels(Layer layer, Layer lastLayer) {
		int mapNum = layer.getOutMapNum();
		int lastMapNum = lastLayer.getOutMapNum();
		// double[][][][] errors = layer.getErrors();
		// double[][][][] lastMaps =
		// lastLayer.getMaps();
		for (int j = 0; j < mapNum; j++) {
			for (int i = 0; i < lastMapNum; i++) {
				// double[][] deltaKernel = Util
				// .convnValid(lastMaps, i, errors, j);
				// 对batch的每个记录delta求和
				double[][] deltaKernel = null;
				for (int r = 0; r < batchSize; r++) {
					double[][] error = layer.getError(r, j);
					if (deltaKernel == null)
						deltaKernel = Util.convnValid(lastLayer.getMap(r, i),
								error);
					else {// 累积求和
						deltaKernel = Util.matrixOp(
								Util.convnValid(lastLayer.getMap(r, i), error),
								deltaKernel, null, null, Util.plus);
					}
				}

				// 除以batchSize
				deltaKernel = Util.matrixOp(deltaKernel, divide_batchSize);
				// 更新卷积核
				double[][] kernel = layer.getKernel(i, j);
				deltaKernel = Util.matrixOp(kernel, deltaKernel,
						multiply_lambda, multiply_alpha, Util.plus);
				layer.setKernel(i, j, deltaKernel);
			}
		}
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
	private void setSampErrors(Layer layer, Layer nextLayer) {
		int mapNum = layer.getOutMapNum();
		final int nextMapNum = nextLayer.getOutMapNum();
		for (int i = 0; i < mapNum; i++) {
			double[][] sum = null;// 对每一个卷积进行求和
			for (int j = 0; j < nextMapNum; j++) {
				double[][] nextError = nextLayer.getError(j);
				double[][] kernel = nextLayer.getKernel(i, j);
				// 对卷积核进行180度旋转，然后进行full模式下得卷积
				if (sum == null)
					sum = Util.convnFull(nextError, Util.rot180(kernel));
				else
					sum = Util.matrixOp(
							Util.convnFull(nextError, Util.rot180(kernel)),
							sum, null, null, Util.plus);
			}
			layer.setError(i, sum);
		}
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
		for (int m = 0; m < mapNum; m++) {
			Size scale = nextLayer.getScaleSize();
			double[][] nextError = nextLayer.getError(m);
			double[][] map = layer.getMap(m);
			// 矩阵相乘，但对第二个矩阵的每个元素value进行1-value操作
			double[][] outMatrix = Util.matrixOp(map, Util.cloneMatrix(map),
					null, Util.one_value, Util.multiply);
			outMatrix = Util
					.matrixOp(outMatrix, Util.kronecker(nextError, scale),
							null, null, Util.multiply);
			layer.setError(m, outMatrix);
		}

	}

	/**
	 * 设置输出层的残差值,输出层的神经单元个数较少，暂不考虑多线程
	 * 
	 * @param record
	 */
	private void setOutLayerErrors(Record record) {
		
		Layer outputLayer = layers.get(layerNum - 1);
		int mapNum = outputLayer.getOutMapNum();
		 double[] target =
		 record.getDoubleEncodeTarget(mapNum);
		 for (int m = 0; m < mapNum; m++) {
		 double[][] outmap = outputLayer.getMap(m);
		 double output = outmap[0][0];
		 double errors = output * (1 - output) *
		 (target[m] - output);
		 outputLayer.setError(m, 0, 0, errors);
		 }
		 
//		double[] errors = new double[mapNum];
//		double[] outmaps = new double[mapNum];
//		for (int m = 0; m < mapNum; m++) {
//			double[][] outmap = outputLayer.getMap(m);
//			outmaps[m] = outmap[0][0];
//
//		}
//
//		errors[record.getLable().intValue()] = 1;
//		for (int m = 0; m < mapNum; m++) {
//			outputLayer.setError(m, 0, 0, outmaps[m] * (1 - outmaps[m])
//					* (errors[m] - outmaps[m]));
//		}
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
		for (int j = 0; j < mapNum; j++) {
			double[][] sum = null;// 对每一个输入map的卷积进行求和
			for (int i = 0; i < lastMapNum; i++) {
				double[][] lastMap = lastLayer.getMap(i);
				double[][] kernel = layer.getKernel(i, j);
				if (sum == null)
					sum = Util.convnValid(lastMap, kernel);
				else
					sum = Util.matrixOp(Util.convnValid(lastMap, kernel), sum,
							null, null, Util.plus);
			}
			final double bias = layer.getBias(j);
			sum = Util.matrixOp(sum, new Operator() {

				@Override
				public double process(double value) {
					return Util.sigmod(value + bias);
				}

			});
			if (sum[0][0] > 1)
				Log.i(sum[0][0] + "");
			layer.setMapValue(j, sum);
		}

	}

	/**
	 * 设置采样层的输出值，采样层是对卷积层的均值处理
	 * 
	 * @param layer
	 * @param lastLayer
	 */
	private void setSampOutput(final Layer layer, final Layer lastLayer) {
		int lastMapNum = lastLayer.getOutMapNum();
		for (int i = 0; i < lastMapNum; i++) {
			double[][] lastMap = lastLayer.getMap(i);
			Size scaleSize = layer.getScaleSize();
			// 按scaleSize区域进行均值处理
			double[][] sampMatrix = Util.scaleMatrix(lastMap, scaleSize);
			layer.setMapValue(i, sampMatrix);
		}
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
				layer.initKerkel(frontMapNum);
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
}
