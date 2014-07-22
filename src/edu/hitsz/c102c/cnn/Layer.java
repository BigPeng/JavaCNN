package edu.hitsz.c102c.cnn;

import java.io.Serializable;

import edu.hitsz.c102c.util.Log;
import edu.hitsz.c102c.util.Util;

/**
 * cnn网络的层
 * 
 * @author jiqunpeng
 * 
 *         创建时间：2014-7-8 下午3:58:46
 */
public class Layer implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -5747622503947497069L;
	private LayerType type;// 层的类型
	private int outMapNum;// 输出map的个数
	private Size mapSize;// map的大小
	private Size kernelSize;// 卷积核大小，只有卷积层有
	private Size scaleSize;// 采样大小，只有采样层有
	private double[][][][] kernel;// 卷积核，只有卷积层和输出层有
	private double[] bias;// 每个map对应一个偏置，只有卷积层和输出层有
	// 保存各个batch的输出map，outmaps[0][0]表示第一条记录训练下第0个输出map
	private double[][][][] outmaps;
	// 残差，与matlab toolbox的d对应
	private double[][][][] errors;

	private static int recordInBatch = 0;// 记录当前训练的是batch的第几条记录

	private int classNum = -1;// 类别个数

	private Layer() {

	}

	/**
	 * 准备下一个batch的训练
	 */
	public static void prepareForNewBatch() {
		recordInBatch = 0;
	}

	/**
	 * 准备下一条记录的训练
	 */
	public static void prepareForNewRecord() {
		recordInBatch++;
	}

	/**
	 * 初始化输入层
	 * 
	 * @param mapSize
	 * @return
	 */
	public static Layer buildInputLayer(Size mapSize) {
		Layer layer = new Layer();
		layer.type = LayerType.input;
		layer.outMapNum = 1;// 输入层的map个数为1，即一张图
		layer.setMapSize(mapSize);//
		return layer;
	}

	/**
	 * 构造卷积层
	 * 
	 * @return
	 */
	public static Layer buildConvLayer(int outMapNum, Size kernelSize) {
		Layer layer = new Layer();
		layer.type = LayerType.conv;
		layer.outMapNum = outMapNum;
		layer.kernelSize = kernelSize;
		return layer;
	}

	/**
	 * 构造采样层
	 * 
	 * @param scaleSize
	 * @return
	 */
	public static Layer buildSampLayer(Size scaleSize) {
		Layer layer = new Layer();
		layer.type = LayerType.samp;
		layer.scaleSize = scaleSize;
		return layer;
	}

	/**
	 * 构造输出层,类别个数，根据类别的个数来决定输出单元的个数
	 * 
	 * @return
	 */
	public static Layer buildOutputLayer(int classNum) {
		Layer layer = new Layer();
		layer.classNum = classNum;
		layer.type = LayerType.output;
		layer.mapSize = new Size(1, 1);
		layer.outMapNum = classNum;
		// int outMapNum = 1;
		// while ((1 << outMapNum) < classNum)
		// outMapNum += 1;
		// layer.outMapNum = outMapNum;
		Log.i("outMapNum:" + layer.outMapNum);
		return layer;
	}

	/**
	 * 获取map的大小
	 * 
	 * @return
	 */
	public Size getMapSize() {
		return mapSize;
	}

	/**
	 * 设置map的大小
	 * 
	 * @param mapSize
	 */
	public void setMapSize(Size mapSize) {
		this.mapSize = mapSize;
	}

	/**
	 * 获取层的类型
	 * 
	 * @return
	 */
	public LayerType getType() {
		return type;
	}

	/**
	 * 获取输出向量个数
	 * 
	 * @return
	 */

	public int getOutMapNum() {
		return outMapNum;
	}

	/**
	 * 设置输出map的个数
	 * 
	 * @param outMapNum
	 */
	public void setOutMapNum(int outMapNum) {
		this.outMapNum = outMapNum;
	}

	/**
	 * 获取卷积核的大小，只有卷积层有kernelSize，其他层均未null
	 * 
	 * @return
	 */
	public Size getKernelSize() {
		return kernelSize;
	}

	/**
	 * 获取采样大小，只有采样层有scaleSize，其他层均未null
	 * 
	 * @return
	 */
	public Size getScaleSize() {
		return scaleSize;
	}

	enum LayerType {
		// 网络层的类型：输入层、输出层、卷积层、采样层
		input, output, conv, samp
	}

	/**
	 * 卷积核或者采样层scale的大小,长与宽可以不等.类型安全，定以后不可修改
	 * 
	 * @author jiqunpeng
	 * 
	 *         创建时间：2014-7-8 下午4:11:00
	 */
	public static class Size implements Serializable {

		private static final long serialVersionUID = -209157832162004118L;
		public final int x;
		public final int y;

		public Size(int x, int y) {
			this.x = x;
			this.y = y;
		}

		public String toString() {
			StringBuilder s = new StringBuilder("Size(").append(" x = ")
					.append(x).append(" y= ").append(y).append(")");
			return s.toString();
		}

		/**
		 * 整除scaleSize得到一个新的Size，要求this.x、this.
		 * y能分别被scaleSize.x、scaleSize.y整除
		 * 
		 * @param scaleSize
		 * @return
		 */
		public Size divide(Size scaleSize) {
			int x = this.x / scaleSize.x;
			int y = this.y / scaleSize.y;
			if (x * scaleSize.x != this.x || y * scaleSize.y != this.y)
				throw new RuntimeException(this + "不能整除" + scaleSize);
			return new Size(x, y);
		}

		/**
		 * 减去size大小，并x和y分别附加一个值append
		 * 
		 * @param size
		 * @param append
		 * @return
		 */
		public Size subtract(Size size, int append) {
			int x = this.x - size.x + append;
			int y = this.y - size.y + append;
			return new Size(x, y);
		}
	}

	/**
	 * 随机初始化卷积核
	 * 
	 * @param frontMapNum
	 */
	public void initKernel(int frontMapNum) {
//		int fan_out = getOutMapNum() * kernelSize.x * kernelSize.y;
//		int fan_in = frontMapNum * kernelSize.x * kernelSize.y;
//		double factor = 2 * Math.sqrt(6 / (fan_in + fan_out));
		this.kernel = new double[frontMapNum][outMapNum][kernelSize.x][kernelSize.y];
		for (int i = 0; i < frontMapNum; i++)
			for (int j = 0; j < outMapNum; j++)
				kernel[i][j] = Util.randomMatrix(kernelSize.x, kernelSize.y,true);
	}

	/**
	 * 输出层的卷积核的大小是上一层的map大小
	 * 
	 * @param frontMapNum
	 * @param size
	 */
	public void initOutputKerkel(int frontMapNum, Size size) {
		kernelSize = size;
//		int fan_out = getOutMapNum() * kernelSize.x * kernelSize.y;
//		int fan_in = frontMapNum * kernelSize.x * kernelSize.y;
//		double factor = 2 * Math.sqrt(6 / (fan_in + fan_out));
		this.kernel = new double[frontMapNum][outMapNum][kernelSize.x][kernelSize.y];
		for (int i = 0; i < frontMapNum; i++)
			for (int j = 0; j < outMapNum; j++)
				kernel[i][j] = Util.randomMatrix(kernelSize.x, kernelSize.y,false);
	}

	/**
	 * 初始化偏置
	 * 
	 * @param frontMapNum
	 */
	public void initBias(int frontMapNum) {
		this.bias = Util.randomArray(outMapNum);
	}

	/**
	 * 初始化输出map
	 * 
	 * @param batchSize
	 */
	public void initOutmaps(int batchSize) {
		outmaps = new double[batchSize][outMapNum][mapSize.x][mapSize.y];
	}

	/**
	 * 设置map值
	 * 
	 * @param mapNo
	 *            第几个map
	 * @param mapX
	 *            map的高
	 * @param mapY
	 *            map的宽
	 * @param value
	 */
	public void setMapValue(int mapNo, int mapX, int mapY, double value) {
		outmaps[recordInBatch][mapNo][mapX][mapY] = value;
	}

	static int count = 0;

	/**
	 * 以矩阵形式设置第mapNo个map的值
	 * 
	 * @param mapNo
	 * @param outMatrix
	 */
	public void setMapValue(int mapNo, double[][] outMatrix) {
		// Log.i(type.toString());
		// Util.printMatrix(outMatrix);
		outmaps[recordInBatch][mapNo] = outMatrix;
	}

	/**
	 * 获取第index个map矩阵。处于性能考虑，没有返回复制对象，而是直接返回引用，调用端请谨慎，
	 * 避免修改outmaps，如需修改请调用setMapValue(...)
	 * 
	 * @param index
	 * @return
	 */
	public double[][] getMap(int index) {
		return outmaps[recordInBatch][index];
	}

	/**
	 * 获取前一层第i个map到当前层第j个map的卷积核
	 * 
	 * @param i
	 *            上一层的map下标
	 * @param j
	 *            当前层的map下标
	 * @return
	 */
	public double[][] getKernel(int i, int j) {
		return kernel[i][j];
	}

	/**
	 * 设置残差值
	 * 
	 * @param mapNo
	 * @param mapX
	 * @param mapY
	 * @param value
	 */
	public void setError(int mapNo, int mapX, int mapY, double value) {
		errors[recordInBatch][mapNo][mapX][mapY] = value;
	}

	/**
	 * 以map矩阵块形式设置残差值
	 * 
	 * @param mapNo
	 * @param matrix
	 */
	public void setError(int mapNo, double[][] matrix) {
		// Log.i(type.toString());
		// Util.printMatrix(matrix);
		errors[recordInBatch][mapNo] = matrix;
	}

	/**
	 * 获取第mapNo个map的残差.没有返回复制对象，而是直接返回引用，调用端请谨慎，
	 * 避免修改errors，如需修改请调用setError(...)
	 * 
	 * @param mapNo
	 * @return
	 */
	public double[][] getError(int mapNo) {
		return errors[recordInBatch][mapNo];
	}

	/**
	 * 获取所有(每个记录和每个map)的残差
	 * 
	 * @return
	 */
	public double[][][][] getErrors() {
		return errors;
	}

	/**
	 * 初始化残差数组
	 * 
	 * @param batchSize
	 */
	public void initErros(int batchSize) {
		errors = new double[batchSize][outMapNum][mapSize.x][mapSize.y];
	}

	/**
	 * 
	 * @param lastMapNo
	 * @param mapNo
	 * @param kernel
	 */
	public void setKernel(int lastMapNo, int mapNo, double[][] kernel) {
		this.kernel[lastMapNo][mapNo] = kernel;
	}

	/**
	 * 获取第mapNo个
	 * 
	 * @param mapNo
	 * @return
	 */
	public double getBias(int mapNo) {
		return bias[mapNo];
	}

	/**
	 * 设置第mapNo个map的偏置值
	 * 
	 * @param mapNo
	 * @param value
	 */
	public void setBias(int mapNo, double value) {
		bias[mapNo] = value;
	}

	/**
	 * 获取batch各个map矩阵
	 * 
	 * @return
	 */

	public double[][][][] getMaps() {
		return outmaps;
	}

	/**
	 * 获取第recordId记录下第mapNo的残差
	 * 
	 * @param recordId
	 * @param mapNo
	 * @return
	 */
	public double[][] getError(int recordId, int mapNo) {
		return errors[recordId][mapNo];
	}

	/**
	 * 获取第recordId记录下第mapNo的输出map
	 * 
	 * @param recordId
	 * @param mapNo
	 * @return
	 */
	public double[][] getMap(int recordId, int mapNo) {
		return outmaps[recordId][mapNo];
	}

	/**
	 * 获取类别个数
	 * 
	 * @return
	 */
	public int getClassNum() {
		return classNum;
	}

	/**
	 * 获取所有的卷积核
	 * 
	 * @return
	 */
	public double[][][][] getKernel() {
		return kernel;
	}

}
