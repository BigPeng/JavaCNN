package edu.hitsz.c102c.util;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import edu.hitsz.c102c.cnn.Layer.Size;
import edu.hitsz.c102c.util.TimedTest.TestTask;

public class Util {

	/**
	 * 矩阵对应元素相乘时在每个元素上的操作
	 * 
	 * @author jiqunpeng
	 * 
	 *         创建时间：2014-7-9 下午9:28:35
	 */
	public interface Operator extends Serializable {
		public double process(double value);
	}

	// 定义每个元素value都进行1-value的操作
	public static final Operator one_value = new Operator() {
		/**
		 * 
		 */
		private static final long serialVersionUID = 3752139491940330714L;

		@Override
		public double process(double value) {
			return 1 - value;
		}
	};

	// digmod函数
	public static final Operator digmod = new Operator() {
		/**
		 * 
		 */
		private static final long serialVersionUID = -1952718905019847589L;

		@Override
		public double process(double value) {
			return 1 / (1 + Math.pow(Math.E, -value));
		}
	};

	interface OperatorOnTwo extends Serializable {
		public double process(double a, double b);
	}

	/**
	 * 定义矩阵对应元素的加法操作
	 */
	public static final OperatorOnTwo plus = new OperatorOnTwo() {
		/**
		 * 
		 */
		private static final long serialVersionUID = -6298144029766839945L;

		@Override
		public double process(double a, double b) {
			return a + b;
		}
	};
	/**
	 * 定义矩阵对应元素的乘法操作
	 */
	public static OperatorOnTwo multiply = new OperatorOnTwo() {
		/**
		 * 
		 */
		private static final long serialVersionUID = -7053767821858820698L;

		@Override
		public double process(double a, double b) {
			return a * b;
		}
	};

	/**
	 * 定义矩阵对应元素的减法操作
	 */
	public static OperatorOnTwo minus = new OperatorOnTwo() {
		/**
		 * 
		 */
		private static final long serialVersionUID = 7346065545555093912L;

		@Override
		public double process(double a, double b) {
			return a - b;
		}
	};

	public static void printMatrix(double[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			String line = Arrays.toString(matrix[i]);
			line = line.replaceAll(", ", "\t");
			System.out.println(line);
		}
		System.out.println();
	}

	/**
	 * 对矩阵进行180度旋转,是在matrix的副本上复制，不会对原来的矩阵进行修改
	 * 
	 * @param matrix
	 */
	public static double[][] rot180(double[][] matrix) {
		matrix = cloneMatrix(matrix);
		int m = matrix.length;
		int n = matrix[0].length;
		// 按列对称进行交换
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n / 2; j++) {
				double tmp = matrix[i][j];
				matrix[i][j] = matrix[i][n - 1 - j];
				matrix[i][n - 1 - j] = tmp;
			}
		}
		// 按行对称进行交换
		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m / 2; i++) {
				double tmp = matrix[i][j];
				matrix[i][j] = matrix[m - 1 - i][j];
				matrix[m - 1 - i][j] = tmp;
			}
		}
		return matrix;
	}

	private static Random r = new Random(2);

	/**
	 * 随机初始化矩阵
	 * 
	 * @param x
	 * @param y
	 * @param b
	 * @return
	 */
	public static double[][] randomMatrix(int x, int y, boolean b) {
		double[][] matrix = new double[x][y];
		int tag = 1;
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				// 随机值在[-0.05,0.05)之间，让权重初始化值较小，有利于于避免过拟合
				matrix[i][j] = (r.nextDouble() - 0.05) / 10;
//				matrix[i][j] = tag * 0.5;
//				if (b)
//					matrix[i][j] *= 1.0*(i + j + 2) / (i + 1) / (j + 1);
//				tag *= -1;
			}
		}
		// printMatrix(matrix);
		return matrix;
	}

	/**
	 * 随机初始化一维向量
	 * 
	 * @param len
	 * @return
	 */
	public static double[] randomArray(int len) {
		double[] data = new double[len];
		for (int i = 0; i < len; i++) {
			// data[i] = r.nextDouble() / 10 - 0.05;
			data[i] = 0;
		}
		return data;
	}

	/**
	 * 随机排列的抽样，随机抽取batchSize个[0,size)的书
	 * 
	 * @param size
	 * @param batchSize
	 * @return
	 */
	public static int[] randomPerm(int size, int batchSize) {
		Set<Integer> set = new HashSet<Integer>();
		while (set.size() < batchSize) {
			set.add(r.nextInt(size));
		}
		int[] randPerm = new int[batchSize];
		int i = 0;
		for (Integer value : set)
			randPerm[i++] = value;
		return randPerm;
	}

	/**
	 * 复制矩阵
	 * 
	 * @param matrix
	 * @return
	 */
	public static double[][] cloneMatrix(final double[][] matrix) {

		final int m = matrix.length;
		int n = matrix[0].length;
		final double[][] outMatrix = new double[m][n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				outMatrix[i][j] = matrix[i][j];
			}
		}
		return outMatrix;
	}

	/**
	 * 对单个矩阵进行操作
	 * 
	 * @param ma
	 * @param operator
	 * @return
	 */
	public static double[][] matrixOp(final double[][] ma, Operator operator) {
		final int m = ma.length;
		int n = ma[0].length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				ma[i][j] = operator.process(ma[i][j]);
			}
		}
		return ma;

	}

	/**
	 * 两个维度相同的矩阵对应元素操作,得到的结果方法mb中，即mb[i][j] = (op_a
	 * ma[i][j]) op (op_b mb[i][j])
	 * 
	 * @param ma
	 * @param mb
	 * @param operatorB
	 *            在第mb矩阵上的操作
	 * @param operatorA
	 *            在ma矩阵元素上的操作
	 * @return
	 * 
	 */
	public static double[][] matrixOp(final double[][] ma, final double[][] mb,
			final Operator operatorA, final Operator operatorB,
			OperatorOnTwo operator) {
		final int m = ma.length;
		int n = ma[0].length;
		if (m != mb.length || n != mb[0].length)
			throw new RuntimeException("两个矩阵大小不一致 ma.length:" + ma.length
					+ "  mb.length:" + mb.length);

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				double a = ma[i][j];
				if (operatorA != null)
					a = operatorA.process(a);
				double b = mb[i][j];
				if (operatorB != null)
					b = operatorB.process(b);
				mb[i][j] = operator.process(a, b);
			}
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
	public static double[][] kronecker(final double[][] matrix, final Size scale) {
		final int m = matrix.length;
		int n = matrix[0].length;
		final double[][] outMatrix = new double[m * scale.x][n * scale.y];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				for (int ki = i * scale.x; ki < (i + 1) * scale.x; ki++) {
					for (int kj = j * scale.y; kj < (j + 1) * scale.y; kj++) {
						outMatrix[ki][kj] = matrix[i][j];
					}
				}
			}
		}
		return outMatrix;
	}

	/**
	 * 对矩阵进行均值缩小
	 * 
	 * @param matrix
	 * @param scaleSize
	 * @return
	 */
	public static double[][] scaleMatrix(final double[][] matrix,
			final Size scale) {
		int m = matrix.length;
		int n = matrix[0].length;
		final int sm = m / scale.x;
		final int sn = n / scale.y;
		final double[][] outMatrix = new double[sm][sn];
		if (sm * scale.x != m || sn * scale.y != n)
			throw new RuntimeException("scale不能整除matrix");
		final int size = scale.x * scale.y;
		for (int i = 0; i < sm; i++) {
			for (int j = 0; j < sn; j++) {
				double sum = 0.0;
				for (int si = i * scale.x; si < (i + 1) * scale.x; si++) {
					for (int sj = j * scale.y; sj < (j + 1) * scale.y; sj++) {
						sum += matrix[si][sj];
					}
				}
				outMatrix[i][j] = sum / size;
			}
		}
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
	 * 计算valid模式的卷积
	 * 
	 * @param matrix
	 * @param kernel
	 * @return
	 */
	public static double[][] convnValid(final double[][] matrix,
			double[][] kernel) {
		//kernel = rot180(kernel);
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

		for (int i = 0; i < kms; i++) {
			for (int j = 0; j < kns; j++) {
				double sum = 0.0;
				for (int ki = 0; ki < km; ki++) {
					for (int kj = 0; kj < kn; kj++)
						sum += matrix[i + ki][j + kj] * kernel[ki][kj];
				}
				outMatrix[i][j] = sum;

			}
		}
		return outMatrix;

	}

	/**
	 * 三维矩阵的卷积,这里要求两个矩阵的一维相同
	 * 
	 * @param matrix
	 * @param kernel
	 * @return
	 */
	public static double[][] convnValid(final double[][][][] matrix,
			int mapNoX, double[][][][] kernel, int mapNoY) {
		int m = matrix.length;
		int n = matrix[0][mapNoX].length;
		int h = matrix[0][mapNoX][0].length;
		int km = kernel.length;
		int kn = kernel[0][mapNoY].length;
		int kh = kernel[0][mapNoY][0].length;
		int kms = m - km + 1;
		int kns = n - kn + 1;
		int khs = h - kh + 1;
		if (matrix.length != kernel.length)
			throw new RuntimeException("矩阵与卷积核在第一维上不同");
		// 结果矩阵
		final double[][][] outMatrix = new double[kms][kns][khs];
		for (int i = 0; i < kms; i++) {
			for (int j = 0; j < kns; j++)
				for (int k = 0; k < khs; k++) {
					double sum = 0.0;
					for (int ki = 0; ki < km; ki++) {
						for (int kj = 0; kj < kn; kj++)
							for (int kk = 0; kk < kh; kk++) {
								sum += matrix[i + ki][mapNoX][j + kj][k + kk]
										* kernel[ki][mapNoY][kj][kk];
							}
					}
					outMatrix[i][j][k] = sum;
				}
		}
		return outMatrix[0];
	}

	public static double sigmod(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}

	/**
	 * 对矩阵元素求和
	 * 
	 * @param error
	 * @return 注意这个求和很可能会溢出
	 */

	public static double sum(double[][] error) {
		int m = error.length;
		int n = error[0].length;
		double sum = 0.0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sum += error[i][j];
			}
		}
		return sum;
	}

	/**
	 * 对errors[...][j]元素求和
	 * 
	 * @param errors
	 * @param j
	 * @return
	 */
	public static double[][] sum(double[][][][] errors, int j) {
		int m = errors[0][j].length;
		int n = errors[0][j][0].length;
		double[][] result = new double[m][n];
		for (int mi = 0; mi < m; mi++) {
			for (int nj = 0; nj < n; nj++) {
				double sum = 0;
				for (int i = 0; i < errors.length; i++)
					sum += errors[i][j][mi][nj];
				result[mi][nj] = sum;
			}
		}
		return result;
	}

	public static int binaryArray2int(double[] array) {
		int[] d = new int[array.length];
		for (int i = 0; i < d.length; i++) {
			if (array[i] >= 0.500000001)
				d[i] = 1;
			else
				d[i] = 0;
		}
		String s = Arrays.toString(d);
		String binary = s.substring(1, s.length() - 1).replace(", ", "");
		int data = Integer.parseInt(binary, 2);
		return data;

	}

	/**
	 * 测试卷积,测试结果：4核下并发并行的卷积提高不到2倍
	 */
	private static void testConvn() {
		int count = 1;
		double[][] m = new double[5][5];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = count++;
		double[][] k = new double[3][3];
		for (int i = 0; i < k.length; i++)
			for (int j = 0; j < k[0].length; j++)
				k[i][j] = 1;
		double[][] out;
		// out= convnValid(m, k);
		Util.printMatrix(m);
		out = convnFull(m, k);
		Util.printMatrix(out);
		// System.out.println();
		// out = convnFull(m, Util.rot180(k));
		// Util.printMatrix(out);

	}

	private static void testScaleMatrix() {
		int count = 1;
		double[][] m = new double[16][16];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = count++;
		double[][] out = scaleMatrix(m, new Size(2, 2));
		Util.printMatrix(m);
		Util.printMatrix(out);
	}

	private static void testKronecker() {
		int count = 1;
		double[][] m = new double[5][5];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = count++;
		double[][] out = kronecker(m, new Size(2, 2));
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
		double[][] out = matrixOp(m, k, new Operator() {

			/**
			 * 
			 */
			private static final long serialVersionUID = -680712567166604573L;

			@Override
			public double process(double value) {
				return value - 1;
			}
		}, new Operator() {

			/**
			 * 
			 */
			private static final long serialVersionUID = -6335660830579545544L;

			@Override
			public double process(double value) {

				return -1 * value;
			}
		}, multiply);
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

	public static void testRot180() {
		double[][] matrix = { { 1, 2, 3, 4 }, { 4, 5, 6, 7 }, { 7, 8, 9, 10 } };
		printMatrix(matrix);
		rot180(matrix);
		System.out.println();
		printMatrix(matrix);
	}

	public static void main(String[] args) {
		// new TimedTest(new TestTask() {
		//
		// @Override
		// public void process() {
		// testConvn();
		// // testScaleMatrix();
		// // testKronecker();
		// // testMatrixProduct();
		// // testCloneMatrix();
		// }
		// }, 1).test();
		// ConcurenceRunner.stop();
		System.out.println(sigmod(0.727855957917715));
		Double a = 1.0;
		int b = 1;
		System.out.println(a.equals(b));
	}

	/**
	 * 取最大的元素的下标
	 * 
	 * @param out
	 * @return
	 */
	public static int getMaxIndex(double[] out) {
		double max = out[0];
		int index = 0;
		for (int i = 1; i < out.length; i++)
			if (out[i] > max) {
				max = out[i];
				index = i;
			}
		return index;
	}

	public static String fomart(double[] data) {
		StringBuilder sb = new StringBuilder("[");
		for (double each : data)
			sb.append(String.format("%4f,", each));
		sb.append("]");
		return sb.toString();
	}

}
