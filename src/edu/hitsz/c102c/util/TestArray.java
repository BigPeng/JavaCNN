package edu.hitsz.c102c.util;

import java.util.Locale;

import edu.hitsz.c102c.util.TimedTest.TestTask;

/**
 * 测试元素直接访问数组与通过函数访问数组的效率， 结论：函数形式访问并没有降低速度
 * 
 * @author jiqunpeng
 * 
 *         创建时间：2014-7-9 下午3:18:30
 */
public class TestArray {
	double[][] data;

	public TestArray(int m, int n) {
		data = new double[m][n];
	}

	public void set(int x, int y, double value) {
		data[x][y] = value;
	}

	private void useOrigin() {
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				data[i][j] = i * j;
	}

	private void useFunc() {
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				set(i, j, i * j);
	}

	public static void main(String[] args) {
		String a = "aAdfa彭_";
		System.out.println(a.toUpperCase(Locale.CHINA));
		double[][] d = new double[3][];
//		d[0] = new double[] { 1,2,3 };
//		d[1] = new double[] { 3,4,5,6 };
		System.out.println(d[1][3]);
		final TestArray t = new TestArray(10000, 1000);
		new TimedTest(new TestTask() {

			@Override
			public void process() {
				t.useFunc();
			}
		}, 1).test();
	
		
	}
}
