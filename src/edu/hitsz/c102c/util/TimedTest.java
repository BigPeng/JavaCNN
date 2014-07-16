package edu.hitsz.c102c.util;

/**
 * 计时的测试工具
 * 
 * @author jiqunpeng
 * 
 *         创建时间：2014-7-8 下午8:21:56
 */
public class TimedTest {
	private int repeat;
	private TestTask task;

	public interface TestTask {
		public void process();
	}

	public TimedTest(TestTask t, int repeat) {
		this.repeat = repeat;
		task = t;
	}

	public void test() {
		long t = System.currentTimeMillis();
		for (int i = 0; i < repeat; i++) {
			task.process();
		}
		double cost = (System.currentTimeMillis() - t) / 1000.0;
		Log.i("cost ", cost + "s");
	}
}
