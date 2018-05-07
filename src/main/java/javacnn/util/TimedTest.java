package javacnn.util;

/**
 * ��ʱ�Ĳ��Թ���
 * 
 * @author jiqunpeng
 * 
 *         ����ʱ�䣺2014-7-8 ����8:21:56
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
		Log.info("cost ", cost + "s");
	}
}
