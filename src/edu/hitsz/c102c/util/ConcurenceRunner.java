package edu.hitsz.c102c.util;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 并发运行工具
 * 
 * @author jiqunpeng
 * 
 *         创建时间：2014-6-16 下午3:33:41
 */
public class ConcurenceRunner {

	private static final ExecutorService exec;
	public static final int cpuNum;
	static {
		cpuNum = Runtime.getRuntime().availableProcessors();
		// cpuNum = 1;
		System.out.println("cpuNum:" + cpuNum);
		exec = Executors.newFixedThreadPool(cpuNum);
	}

	public static void run(Runnable task) {
		exec.execute(task);
	}

	public static void stop() {
		exec.shutdown();
	}

	// public abstract static class Task implements
	// Runnable {
	// int start, end;
	//
	// public Task(int start, int end) {
	// this.start = start;
	// this.end = end;
	// // Log.i("new Task",
	// // "start "+start+" end "+end);
	// }
	//
	// @Override
	// public void run() {
	// process(start, end);
	// }
	//
	// public abstract void process(int start, int
	// end);
	//
	// }

	public abstract static class TaskManager {
		private int workLength;

		public TaskManager(int workLength) {
			this.workLength = workLength;
		}

		public void start() {
			int runCpu = cpuNum < workLength ? cpuNum : 1;
			// 分片长度向上取整
			final CountDownLatch gate = new CountDownLatch(runCpu);
			int fregLength = (workLength + runCpu - 1) / runCpu;
			for (int cpu = 0; cpu < runCpu; cpu++) {
				final int start = cpu * fregLength;
				int tmp = (cpu + 1) * fregLength;
				final int end = tmp <= workLength ? tmp : workLength;
				Runnable task = new Runnable() {

					@Override
					public void run() {
						process(start, end);
						gate.countDown();
					}

				};
				ConcurenceRunner.run(task);
			}
			try {// 等待所有线程跑完
				gate.await();
			} catch (InterruptedException e) {
				e.printStackTrace();
				throw new RuntimeException(e);
			}
		}

		public abstract void process(int start, int end);

	}

}
