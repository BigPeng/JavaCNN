package edu.hitsz.c102c.util;

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
		//cpuNum = 1;		
		System.out.println("cpuNum:" + cpuNum);
		exec = Executors.newFixedThreadPool(cpuNum);
	}

	public void run(Runnable task) {
		exec.execute(task);
	}

	public static void stop() {
		exec.shutdown();
	}

	public abstract static class Task implements Runnable {
		int start, end;

		public Task(int start, int end) {
			this.start = start;
			this.end = end;
			//Log.i("new Task", "start "+start+" end "+end);
		}

		@Override
		public void run() {
			process(start, end);
		}

		public abstract void process(int start, int end);

	}

}
