package javacnn.util;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javacnn.cnn.Process;

/**
 * Concurrent running tools
 * <p>
 * Created: 2014-6-16 at 3:33:41 PM
 *
 * @author jiqunpeng
 */
public class ConcurenceRunner implements Runner {

	private final ExecutorService exec;
	private final int cpuNum;

	public ConcurenceRunner() {
		cpuNum = Runtime.getRuntime().availableProcessors();
		System.out.println("cpuNum:" + cpuNum);
		exec = Executors.newFixedThreadPool(cpuNum);
	}

	public void stop() {
		exec.shutdown();
	}

	@Override
	public void startProcess(final int mapNum, final Process process) {
		new TaskManager(mapNum).start(process);
	}

	private void run(Runnable task) {
		exec.execute(task);
	}


	private class TaskManager {
		private int workLength;

		private TaskManager(int workLength) {
			this.workLength = workLength;
		}

		private void start(final Process processor) {
			int runCpu = cpuNum < workLength ? cpuNum : 1;

			// Fragment length rounded up
			final CountDownLatch gate = new CountDownLatch(runCpu);

			final int fregLength = (workLength + runCpu - 1) / runCpu;

			for (int cpu = 0; cpu < runCpu; cpu++) {
				final int start = cpu * fregLength;

				final int tmp = (cpu + 1) * fregLength;
				final int end = tmp <= workLength ? tmp : workLength;

				final Runnable task = new Runnable() {
					@Override
					public void run() {
						processor.process(start, end);
						gate.countDown();
					}
				};

				ConcurenceRunner.this.run(task);
			}
			try {// Wait for all threads to finish running
				gate.await();
			} catch (InterruptedException e) {
				e.printStackTrace();
				throw new RuntimeException(e);
			}
		}

	}

}
