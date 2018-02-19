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
	private final int threadCount;

	/**
	 * Starting ConcurrenceRunner with one thread for each CPU.
	 */
	public ConcurenceRunner() {
		this(Runtime.getRuntime().availableProcessors());
	}

	/**
	 * Starting ConcurenceRunner with the given count of threads.
	 *
	 * @param threadCount Threads to start (must be &gt; 0).
	 */
	public ConcurenceRunner(final int threadCount) {
		this.threadCount = threadCount;
		exec = Executors.newFixedThreadPool(this.threadCount);
	}

	public void shutdown() {
		exec.shutdown();
	}

	@Override
	public void startProcess(final int mapNum, final Process process) {
		final int runCpu = threadCount < mapNum ? threadCount : 1;

		// Fragment length rounded up
		final CountDownLatch gate = new CountDownLatch(runCpu);

		final int fregLength = (mapNum + runCpu - 1) / runCpu;

		for (int cpu = 0; cpu < runCpu; cpu++) {
			final int start = cpu * fregLength;

			final int tmp = (cpu + 1) * fregLength;
			final int end = tmp <= mapNum ? tmp : mapNum;

			final Runnable task = new Runnable() {
				@Override
				public void run() {
					process.process(start, end);
					gate.countDown();
				}
			};

			exec.execute(task);
		}
		try {// Wait for all threads to finish running
			gate.await();
		} catch (InterruptedException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}

}
