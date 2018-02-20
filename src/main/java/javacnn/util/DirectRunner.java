package javacnn.util;

import javacnn.cnn.Process;

/**
 * <p/>
 * Created: 20.02.2018 11:03
 *
 * @author Ralf Th. Pietsch &lt;ratopi@abwesend.de&gt;
 */
public class DirectRunner implements Runner {

	@Override
	public void startProcess(final int mapNum, final Process process) {
		final int runCpu = 1;

		// Fragment length rounded up
		final int fregLength = (mapNum + runCpu - 1) / runCpu;

		for (int cpu = 0; cpu < runCpu; cpu++) {
			final int start = cpu * fregLength;

			final int tmp = (cpu + 1) * fregLength;
			final int end = tmp <= mapNum ? tmp : mapNum;

			process.process(start, end);
		}
	}
}
