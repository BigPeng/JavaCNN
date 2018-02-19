package javacnn.util;

import javacnn.cnn.Process;

/**
 * <p/>
 * Created: 2018-02-19 08:57
 *
 * @author Ralf Th. Pietsch &lt;ratopi@abwesend.de&gt;
 */
public interface Runner {
	void startProcess(int mapNum, Process process);
}
