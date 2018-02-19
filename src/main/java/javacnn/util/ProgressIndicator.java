package javacnn.util;

/**
 * Interface for feedback progress of any kind
 * <p/>
 * Created: 19.02.2018 10:52
 *
 * @author Ralf Th. Pietsch &lt;ratopi@abwesend.de&gt;
 */
public interface ProgressIndicator {
	void start();
	void progress();
	void finished();
}
