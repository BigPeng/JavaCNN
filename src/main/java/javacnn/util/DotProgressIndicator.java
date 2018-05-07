package javacnn.util;

import java.io.Serializable;

/**
 * <p/>
 * Created: 19.02.2018 10:53
 *
 * @author Ralf Th. Pietsch &lt;ratopi@abwesend.de&gt;
 */
public class DotProgressIndicator implements ProgressIndicator, Serializable {

	private static final long serialVersionUID = 1L;

	private int cycle;

	private int count = 0;


	public DotProgressIndicator() {
		this(50);
	}

	public DotProgressIndicator(final int cycle) {
		this.cycle = cycle;
	}


	@Override
	public void start() {
		count = 0;
	}

	@Override
	public void progress() {
		count++;
		if (count > cycle) {
			System.out.print(".");
			count = 0;
		}
	}

	@Override
	public void finished() {
		System.out.println();
	}
}
