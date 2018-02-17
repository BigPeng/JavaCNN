package javacnn.util;

import java.io.PrintStream;

public class Log {
	private static final PrintStream stream = System.out;

	public static void info(String tag, String msg) {
		stream.println(tag + "\t" + msg);
	}

	public static void info(String msg) {
		stream.println(msg);
	}

}
