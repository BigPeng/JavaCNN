package javacnn.util;

import java.io.PrintStream;

public class Log {
	private static final PrintStream stream = System.out;

	private static boolean on = false;

	public static void switchOn() {
		on = true;
	}

	public static void info(String tag, String msg) {
		if (on) stream.println(tag + "\t" + msg);
	}

	public static void info(String msg) {
		if (on) stream.println(msg);
	}

}
