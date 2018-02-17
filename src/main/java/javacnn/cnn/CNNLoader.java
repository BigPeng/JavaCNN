package javacnn.cnn;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class CNNLoader {

	public static void saveModel(final String fileName, final CNN cnn) throws IOException {
		saveModel(cnn, new FileOutputStream(fileName));
		return;
	}

	public static void saveModel(final CNN cnn, final FileOutputStream fileOutputStream) throws IOException {
		ObjectOutputStream oos = new ObjectOutputStream(fileOutputStream);
		oos.writeObject(cnn);
		oos.flush();
		oos.close();
	}

	public static CNN loadModel(String fileName) throws IOException, ClassNotFoundException {
		return loadModel(new FileInputStream(fileName));
	}

	public static CNN loadModel(final FileInputStream fileInputStream) throws IOException, ClassNotFoundException {
		final ObjectInputStream in = new ObjectInputStream(fileInputStream);
		final CNN cnn = (CNN) in.readObject();
		in.close();

		return cnn;
	}

}
