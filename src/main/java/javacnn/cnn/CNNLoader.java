package javacnn.cnn;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

public class CNNLoader {

	public static void saveModel(final String fileName, final CNN cnn) throws IOException {
		saveModel(cnn, new FileOutputStream(fileName));
	}

	public static void saveModel(final CNN cnn, final OutputStream outputStream) throws IOException {
		ObjectOutputStream oos = new ObjectOutputStream(outputStream);
		oos.writeObject(cnn);
		oos.flush();
		oos.close();
	}

	public static CNN loadModel(final String fileName) throws IOException, ClassNotFoundException {
		return loadModel(new FileInputStream(fileName));
	}

	public static CNN loadModel(final InputStream inputStream) throws IOException, ClassNotFoundException {
		final ObjectInputStream in = new ObjectInputStream(inputStream);
		final CNN cnn = (CNN) in.readObject();
		in.close();

		return cnn;
	}

}
