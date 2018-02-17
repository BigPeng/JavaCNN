package javacnn.dataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class DatasetLoader {
	public static Dataset load(final String filePath, final String tag, final int labelIndex) throws IOException {
		final Dataset dataset = new Dataset();

		final BufferedReader in = new BufferedReader(new FileReader(filePath));

		String line;
		while ((line = in.readLine()) != null) {

			final String[] datas = line.split(tag);

			if (datas.length == 0) {
				continue;
			}

			final int vectorLength = labelIndex < 0 ? datas.length : datas.length - 1;

			final double[] data = new double[vectorLength];

			for (int i = 0; i < vectorLength; i++) {
				data[i] = Double.parseDouble(datas[i]);
			}

			final Double label = labelIndex < 0 ? null : Double.parseDouble(datas[labelIndex]);

			dataset.append(data, label);
		}
		in.close();

		System.out.println("Read " + dataset.size() + " records");

		return dataset;
	}

}
