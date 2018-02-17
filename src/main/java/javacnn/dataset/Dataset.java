package javacnn.dataset;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class Dataset {
	private List<Record> records;

	public Dataset() {
		records = new ArrayList<>();
	}

	public Dataset(final List<double[]> datas, final List<Double> labels) {
		this();

		if (datas.size() != labels.size()) {
			throw new IllegalArgumentException("Lengths differs: " + datas.size() + " datas and " + labels.size() + " labels");
		}

		for (int i = 0; i < datas.size(); i++) {
			final double[] data = datas.get(i);
			final Double label = labels.get(i);
			append(new Record(data, label));
		}
	}

	public int size() {
		return records.size();
	}

	public void append(Record record) {
		records.add(record);
	}

	public void clear() {
		records.clear();
	}

	public void append(double[] attrs, Double label) {
		records.add(new Record(attrs, label));
	}

	public Iterator<Record> iter() {
		return records.iterator();
	}

	public double[] getAttrs(int index) {
		return records.get(index).getAttrs();
	}

	public Double getLabel(int index) {
		return records.get(index).getLabel();
	}

	public Record getRecord(int index) {
		return records.get(index);
	}

	// ---

	public static class Record {
		private double[] attrs;
		private Double label;

		private Record(double[] attrs, Double label) {
			this.attrs = attrs;
			this.label = label;
		}

		public double[] getAttrs() {
			return attrs;
		}

		public Double getLabel() {
			return label;
		}

		@Override
		public String toString() {
			return "Record{" +
					"attrs=" + Arrays.toString(attrs) +
					", label=" + label +
					'}';
		}
	}

}
