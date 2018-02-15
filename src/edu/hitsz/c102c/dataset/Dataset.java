package edu.hitsz.c102c.dataset;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class Dataset {
	// ��������
	private List<Record> records;
	// ����±�
	private int lableIndex;

	private double maxLable = -1;

	public Dataset(int classIndex) {

		this.lableIndex = classIndex;
		records = new ArrayList<Record>();
	}

	public Dataset(List<double[]> datas) {
		this();
		for (double[] data : datas) {
			append(new Record(data));
		}
	}

	private Dataset() {
		this.lableIndex = -1;
		records = new ArrayList<Record>();
	}

	public int size() {
		return records.size();
	}

	public int getLableIndex() {
		return lableIndex;
	}

	public void append(Record record) {
		records.add(record);
	}

	/**
	 * �������
	 */
	public void clear() {
		records.clear();
	}

	/**
	 * ���һ����¼
	 * 
	 * @param attrs
	 *            ��¼������
	 * @param label
	 *            ��¼�����
	 */
	public void append(double[] attrs, Double label) {
		records.add(new Record(attrs, label));
	}

	public Iterator<Record> iter() {
		return records.iterator();
	}

	/**
	 * ��ȡ��index����¼������
	 * 
	 * @param index
	 * @return
	 */
	public double[] getAttrs(int index) {
		return records.get(index).getAttrs();
	}

	public Double getLable(int index) {
		return records.get(index).getLable();
	}

	/**
	 * �������ݼ�
	 * 
	 * @param filePath
	 *            �ļ�����·��
	 * @param tag
	 *            �ֶηָ���
	 * @param lableIndex
	 *            ����±꣬��0��ʼ
	 * @return
	 */
	public static Dataset load(String filePath, String tag, int lableIndex) {
		Dataset dataset = new Dataset();
		dataset.lableIndex = lableIndex;
		File file = new File(filePath);
		try {

			BufferedReader in = new BufferedReader(new FileReader(file));
			String line;
			while ((line = in.readLine()) != null) {
				String[] datas = line.split(tag);
				if (datas.length == 0)
					continue;
				double[] data = new double[datas.length];
				for (int i = 0; i < datas.length; i++)
					data[i] = Double.parseDouble(datas[i]);
				Record record = dataset.new Record(data);
				dataset.append(record);
			}
			in.close();

		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		System.out.println("��������:" + dataset.size());
		return dataset;
	}

	/**
	 * ���ݼ�¼(ʵ��),��¼�����Ժ�������,������Ϊ��һ�л������һ�л��߿�
	 * 
	 * @author jiqunpeng
	 * 
	 *         ����ʱ�䣺2014-6-15 ����8:03:29
	 */
	public class Record {
		// �洢����
		private double[] attrs;
		private Double label;

		private Record(double[] attrs, Double label) {
			this.attrs = attrs;
			this.label = label;
		}

		public Record(double[] data) {
			if (lableIndex == -1)
				attrs = data;
			else {
				label = data[lableIndex];
				if (label > maxLable)
					maxLable = label;
				if (lableIndex == 0)
					attrs = Arrays.copyOfRange(data, 1, data.length);
				else
					attrs = Arrays.copyOfRange(data, 0, data.length - 1);
			}
		}

		/**
		 * �ü�¼������
		 * 
		 * @return
		 */
		public double[] getAttrs() {
			return attrs;
		}

		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append("attrs:");
			sb.append(Arrays.toString(attrs));
			sb.append("label:");
			sb.append(label);
			return sb.toString();
		}

		/**
		 * �ü�¼�����
		 * 
		 * @return
		 */
		public Double getLable() {
			if (lableIndex == -1)
				return null;
			return label;
		}

		/**
		 * �������ж����Ʊ���
		 * 
		 * @param n
		 * @return
		 */
		public int[] getEncodeTarget(int n) {
			String binary = Integer.toBinaryString(label.intValue());
			byte[] bytes = binary.getBytes();
			int[] encode = new int[n];
			int j = n;
			for (int i = bytes.length - 1; i >= 0; i--)
				encode[--j] = bytes[i] - '0';

			return encode;
		}

		public double[] getDoubleEncodeTarget(int n) {
			String binary = Integer.toBinaryString(label.intValue());
			byte[] bytes = binary.getBytes();
			double[] encode = new double[n];
			int j = n;
			for (int i = bytes.length - 1; i >= 0; i--)
				encode[--j] = bytes[i] - '0';

			return encode;
		}

	}

	public static void main(String[] args) {
		Dataset d = new Dataset();
		d.lableIndex = 10;
		Record r = d.new Record(new double[] { 3, 2, 2, 5, 4, 5, 3, 11, 3, 12,
				1 });
		int[] encode = r.getEncodeTarget(4);

		System.out.println(r.label);
		System.out.println(Arrays.toString(encode));
	}

	/**
	 * ��ȡ��index����¼
	 * 
	 * @param index
	 * @return
	 */
	public Record getRecord(int index) {
		return records.get(index);
	}

}
