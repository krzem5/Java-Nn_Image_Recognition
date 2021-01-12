package com.krzem.nn_image_recognition;



import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.lang.Exception;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;



public class MNISTDatasetLoader{
	public static Dataset load(String fp){
		try{
			Map<String,List<double[]>> data=new HashMap<String,List<double[]>>();
			BufferedReader br=new BufferedReader(new FileReader(new File(fp)));
			String _l;
			for (int i=0;i<10;i++){
				data.put(Integer.toString(i),new ArrayList<double[]>());
			}
			int sz=0;
			while ((_l=br.readLine())!=null){
				String[] l=_l.split(",");
				double[] o=new double[32*32];
				for (int i=2;i<30;i++){
					for (int j=2;j<30;j++){
						o[i*32+j]=Integer.parseInt(l[(i-2)*28+j-1])/255d;
					}
				}
				sz++;
				data.get(l[0]).add(o);
			}
			br.close();
			return new Dataset(data,sz);
		}
		catch (Exception e){
			e.printStackTrace();
		}
		return null;
	}
}