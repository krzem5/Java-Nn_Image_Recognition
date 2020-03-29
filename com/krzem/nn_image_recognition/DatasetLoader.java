package com.krzem.nn_image_recognition;



import java.awt.image.BufferedImage;
import java.io.File;
import java.lang.Exception;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;



public class DatasetLoader{
	public static Dataset load(String fp){
		try{
			Map<String,List<double[]>> data=new HashMap<String,List<double[]>>();
			int sz=0;
			for (String lb:new String[]{"empty","fish"}){
				data.put(lb,new ArrayList<double[]>());
				for (File f:new File(fp+lb+"\\").listFiles()){
					BufferedImage i=ImageIO.read(f);
					double[] dt=new double[28*28];
					for (int j=0;j<28;j++){
						for (int k=0;k<28;k++){
							dt[j*28+k]=((i.getRGB(j,k)&0xff)/255d==0?0:1);
						}
					}
					sz++;
					data.get(lb).add(dt);
				}
			}
			return new Dataset(data,Arrays.asList("empty","fish"),sz);
		}
		catch (Exception e){
			e.printStackTrace();
		}
		return null;
	}
}