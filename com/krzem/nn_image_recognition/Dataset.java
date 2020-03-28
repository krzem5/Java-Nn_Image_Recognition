package com.krzem.nn_image_recognition;



import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;



public class Dataset{
	private double[][][] data;
	private Map<String,List<double[]>> m_data;



	public Dataset(Map<String,List<double[]>> m_data,List<String> ll,int sz){
		this.m_data=m_data;
		this._load(ll,sz);
	}



	public int size(){
		return this.data.length;
	}



	public double[][][] data(){
		return this.data;
	}



	public List<String> labels(){
		List<String> o=new ArrayList<String>();
		for (String k:this.m_data.keySet()){
			o.add(k);
		}
		return o;
	}



	public List<double[]> get(String l){
		return this.m_data.get(l);
	}



	public int[] get_r_batch_idx(int bs){
		int[] o=new int[bs];
		Arrays.fill(o,-1);
		for (int i=0;i<bs;i++){
			while (true){
				boolean n=true;
				int j=(int)Math.abs(Math.floor(Math.min(Math.abs(Random.next()),1)*this.size())-1);
				for (int k=0;k<i;k++){
					if (o[k]==j){
						n=false;
						break;
					}
				}
				if (n==true){
					o[i]=j;
					break;
				}
			}
		}
		return o;
	}



	private void _load(List<String> ll,int sz){
		this.data=new double[sz][2][];
		int i=0;
		for (String l:ll){
			double[] ld=new double[ll.size()];
			for (int j=0;j<ll.size();j++){
				if (ll.get(j).equals(l)){
					ld[j]=1;
					break;
				}
			}
			for (double[] k:this.m_data.get(l)){
				this.data[i]=new double[][]{k,ld};
				i++;
			}
		}
	}
}