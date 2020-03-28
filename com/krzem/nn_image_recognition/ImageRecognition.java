package com.krzem.nn_image_recognition;



import com.krzem.NN.NeuralNetwork;
import java.io.File;



public class ImageRecognition{
	private NeuralNetwork nn;



	public ImageRecognition(String fp){
		if (new File(fp).exists()){
			this.nn=NeuralNetwork.fromFile(fp);
		}
		else{
			this.nn=new NeuralNetwork(28*28,new int[]{28*29},2,0.01);
		}
	}



	public int predict(double[] _i){
		double[] o=this.nn.predict(_i);
		double m=-Double.MAX_VALUE;
		int b=-1;
		for (int i=0;i<o.length;i++){
			if (o[i]>m){
				m=o[i];
				b=i+0;
			}
		}
		return b;
	}



	public void train(Dataset d,Dataset td,int tbs,int itr,String ef){
		this.nn.train_multiple(d.data(),td.data(),itr,0,10000,true,ef);
	}



	public double acc(Dataset d){
		return this.nn.acc(d.data());
	}



	public void save(String fp){
		this.nn.toFile(fp);
	}
}