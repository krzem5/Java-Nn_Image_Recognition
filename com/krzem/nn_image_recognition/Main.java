package com.krzem.nn_image_recognition;



public class Main{
	public static void main(String[] args){
		new Main();
	}



	public Main(){
		ImageRecognition ir=new ImageRecognition("./data/full.nn-data");
		Dataset tr=DatasetLoader.load("D:\\K\\Project\\project2\\DATA\\final\\train\\");
		Dataset ts=DatasetLoader.load("D:\\K\\Project\\project2\\DATA\\final\\test\\");
		System.out.printf("Memory Usage: %,db\n",Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory());
		ir.train(tr,ts,64,100,"./end.txt");
		// System.out.println(ir.acc(ts));
		ir.save("./data/full.nn-data");
	}
}