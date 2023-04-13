package com.krzem.nn_image_recognition;



public class Random{
	public static final int SEED=12345;
	private static java.util.Random RG=new java.util.Random(SEED);



	public static double next(){
		return Random.RG.nextGaussian();
	}
}
